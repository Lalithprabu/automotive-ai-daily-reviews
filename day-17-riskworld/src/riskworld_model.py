"""Top-level RiskWorld model: wires the BEV encoder, temporal actor encoder,
spatial risk field, and flow-guided occupancy evolution modules into a single
forward pass (encode -> fuse actor context -> predict risk field -> flow-warp
+ residual-correct occupancy for each future planning step).

Trajectory-level planning (collision scoring + selective replacement) is
deliberately NOT part of this class -- per the task spec it runs at planning
time on top of this model's outputs (see collision_score.py and
trajectory_selection.py), mirroring the paper's description of the model
producing one occupancy forecast per step that planning-time logic then reuses
across all trajectory candidates.

RECONSTRUCTION NOTE: the specific fusion step (concat BEV + scattered-actor
feature maps, project with a 1x1 conv) and conditioning the flow/residual
heads on the risk field (by concatenating it as an extra channel) are this
repo's own design choices, not paper-verified.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .actor_encoder import TemporalActorEncoder
from .bev_encoder import BEVEncoder
from .flow_occupancy import FlowGuidedOccupancyEvolution
from .risk_field import SpatialRiskField


class RiskWorld(nn.Module):
    """RiskWorld reconstruction: BEV+actor fusion -> spatial risk field ->
    flow-guided occupancy evolution, run autoregressively over a short horizon.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        grid_size = cfg["data"]["grid_size"]
        max_horizon = cfg["data"]["horizon"]

        self.grid_size = grid_size
        self.max_horizon = max_horizon

        self.bev_encoder = BEVEncoder(m["bev_in_channels"], m["bev_feat_channels"])
        self.actor_encoder = TemporalActorEncoder(
            in_features=4,  # (x, y, vx, vy)
            hidden_dim=m["actor_hidden_dim"],
            feat_dim=m["actor_feat_dim"],
            grid_size=grid_size,
        )
        fused_in = m["bev_feat_channels"] + m["actor_feat_dim"]
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_in, m["bev_feat_channels"], kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.risk_field = SpatialRiskField(m["bev_feat_channels"], m["risk_head_hidden"])

        flow_in_channels = m["bev_feat_channels"] + 1  # +1 for the risk-field conditioning channel
        self.occupancy_evolution = FlowGuidedOccupancyEvolution(
            in_channels=flow_in_channels,
            grid_size=grid_size,
            flow_hidden=m["flow_head_hidden"],
            residual_hidden=m["residual_head_hidden"],
            max_horizon=max_horizon,
        )

    def forward(
        self,
        bev_grid: torch.Tensor,
        agent_history: torch.Tensor,
        agent_last_pos: torch.Tensor,
        agent_mask: torch.Tensor,
        prev_occupancy: torch.Tensor,
        horizon: int | None = None,
    ) -> dict:
        """
        Args:
            bev_grid: (B, C_in, H, W) rasterized BEV scene.
            agent_history: (B, A, T, 4) per-agent (x, y, vx, vy) history.
            agent_last_pos: (B, A, 2) each agent's most recent (x, y) in cells.
            agent_mask: (B, A) presence mask, 1 = real agent.
            prev_occupancy: (B, 1, H, W) last-observed occupancy grid (t=0).
            horizon: number of future steps to forecast (defaults to the
                configured max_horizon).
        Returns:
            dict with:
              "risk_field": (B, 1, H, W)
              "forecast_occupancy": (B, horizon, 1, H, W) -- flow-warped +
                  residual-corrected forecast, one map per future step.
              "persistence_occupancy": (B, horizon, 1, H, W) -- naive baseline,
                  the input prev_occupancy simply repeated (unwarped).
              "flows": (B, horizon, 2, H, W)
              "residuals": (B, horizon, 1, H, W)
              "fused_feat": (B, C, H, W) fused BEV+actor features (exposed for
                  inspection/tests).
        """
        horizon = horizon or self.max_horizon
        assert horizon <= self.max_horizon, "requested horizon exceeds module's max_horizon"

        bev_feat = self.bev_encoder(bev_grid)                                  # (B, C_bev, H, W)
        actor_feat_map, _ = self.actor_encoder(agent_history, agent_last_pos, agent_mask)  # (B, C_actor, H, W)
        fused = self.fusion(torch.cat([bev_feat, actor_feat_map], dim=1))         # (B, C_bev, H, W)

        risk_field = self.risk_field(fused)                                          # (B, 1, H, W)
        flow_input_feat = torch.cat([fused, risk_field], dim=1)                        # (B, C_bev+1, H, W)

        occ = prev_occupancy
        forecast_steps, flow_steps, residual_steps = [], [], []
        for t in range(horizon):
            step_out = self.occupancy_evolution(flow_input_feat, occ, step_idx=t)
            occ = step_out["occupancy"]                                                  # feed forward autoregressively
            forecast_steps.append(occ)
            flow_steps.append(step_out["flow"])
            residual_steps.append(step_out["residual"])

        forecast_occupancy = torch.stack(forecast_steps, dim=1)          # (B, horizon, 1, H, W)
        flows = torch.stack(flow_steps, dim=1)                             # (B, horizon, 2, H, W)
        residuals = torch.stack(residual_steps, dim=1)                       # (B, horizon, 1, H, W)
        persistence_occupancy = prev_occupancy.unsqueeze(1).expand(-1, horizon, -1, -1, -1).contiguous()  # (B, horizon, 1, H, W)

        return {
            "risk_field": risk_field,
            "forecast_occupancy": forecast_occupancy,
            "persistence_occupancy": persistence_occupancy,
            "flows": flows,
            "residuals": residuals,
            "fused_feat": fused,
        }
