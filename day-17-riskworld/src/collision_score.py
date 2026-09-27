"""Collision-score module.

Paper-sourced facts: each trajectory candidate receives "a nonnegative
collision-score correction" via comparison against "a current-state
persistence reference" -- i.e. the naive baseline is simply "the last
observed occupancy grid, held static / unwarped, for the whole horizon", and
RiskWorld's own (flow-warped + residual-corrected) forecast occupancy is
compared against that naive baseline: correction = max(0, risk_under_forecast
- risk_under_persistence). This captures "how much *additional* risk the more
realistic forecast reveals that the naive baseline would have missed."

RECONSTRUCTION NOTE: the exact scoring function (bilinear-sampling occupancy
+ risk field along each candidate trajectory's waypoints, then averaging over
the horizon with fixed combination weights) is this repo's own default -- the
abstract does not specify how occupancy/risk are turned into a scalar
per-candidate score.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import grid_to_norm


class CollisionScoreModule(nn.Module):
    """Scores candidate ego trajectories against forecast occupancy + risk field,
    and derives a nonnegative collision-score correction vs. a persistence
    (static, unwarped last-observed-occupancy) baseline.
    """

    def __init__(self, grid_size: int, occ_weight: float = 1.0, risk_weight: float = 1.0):
        super().__init__()
        self.grid_size = grid_size
        self.occ_weight = occ_weight
        self.risk_weight = risk_weight

    def sample_at_positions(self, grid_map: torch.Tensor, positions_cells: torch.Tensor) -> torch.Tensor:
        """Bilinearly samples a (B, 1, H, W) map at K query positions per batch.

        Args:
            grid_map: (B, 1, H, W)
            positions_cells: (B, K, 2) positions in grid-cell coordinates (x, y).
        Returns:
            (B, K) sampled scalar values.
        """
        B, K, _ = positions_cells.shape
        norm_pos = grid_to_norm(positions_cells, self.grid_size)   # (B, K, 2), in [-1, 1]
        sample_grid = norm_pos.unsqueeze(2)                          # (B, K, 1, 2) -- grid_sample wants (B, Hout, Wout, 2)
        sampled = F.grid_sample(
            grid_map, sample_grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # (B, 1, K, 1)
        return sampled.squeeze(1).squeeze(-1)  # (B, K)

    def forward(
        self,
        forecast_occupancy: torch.Tensor,
        persistence_occupancy: torch.Tensor,
        risk_field: torch.Tensor,
        candidate_trajs: torch.Tensor,
    ) -> dict:
        """
        Args:
            forecast_occupancy: (B, horizon, 1, H, W) RiskWorld's flow-warped +
                residual-corrected occupancy forecast, one map per future step.
            persistence_occupancy: (B, horizon, 1, H, W) the naive baseline --
                the last-observed occupancy grid repeated (unwarped) for every
                future step.
            risk_field: (B, 1, H, W) static spatial risk field (computed once
                per planning step, shared across the horizon and all
                candidates -- mirrors the abstract's "reused across
                candidates").
            candidate_trajs: (B, K, horizon, 2) candidate ego waypoints in
                grid-cell coordinates.
        Returns:
            dict with:
              "forecast_score": (B, K) risk score under RiskWorld's forecast.
              "persistence_score": (B, K) risk score under the persistence baseline.
              "correction": (B, K) nonnegative collision-score correction,
                             = max(0, forecast_score - persistence_score).
        """
        B, K, horizon, _ = candidate_trajs.shape
        forecast_terms = []
        persistence_terms = []
        for t in range(horizon):
            pos_t = candidate_trajs[:, :, t, :]                       # (B, K, 2)
            occ_f = self.sample_at_positions(forecast_occupancy[:, t], pos_t)      # (B, K)
            occ_p = self.sample_at_positions(persistence_occupancy[:, t], pos_t)     # (B, K)
            risk_t = self.sample_at_positions(risk_field, pos_t)                       # (B, K), same field reused every step

            forecast_terms.append(self.occ_weight * occ_f + self.risk_weight * risk_t)
            persistence_terms.append(self.occ_weight * occ_p + self.risk_weight * risk_t)

        forecast_score = torch.stack(forecast_terms, dim=0).mean(dim=0)        # (B, K)
        persistence_score = torch.stack(persistence_terms, dim=0).mean(dim=0)    # (B, K)
        correction = torch.clamp(forecast_score - persistence_score, min=0.0)      # (B, K), nonnegative by construction

        return {
            "forecast_score": forecast_score,
            "persistence_score": persistence_score,
            "correction": correction,
        }
