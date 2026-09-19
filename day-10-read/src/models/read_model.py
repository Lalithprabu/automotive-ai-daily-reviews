"""READModel: top-level wiring of SceneEncoder + RiskFieldNetwork, plus
the training loss and the trajectory-planning cost used at inference.

Reconstruction note on the loss: the paper's exact loss formula was not
recoverable from the sources available (see README "Sourcing note").
`risk_ranking_loss` below is our own well-commented reconstruction of
what the paper's qualitative description implies -- a margin/ranking
term that pushes agent-proximal probe risk above background probe risk,
plus the background regularizer described in `positional_encoding.py`
that pulls background risk toward 0 directly (this is what actually
fixes the checkerboard-overfitting bug, and the ranking term alone is
not sufficient without it).
"""

import torch
import torch.nn as nn

from src.models.scene_encoder import SceneEncoder
from src.models.risk_field_net import RiskFieldNetwork


class READModel(nn.Module):
    """Wires the SceneEncoder and RiskFieldNetwork into a single module.

    forward(bev_grid, agent_history, query_xyt) -> risk [B, Q]
    """

    def __init__(
        self,
        bev_channels: int = 3,
        agent_features: int = 4,
        hidden_dim: int = 128,
        pooled_size: int = 4,
        num_scene_layers: int = 2,
        num_scene_heads: int = 4,
        num_freqs: int = 8,
        num_cross_layers: int = 2,
        num_risk_heads: int = 4,
    ):
        super().__init__()
        self.scene_encoder = SceneEncoder(
            bev_channels=bev_channels,
            agent_features=agent_features,
            hidden_dim=hidden_dim,
            pooled_size=pooled_size,
            num_transformer_layers=num_scene_layers,
            num_heads=num_scene_heads,
        )
        self.risk_field = RiskFieldNetwork(
            hidden_dim=hidden_dim,
            num_freqs=num_freqs,
            num_cross_layers=num_cross_layers,
            num_heads=num_risk_heads,
        )

    def encode_scene(self, bev_grid: torch.Tensor, agent_history: torch.Tensor) -> torch.Tensor:
        """Returns scene_tokens [B, N, D] -- exposed separately so
        simulate.py / refine_trajectory_by_risk_descent can reuse a
        single scene encoding across many risk-field queries."""
        return self.scene_encoder(bev_grid, agent_history)

    def forward(self, bev_grid: torch.Tensor, agent_history: torch.Tensor,
                query_xyt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_grid: [B, C, H, W]
            agent_history: [B, A, T, F]
            query_xyt: [B, Q, 3]
        Returns:
            risk: [B, Q]
        """
        scene_tokens = self.encode_scene(bev_grid, agent_history)
        risk = self.risk_field(query_xyt, scene_tokens)
        return risk


def risk_ranking_loss(
    model: READModel,
    bev_grid: torch.Tensor,
    agent_history: torch.Tensor,
    agent_probe_xyt: torch.Tensor,
    background_probe_xyt: torch.Tensor,
    margin: float = 0.5,
    background_weight: float = 1.0,
) -> torch.Tensor:
    """Reconstructed training loss for the risk field.

    Two terms:
      1. Ranking/margin term: agent-proximal probes should score higher
         risk than background probes by at least `margin` (a hinge loss
         on the mean agent-probe risk vs. mean background-probe risk,
         per scene). This is what teaches the field *where* danger is.
      2. Background regularizer (`background_weight`): directly pulls
         risk at background probes toward 0 with a plain MSE term. This
         is the fix for the Fourier-encoding overfitting/checkerboard
         bug described in `positional_encoding.py` -- without it, the
         high-frequency coordinate MLP is free to hallucinate risk in
         the (x, y, t) gaps between sparse agent probes, since the
         ranking term alone only constrains *relative* risk, not
         absolute risk far from any supervision signal.

    Args:
        model: a READModel.
        bev_grid: [B, C, H, W]
        agent_history: [B, A, T, F]
        agent_probe_xyt: [B, P_a, 3] probes near agents (should be risky).
        background_probe_xyt: [B, P_b, 3] random probes (should be safe).
        margin: hinge margin between agent-probe and background-probe risk.
        background_weight: weight on the background MSE-to-zero term.
    Returns:
        scalar loss.
    """
    scene_tokens = model.encode_scene(bev_grid, agent_history)

    agent_risk = model.risk_field(agent_probe_xyt, scene_tokens)  # [B, P_a]
    background_risk = model.risk_field(background_probe_xyt, scene_tokens)  # [B, P_b]

    # Ranking/margin term: mean agent risk should exceed mean background
    # risk by at least `margin`.
    ranking_term = torch.clamp(margin - (agent_risk.mean(dim=-1) - background_risk.mean(dim=-1)), min=0.0)
    ranking_loss = ranking_term.mean()

    # Background regularizer: pulls background risk directly toward 0.
    background_loss = (background_risk ** 2).mean()

    total_loss = ranking_loss + background_weight * background_loss
    return total_loss


def trajectory_planning_cost(
    model: READModel,
    bev_grid: torch.Tensor,
    agent_history: torch.Tensor,
    trajectory_xy: torch.Tensor,
    timestamps: torch.Tensor,
) -> torch.Tensor:
    """Integrates the learned risk field along a candidate trajectory --
    the quantity a downstream planner/VLA would minimize (directly, via
    gradient descent on the waypoints, or as one term in a larger
    trajectory-scoring objective alongside comfort/progress costs).

    Args:
        model: a READModel.
        bev_grid: [B, C, H, W]
        agent_history: [B, A, T, F]
        trajectory_xy: [B, Q, 2] candidate waypoints.
        timestamps: [B, Q] timestamp per waypoint.
    Returns:
        cost: [B]  mean risk integrated along the trajectory.
    """
    scene_tokens = model.encode_scene(bev_grid, agent_history)
    query_xyt = torch.cat([trajectory_xy, timestamps.unsqueeze(-1)], dim=-1)  # [B, Q, 3]
    risk = model.risk_field(query_xyt, scene_tokens)  # [B, Q]
    cost = risk.mean(dim=-1)  # [B]
    return cost
