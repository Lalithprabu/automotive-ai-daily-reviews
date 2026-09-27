"""Temporal actor context encoder.

RECONSTRUCTION NOTE: the paper's abstract says "temporal actor context" is
combined with BEV features -- it does not specify GRU vs. Transformer, or the
scatter mechanism. This module is this repo's own default: a single-layer GRU
over each tracked agent's recent (x, y, vx, vy) history, followed by
scattering each agent's final hidden state onto its last-known BEV grid cell
(a simple, cheap fusion pattern common in BEV-forecasting literature, not a
paper-verified detail).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalActorEncoder(nn.Module):
    """GRU encoder over per-agent kinematic history, scattered onto the BEV grid.

    Input history:  (B, A, T, F) with F=4 features (x, y, vx, vy) in grid-cell
                     units, and an (B, A) presence mask (1 = real agent,
                     0 = padding slot).
    Output:
        actor_feat_map: (B, D, H, W) -- per-agent context embeddings scattered
                         onto the BEV grid at each agent's most recent
                         position (cells with no agent are zero).
        agent_ctx:      (B, A, D) -- the raw per-agent embeddings, exposed for
                         downstream modules (e.g. trajectory planning could
                         reuse them; unused elsewhere in this reconstruction).
    """

    def __init__(self, in_features: int, hidden_dim: int, feat_dim: int, grid_size: int):
        super().__init__()
        self.grid_size = grid_size
        self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, feat_dim)
        self.feat_dim = feat_dim

    def forward(
        self,
        history: torch.Tensor,
        last_pos_cells: torch.Tensor,
        presence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            history: (B, A, T, F) float tensor of per-agent past kinematics.
            last_pos_cells: (B, A, 2) float tensor, each agent's most recent
                (x, y) position in grid-cell units (used for scattering).
            presence_mask: (B, A) float/bool tensor, 1 where the agent slot is
                a real (non-padding) agent.
        Returns:
            actor_feat_map: (B, feat_dim, H, W)
            agent_ctx: (B, A, feat_dim)
        """
        B, A, T, F = history.shape
        flat = history.reshape(B * A, T, F)              # (B*A, T, F)
        _, h_n = self.gru(flat)                             # h_n: (1, B*A, hidden)
        h_n = h_n.squeeze(0).reshape(B, A, -1)                # (B, A, hidden)
        agent_ctx = self.proj(h_n)                             # (B, A, feat_dim)
        agent_ctx = agent_ctx * presence_mask.unsqueeze(-1)      # zero out padding slots

        H = W = self.grid_size
        actor_feat_map = torch.zeros(B, self.feat_dim, H, W, device=history.device, dtype=agent_ctx.dtype)

        # Scatter each present agent's embedding into its rasterized cell.
        # Loop over the (small, fixed) agent dimension -- A is at most a
        # handful of agents (config.data.max_agents), so this stays cheap and
        # keeps the scatter logic easy to read/verify.
        xs = last_pos_cells[..., 0].round().long().clamp(0, W - 1)  # (B, A)
        ys = last_pos_cells[..., 1].round().long().clamp(0, H - 1)  # (B, A)
        for b in range(B):
            for a in range(A):
                if presence_mask[b, a] > 0:
                    actor_feat_map[b, :, ys[b, a], xs[b, a]] = agent_ctx[b, a]

        return actor_feat_map, agent_ctx
