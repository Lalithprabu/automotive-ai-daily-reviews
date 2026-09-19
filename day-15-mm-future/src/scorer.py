"""Future-conditioned proposal scorer.

Paper mechanism: "A future-conditioned scorer ranks trajectories using both
shared history and each candidate's paired future scene representation.
Self-attention relates candidates; block-diagonal attention processes
paired futures independently to prevent information leakage between
hypotheses."

The "block-diagonal" part falls out naturally from how MM-Future's flow
transformer already runs: every proposal's future (action + scene) tokens
are generated independently (the proposal axis is a batch axis throughout
`BidirectionalFlowTransformer`), so by construction proposal m's future
tokens never attend to proposal m''s future tokens during generation. This
scorer's only *cross-proposal* step is the final self-attention layer below
— exactly the "self-attention relates candidates" stage.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ProposalScorer(nn.Module):
    def __init__(self, d_model: int, action_dim: int, n_heads: int = 4):
        super().__init__()
        self.action_embed = nn.Linear(action_dim, d_model)
        self.summary_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.proposal_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.score_head = nn.Linear(d_model, 1)

    def forward(self, history_summary: torch.Tensor, action_z1: torch.Tensor,
                scene_z1: torch.Tensor) -> torch.Tensor:
        """
        history_summary: [B, D]                 mean-pooled history MM-Tokens
        action_z1:       [B, M, Ta, action_dim]  generated (t=1) action tokens, per proposal
        scene_z1:        [B, M, Ts, D]           generated (t=1) scene tokens, per proposal
        returns scores:  [B, M]
        """
        b, m = action_z1.shape[0], action_z1.shape[1]
        a_emb = self.action_embed(action_z1).mean(dim=2)   # [B, M, D]
        s_emb = scene_z1.mean(dim=2)                          # [B, M, D]
        hist_rep = history_summary.unsqueeze(1).expand(-1, m, -1)  # [B, M, D]

        combo = torch.cat([hist_rep, a_emb, s_emb], dim=-1)   # [B, M, 3D]
        summary = self.summary_proj(combo)                     # [B, M, D]  (block-diagonal: independent per-proposal so far)

        normed = self.norm(summary)
        attn_out, _ = self.proposal_attn(normed, normed, normed, need_weights=False)  # cross-proposal comparison
        summary = summary + attn_out

        return self.score_head(summary).squeeze(-1)  # [B, M]
