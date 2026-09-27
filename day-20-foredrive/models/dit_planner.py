"""
Diffusion Transformer (DiT) trajectory planner.

Per the abstract: "We propose ForeDrive, which ... couples [the latent world
model] asymmetrically to a Diffusion Transformer (DiT) planner." This module
is that planner: a small conditional denoising transformer over ego future
trajectory waypoints, conditioned on the fused visual + future-latent tokens
(models/fusion.py) via cross-attention, with Trajectory-Adaptive Bias
(models/tab.py) injected directly into that cross-attention's logits.

DESIGN CHOICES (all ours — not specified in the abstract, see SOURCING.md):
  - Objective: DDPM-style EPSILON-PREDICTION (predict the noise added to the
    clean trajectory), trained with a random-timestep MSE loss. We chose
    epsilon-prediction over direct x0-prediction because it is the more
    standard/well-conditioned DDPM objective and keeps the sampler
    (src/utils/diffusion.py) simple.
  - Conditioning mechanism: rather than full adaLN-Zero (which needs a
    separate scale/shift/gate triple per sub-layer), we use (a) an additive
    sinusoidal diffusion-timestep embedding broadcast onto every waypoint
    token before self-attention, and (b) real cross-attention from waypoint
    tokens to the fused conditioning tokens, with TAB injected as an
    additive bias on that cross-attention's logits. This is simpler to
    implement correctly and small enough to train on CPU while still being
    a real, non-pseudocode Transformer block stack (num_layers of them).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.diffusion import sinusoidal_timestep_embedding
from models.tab import TrajectoryAdaptiveBias


class DiTBlock(nn.Module):
    """One DiT-style block: self-attention over waypoint tokens, then
    TAB-biased cross-attention to the fused conditioning tokens, then an MLP.
    Pre-norm residual structure throughout."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm3 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, tab_bias: torch.Tensor) -> torch.Tensor:
        """
        x        : (B, T, hidden_dim)   waypoint tokens
        cond      : (B, S, hidden_dim)    fused conditioning tokens (already
                                            projected to hidden_dim by the caller)
        tab_bias   : (B, T, S)              additive bias for the cross-attention
                                              logits (see models/tab.py); this
                                              method repeats it across attention
                                              heads before handing it to
                                              nn.MultiheadAttention as `attn_mask`
        """
        # --- self-attention among waypoint tokens ---
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + attn_out

        # --- TAB-biased cross-attention to fused visual/future tokens ---
        h = self.norm2(x)
        b, t, _ = h.shape
        s = cond.shape[1]
        # nn.MultiheadAttention expects attn_mask shaped (B*num_heads, T, S) for a
        # per-batch-element float bias; we broadcast the same TAB bias to every
        # head (TAB itself is head-agnostic in our reconstruction — see tab.py).
        attn_mask = tab_bias.unsqueeze(1).expand(b, self.num_heads, t, s).reshape(b * self.num_heads, t, s)
        cross_out, cross_weights = self.cross_attn(h, cond, cond, attn_mask=attn_mask, need_weights=False)
        x = x + cross_out

        # --- feedforward ---
        h = self.norm3(x)
        x = x + self.mlp(h)

        return x


class DiTPlanner(nn.Module):
    """Full DiT planner: embeds noisy waypoints + diffusion timestep, runs
    `num_layers` DiTBlocks (self-attn + TAB cross-attn + MLP), and projects
    back to a per-waypoint (dx, dy) noise prediction.

    Shapes:
      noisy_trajectory : (B, T, 2)
      timestep           : (B,)                integer diffusion step indices
      fused_tokens        : (B, S, fused_token_dim)
      returns               : eps_pred (B, T, 2)
    """

    def __init__(self, fused_token_dim: int, num_waypoints: int, hidden_dim: int,
                 num_layers: int, num_heads: int, dropout: float, tab_hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_waypoints = num_waypoints

        self.waypoint_embed = nn.Linear(2, hidden_dim)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_proj = nn.Linear(fused_token_dim, hidden_dim)

        self.tab = TrajectoryAdaptiveBias(fused_token_dim=hidden_dim, num_waypoints=num_waypoints,
                                            hidden_dim=tab_hidden_dim)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 2)

    def forward(self, noisy_trajectory: torch.Tensor, timestep: torch.Tensor,
                fused_tokens: torch.Tensor) -> torch.Tensor:
        b, t, _ = noisy_trajectory.shape

        x = self.waypoint_embed(noisy_trajectory)                                  # (B, T, hidden)
        t_emb = sinusoidal_timestep_embedding(timestep, self.hidden_dim)               # (B, hidden)
        t_emb = self.timestep_mlp(t_emb)                                                 # (B, hidden)
        x = x + t_emb.unsqueeze(1)                                                          # broadcast over T

        cond = self.cond_proj(fused_tokens)                                                    # (B, S, hidden)

        # TAB bias is recomputed from the CURRENT noisy trajectory at every call
        # (i.e. at every denoising step during sampling), per the module's
        # docstring: attention adapts to where the candidate trajectory
        # currently is, not a fixed learned bias.
        tab_bias = self.tab(noisy_trajectory, cond)                                              # (B, T, S)

        for block in self.blocks:
            x = block(x, cond, tab_bias)

        x = self.out_norm(x)
        eps_pred = self.out_proj(x)   # (B, T, 2)
        return eps_pred
