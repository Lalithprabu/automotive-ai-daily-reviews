"""
Trajectory-Adaptive Bias (TAB).

Per the abstract: "BEV trajectories are misaligned with image tokens, [so]
we use ... Trajectory-Adaptive Bias (TAB) to inject future latents as
guidance without overriding the current observation."

The problem TAB addresses: the DiT planner's conditioning tokens (current +
gated future visual tokens, see models/fusion.py) live in an ego-centric BEV
occupancy grid's token space, while the trajectory being denoised is a set
of continuous (x, y) waypoints. A trajectory near the road edge should
probably attend more to the visual tokens covering that region; a trajectory
that currently predicts hard braking should probably attend more to the
near-horizon future tokens where a lead-vehicle stop event would show up.
A FIXED (trajectory-independent) learned attention bias cannot express this
— it cannot know "where the candidate trajectory currently is" — so TAB is
instead COMPUTED FRESH, at every denoising step, from the current (noisy)
candidate trajectory.

RECONSTRUCTION DISCLOSURE: the exact TAB formulation is not published at the
abstract level we have access to. We implement TAB as: project each
trajectory waypoint into the same embedding space as the fused conditioning
tokens, then take a scaled dot product between the two to produce a
per-(waypoint, conditioning-token) bias, added directly into the DiT cross-
attention logits before softmax (see models/dit_planner.py, which passes
this bias in as `attn_mask` to `nn.MultiheadAttention` — PyTorch adds a
float `attn_mask` to the raw attention scores prior to softmax, which is
exactly the "bias added to attention logits" mechanism described in the
task spec). This is our own design choice, not the paper's formula.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TrajectoryAdaptiveBias(nn.Module):
    """Computes a per-(waypoint, conditioning-token) additive attention bias
    from the CURRENT (possibly noisy) candidate trajectory.

    Shapes:
      trajectory       : (B, T, 2)                 T waypoints, (x, y) each
      fused_tokens       : (B, S, fused_token_dim)     S = N * (1 + H) conditioning tokens
      returns              : (B, T, S)                  additive bias, NOT yet
                                                            expanded across attention heads
                                                            (the DiT cross-attention module
                                                            repeats it per head before use)
    """

    def __init__(self, fused_token_dim: int, num_waypoints: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Per-waypoint positional index embedding: the bias should be able to
        # depend on WHICH waypoint (near-term vs. far-term) is being placed,
        # not only on its raw (x, y) value.
        self.waypoint_pos_embed = nn.Parameter(torch.randn(num_waypoints, hidden_dim) * 0.02)

        # Project each (x, y) waypoint -> hidden_dim, then add the positional
        # embedding, giving a "trajectory query" in the shared bias space.
        self.traj_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Project the fused conditioning tokens into the SAME hidden_dim space
        # ("project ... into the same space as the fused visual/future
        # tokens" per the task spec) so a dot product is a meaningful
        # compatibility score.
        self.cond_key_proj = nn.Linear(fused_token_dim, hidden_dim)

        self.scale = 1.0 / math.sqrt(hidden_dim)

    def forward(self, trajectory: torch.Tensor, fused_tokens: torch.Tensor) -> torch.Tensor:
        b, t, _ = trajectory.shape
        traj_emb = self.traj_proj(trajectory) + self.waypoint_pos_embed.unsqueeze(0)  # (B, T, hidden)
        cond_key = self.cond_key_proj(fused_tokens)                                       # (B, S, hidden)

        # Scaled dot-product compatibility between each waypoint and each
        # conditioning token -> (B, T, S). This is added as a bias into the
        # DiT's real cross-attention logits (see dit_planner.py), so it
        # directly reshapes which conditioning tokens each waypoint attends
        # to, conditioned on where that waypoint currently sits in the
        # denoising trajectory (i.e. it changes every diffusion step).
        bias = torch.einsum("bth,bsh->bts", traj_emb, cond_key) * self.scale  # (B, T, S)
        return bias
