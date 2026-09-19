"""Bidirectional conditional-flow-matching transformer.

Paper mechanism: "A modality-aware Transformer applies conditional flow
matching to co-evolve action and scene streams. Historical MM-Tokens form a
clean prefix. Future tokens progressively move toward targets through
interpolation: z_t = (1-t)z0 + t*z1. The attention structure is asymmetric:
historical queries see only history (preserving causality), while future
queries access both history and the developing alternative stream. Separate
normalization and feed-forward branches maintain modality-specific
processing while shared attention enables cross-stream communication."

This module implements exactly that: one shared self-attention over the
concatenated [history | action | scene] sequence under an asymmetric mask,
followed by three *separate* per-modality FFN branches.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of the flow-matching interpolation time t in [0, 1]."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] in [0, 1] -> [B, d_model]
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0) * 2 * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.d_model:
            emb = torch.nn.functional.pad(emb, (0, self.d_model - emb.shape[-1]))
        return self.mlp(emb)


class _FFNBranch(nn.Module):
    def __init__(self, d_model: int, mult: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * mult), nn.GELU(), nn.Linear(d_model * mult, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class AsymmetricFlowBlock(nn.Module):
    """One transformer block: shared self-attention under an asymmetric
    causal-ish mask, then three modality-specific FFN branches."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.hist_ffn = _FFNBranch(d_model)
        self.action_ffn = _FFNBranch(d_model)
        self.scene_ffn = _FFNBranch(d_model)

    def forward(self, seq: torch.Tensor, hist_len: int, action_len: int,
                scene_len: int, attn_mask: torch.Tensor) -> torch.Tensor:
        normed = self.pre_norm(seq)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask, need_weights=False)
        seq = seq + attn_out
        h, a, s = torch.split(seq, [hist_len, action_len, scene_len], dim=1)
        h = self.hist_ffn(h)
        a = self.action_ffn(a)
        s = self.scene_ffn(s)
        return torch.cat([h, a, s], dim=1)


def build_asymmetric_mask(hist_len: int, action_len: int, scene_len: int,
                           device) -> torch.Tensor:
    """True = attention disallowed.
    History rows may only attend to history columns.
    Action/scene (future) rows may attend to everything (history + future),
    which is how the two future streams "communicate" with each other."""
    total = hist_len + action_len + scene_len
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    future_slice = slice(hist_len, total)
    hist_slice = slice(0, hist_len)
    mask[hist_slice, future_slice] = True
    return mask


class BidirectionalFlowTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, action_dim: int):
        super().__init__()
        self.d_model = d_model
        self.time_embed = TimeEmbedding(d_model)
        self.action_in_proj = nn.Linear(action_dim, d_model)
        self.action_out_proj = nn.Linear(d_model, action_dim)
        self.scene_out_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            [AsymmetricFlowBlock(d_model, n_heads) for _ in range(n_layers)]
        )

    def forward(self, history_tokens: torch.Tensor, action_zt: torch.Tensor,
                scene_zt: torch.Tensor, t: torch.Tensor):
        """
        history_tokens: [B, Th, D]  (clean prefix, no time-conditioning)
        action_zt:      [B, Ta, action_dim]  (noised action tokens at time t)
        scene_zt:       [B, Ts, D]           (noised scene tokens at time t)
        t:              [B]  interpolation time in [0, 1]
        returns velocity predictions (v_action: [B, Ta, action_dim], v_scene: [B, Ts, D])
        """
        b, th, _ = history_tokens.shape
        ta = action_zt.shape[1]
        ts = scene_zt.shape[1]
        te = self.time_embed(t)  # [B, D]

        a_tok = self.action_in_proj(action_zt) + te.unsqueeze(1)
        s_tok = scene_zt + te.unsqueeze(1)
        seq = torch.cat([history_tokens, a_tok, s_tok], dim=1)  # [B, Th+Ta+Ts, D]

        mask = build_asymmetric_mask(th, ta, ts, seq.device)
        for block in self.blocks:
            seq = block(seq, th, ta, ts, mask)

        h, a, s = torch.split(seq, [th, ta, ts], dim=1)
        v_action = self.action_out_proj(a)   # [B, Ta, action_dim]
        v_scene = self.scene_out_proj(s)      # [B, Ts, D]
        return v_action, v_scene
