"""Small encoder-only Transformer used as a stand-in for the paper's Chronos-2.

The paper fine-tunes the 121M-parameter Chronos-2 (encoder-only transformer
with group attention).  That checkpoint is far outside a CPU-only demo, so this
is a ~0.3M-parameter encoder-only transformer with the same *interface*:
(8 observed positions) -> (12 future positions).  All dimensions are this
repo's own reconstruction defaults.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        # Separate q/k/v/o nn.Linear layers so LoRA can wrap each one.
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, d)
        B, T, d = x.shape
        split = lambda t: t.view(B, T, self.h, self.dh).transpose(1, 2)  # (B, h, T, dh)
        out = F.scaled_dot_product_attention(split(self.q(x)), split(self.k(x)), split(self.v(x)))
        return self.o(out.transpose(1, 2).reshape(B, T, d))  # (B, T, d)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ff: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = SelfAttention(d, heads)
        self.fc1, self.fc2 = nn.Linear(d, ff), nn.Linear(ff, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TrajTransformer(nn.Module):
    """(B, obs, 4) canonical history -> (B, horizon, 2) canonical future offsets.

    Inputs/outputs are in units of `scale` metres (default 5 m) for stable training.
    """

    def __init__(self, d: int = 128, heads: int = 4, layers: int = 2, ff: int = 256,
                 obs: int = 8, horizon: int = 12, in_dim: int = 4):
        super().__init__()
        self.horizon = horizon
        self.inp = nn.Linear(in_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, obs, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(Block(d, heads, ff) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, horizon * 2)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.inp(feats) + self.pos          # (B, obs, d)
        for blk in self.blocks:
            x = blk(x)                           # (B, obs, d)
        last = self.ln(x)[:, -1]                 # (B, d) -- last observed step summarises history
        return self.head(last).view(-1, self.horizon, 2)  # (B, horizon, 2)
