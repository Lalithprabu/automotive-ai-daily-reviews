"""CAA-GMM reconstruction: Context-Aware Attention-based Gaussian Mixture Model (PyTorch).

Paper-sourced (abstract level only): raster map/environment cues + agent motion history are fused by a
*lightweight attention mechanism* that weighs (a) inter-agent interactions and (b) the relative importance of
contextual cues; the output is a Gaussian mixture, one interpretable Gaussian component per future mode.
Everything else here (dims, layer layout, 3-way gate, per-mode head, null token) is this project's own
reconstruction choice -- see SOURCING.md.

Tensor shapes (B=batch, N=neighbour slots=6, T_H=8, T_F=16, K=6 modes, d=64):
  raster (B,4,64,64)  ego (B,8,4)  nb (B,N,8,4)  nb_mask (B,N) bool
  -> logits (B,K)   mu (B,K,T_F,2) [m, ego frame]   log_sigma (B,K,T_F,2)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import N_NB, T_F, T_H


@dataclass
class CAAConfig:
    d_model: int = 64
    n_heads: int = 4
    n_modes: int = 6
    t_future: int = T_F
    in_ch: int = 4


class RasterEncoder(nn.Module):
    """3 strided convs: (B,C,64,64) -> 8x8 grid -> (B,64,c3) tokens (+ learned positional embedding)."""

    def __init__(self, in_ch=4, c1=16, c2=32, c3=48):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.GroupNorm(4, o), nn.ReLU(inplace=True))
        self.net = nn.Sequential(blk(in_ch, c1), blk(c1, c2), blk(c2, c3))       # 64 -> 32 -> 16 -> 8
        self.pos = nn.Parameter(torch.zeros(64, c3))
        nn.init.normal_(self.pos, std=0.02)
        self.c3 = c3

    def grid(self, raster):                                  # (B,C,64,64) -> (B,c3,8,8)
        return self.net(raster)

    def forward(self, raster):                               # -> (B,64,c3)
        g = self.grid(raster)
        return g.flatten(2).transpose(1, 2) + self.pos


class GMMHead(nn.Module):
    """Shared decoder: fused (B,d) -> K Gaussian components. Mode diversity comes from learned mode embeddings."""

    def __init__(self, d, K, T):
        super().__init__()
        self.K, self.T = K, T
        self.trunk = nn.Sequential(nn.Linear(d, 128), nn.ReLU(inplace=True))
        self.mode_emb = nn.Parameter(torch.randn(K, 128) * 0.5)
        self.per_mode = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.out = nn.Linear(128, T * 4 + 1)             # T*2 step increments, T*2 log-sigma, 1 logit
        with torch.no_grad():                            # start near "keep driving straight at ~7 m/s"
            self.out.bias.zero_()
            b = self.out.bias[: T * 2].view(T, 2)
            b[:, 1] = 7.0 * 0.25

    def forward(self, f):
        B = f.shape[0]
        h = self.trunk(f)[:, None, :] + self.mode_emb[None]          # (B,K,128)
        h = self.per_mode(h)
        o = self.out(h)                                              # (B,K,T*4+1)
        inc = o[..., : self.T * 2].view(B, self.K, self.T, 2)
        mu = torch.cumsum(inc, dim=2)                                # step displacements -> positions
        log_sigma = o[..., self.T * 2: self.T * 4].view(B, self.K, self.T, 2).clamp(-3.0, 1.5)
        logits = o[..., -1]
        return logits, mu, log_sigma


class CAAGMM(nn.Module):
    """Lightweight attention fusion + GMM head."""

    def __init__(self, cfg: CAAConfig | None = None):
        super().__init__()
        c = cfg or CAAConfig()
        self.cfg = c
        d = c.d_model
        self.raster_enc = RasterEncoder(c.in_ch)
        self.map_proj = nn.Linear(self.raster_enc.c3, d)
        self.ego_gru = nn.GRU(4, d, batch_first=True)
        self.nb_gru = nn.GRU(4, d, batch_first=True)                 # shared across neighbours
        self.null_tok = nn.Parameter(torch.zeros(1, 1, d))           # always-valid key: no NaN if all dropped
        self.int_attn = nn.MultiheadAttention(d, c.n_heads, batch_first=True)     # agent <- neighbours
        self.map_attn = nn.MultiheadAttention(d, c.n_heads, batch_first=True)     # agent <- raster cells
        self.gate = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(inplace=True), nn.Linear(d, 3))
        self.fuse = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(inplace=True), nn.Linear(2 * d, d))
        self.norm = nn.LayerNorm(d)
        self.head = GMMHead(d, c.n_modes, c.t_future)

    def forward(self, raster, ego, nb, nb_mask, return_attn: bool = False):
        B, N = nb_mask.shape
        d = self.cfg.d_model
        _, h_ego = self.ego_gru(ego)                                 # (1,B,d)
        q = h_ego.transpose(0, 1)                                    # (B,1,d)  agent query

        # --- neighbour tokens (B,1+N,d); slot 0 is the always-valid null token
        _, h_nb = self.nb_gru(nb.reshape(B * N, T_H, 4))             # (1,B*N,d)
        nb_tok = h_nb[0].view(B, N, d)
        keys = torch.cat([self.null_tok.expand(B, 1, d), nb_tok], 1)
        valid = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=nb_mask.device), nb_mask], 1)
        c_int, w_int = self.int_attn(q, keys, keys, key_padding_mask=~valid, need_weights=True)

        # --- map / context tokens (B,64,d)
        m_tok = self.map_proj(self.raster_enc(raster))
        c_map, w_map = self.map_attn(q + c_int, m_tok, m_tok, need_weights=True)   # query already knows the traffic

        # --- relative importance of [motion | interaction | context]: 3-way softmax gate
        g = F.softmax(self.gate(torch.cat([q, c_int, c_map], -1)), dim=-1)         # (B,1,3)
        fused = g[..., 0:1] * q + g[..., 1:2] * c_int + g[..., 2:3] * c_map        # (B,1,d)
        fused = self.norm(fused + self.fuse(fused))[:, 0]                          # (B,d)

        logits, mu, log_sigma = self.head(fused)
        if return_attn:
            return logits, mu, log_sigma, dict(w_int=w_int[:, 0, 1:], w_null=w_int[:, 0, 0],
                                               w_map=w_map[:, 0].view(B, 8, 8), gate=g[:, 0])
        return logits, mu, log_sigma


class RasterBaseline(nn.Module):
    """'Old way': raster CNN -> flatten -> MLP, concat with GRU motion state. No attention, no neighbour tokens
    (neighbours reach the model only as pixels in the occupancy channel). Same GMM head => only fusion differs."""

    def __init__(self, cfg: CAAConfig | None = None):
        super().__init__()
        c = cfg or CAAConfig()
        self.cfg = c
        d = c.d_model
        self.raster_enc = RasterEncoder(c.in_ch)
        self.red = nn.Conv2d(self.raster_enc.c3, 16, 1)
        self.map_fc = nn.Linear(16 * 64, d)
        self.ego_gru = nn.GRU(4, d, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.ReLU(inplace=True), nn.Linear(2 * d, d))
        self.head = GMMHead(d, c.n_modes, c.t_future)

    def forward(self, raster, ego, nb=None, nb_mask=None, return_attn: bool = False):
        _, h = self.ego_gru(ego)
        m = F.relu(self.red(self.raster_enc.grid(raster))).flatten(1)       # (B,1024)
        f = self.fuse(torch.cat([h[0], F.relu(self.map_fc(m))], -1))
        out = self.head(f)
        return (*out, {}) if return_attn else out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
