"""NegWAM-style unified world-action model with near-expert negatives (reconstruction from the abstract-level
description of arXiv:2611.08111; every dimension below is THIS repo's choice, not a paper value).

Token layout (B = batch, d = width):
    [ scene tokens S=4 | world queries T | action tokens T | rider tokens R ]
Attention mask (True = blocked):
    scene  -> scene
    world  -> scene + world                (predicts the lead car's future positions)
    action -> scene + world + action       (action generation can read the predicted future: "unified WAM")
    rider  -> scene + itself               (training-only branch)
    NOTHING attends to riders -> dropping them at inference changes no output (tested) => zero inference overhead.

Training objective (see losses.py):
    L = L_flow (action flow matching) + L_world (future lead positions)
        + lam_rep * L_repulse (push generated trajectory away from near-expert negatives)
        + lam_rider * L_rider (rider tokens classify expert=compatible vs negatives=incompatible)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from .world import T

S_TOK = 4          # scene tokens: ego / road / lead / context
N_RIDER = 3        # riders per sample: [expert, neg_long, neg_lat]


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of flow time t in [0,1]. (B,) -> (B, dim)"""
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * 1000.0 * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class Block(nn.Module):
    """Pre-norm self-attention block over the full token sequence (mask decides who sees whom)."""
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class NegWAM(nn.Module):
    def __init__(self, d: int = 96, layers: int = 4, heads: int = 4):
        super().__init__()
        self.d = d
        # --- scene tokens: split the 7-d scene vector into 4 semantic groups -------------------------
        self.groups = [[0, 2, 6], [1], [3, 4, 5], [0, 3, 4]]        # ego | road | lead | context(ego speed + lead gap/speed)
        self.scene_proj = nn.ModuleList([nn.Linear(len(g), d) for g in self.groups])
        self.scene_pos = nn.Parameter(torch.randn(S_TOK, d) * 0.02)
        # --- world queries / action tokens / riders -----------------------------------------------------
        self.world_q = nn.Parameter(torch.randn(T, d) * 0.02)
        self.act_in = nn.Linear(2, d)
        self.act_pos = nn.Parameter(torch.randn(T, d) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.rider_in = nn.Sequential(nn.Linear(2 * T, d), nn.GELU(), nn.Linear(d, d))
        self.rider_pos = nn.Parameter(torch.randn(N_RIDER, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        # --- heads ---------------------------------------------------------------------------------------
        self.world_head = nn.Linear(d, 1)       # future lead x per step (normalised)
        self.act_head = nn.Linear(d, 2)         # flow velocity per waypoint
        self.rider_head = nn.Linear(d, 1)       # compatibility logit
        nn.init.zeros_(self.act_head.weight); nn.init.zeros_(self.act_head.bias)

    # ----------------------------------------------------------------------------------------------------
    def _mask(self, n_rider: int, device) -> torch.Tensor:
        S, W, A, R = S_TOK, T, T, n_rider
        n = S + W + A + R
        m = torch.ones(n, n, dtype=torch.bool, device=device)          # True = blocked
        sc, wo, ac = slice(0, S), slice(S, S + W), slice(S + W, S + W + A)
        m[sc, sc] = False
        m[wo, sc] = False; m[wo, wo] = False
        m[ac, sc] = False; m[ac, wo] = False; m[ac, ac] = False
        for r in range(R):
            i = S + W + A + r
            m[i, sc] = False; m[i, i] = False
        return m

    def forward(self, scene_n, xt, t, riders=None):
        """
        scene_n : (B,7)   normalised scene
        xt      : (B,T,2) noisy trajectory at flow time t (normalised coords)
        t       : (B,)    flow time
        riders  : (B,R,T,2) candidate trajectories for the training-only rider branch, or None (inference)
        returns : v (B,T,2) flow velocity, lead_pred (B,T), rider_logit (B,R) or None
        """
        B = scene_n.shape[0]
        S = torch.stack([p(scene_n[:, g]) for p, g in zip(self.scene_proj, self.groups)], 1) + self.scene_pos      # (B,4,d)
        W = self.world_q[None].expand(B, -1, -1)                                                                     # (B,T,d)
        te = self.t_mlp(timestep_embedding(t, self.d))[:, None]                                                      # (B,1,d)
        A = self.act_in(xt) + self.act_pos + te                                                                      # (B,T,d)
        parts = [S, W, A]
        R = 0
        if riders is not None:
            R = riders.shape[1]
            parts.append(self.rider_in(riders.flatten(2)) + self.rider_pos[:R])                                      # (B,R,d)
        x = torch.cat(parts, 1)                                                                                      # (B,4+2T+R,d)
        mask = self._mask(R, x.device)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x)
        w = x[:, S_TOK:S_TOK + T]; a = x[:, S_TOK + T:S_TOK + 2 * T]
        v = self.act_head(a)
        lead = self.world_head(w).squeeze(-1)
        logit = self.rider_head(x[:, S_TOK + 2 * T:]).squeeze(-1) if R else None
        return v, lead, logit

    # ----------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, scene_n, steps: int = 8, noise=None, generator=None):
        """Euler ODE from noise to trajectory. Riders are NOT used (inference = plain WAM). Returns normalised traj (B,T,2)."""
        B = scene_n.shape[0]
        x = noise if noise is not None else torch.randn(B, T, 2, generator=generator)
        for i in range(steps):
            t = torch.full((B,), i / steps)
            v, _, _ = self.forward(scene_n, x, t, riders=None)
            x = x + v / steps
        return x
