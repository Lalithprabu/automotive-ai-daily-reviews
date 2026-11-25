"""PyTorch core of a glance-then-gaze trajectory planner: each candidate trajectory picks its own visual evidence.

Re-implementation of the idea in arXiv 2611.27330 (Sun, Li, Shi, Zheng, Hoe, Bai, Mi, "GGDriver: Glance at What Is
Shared, Gaze at What Differs for Perception-Free Planning"), built from the abstract only.  What the abstract states:

    (1) GLANCE: a global glance produces trajectory hypotheses from SHARED scene features
    (2) GAZE: a corridor around each trajectory is projected into the camera views; the resulting mask selects patch
        features at the backbone's NATIVE resolution; only the predicted trajectory + camera calibration are needed
    (3) after refinement, the scorer RE-PROJECTS each final proposal and ranks it on evidence along its own path
    (4) ablations: path-aligned patches beat a matched number of random patches and attention over all patches;
        gazing for ranking helps more than gazing for refinement only

Everything else (backbone, register count, sizes, heads, losses, the 45-anchor vocabulary) is my construction.

Shapes (B scenes, K = 45 candidate anchors, D = 96, patch grid 12 x 20 = N = 240 tokens, R = 8 registers):

    image       (B, 3, 96, 160)  -> conv backbone, stride 8                       -> P  (B, N, D)   native patches
    glance      R learned queries cross-attend to all of P (+ ego speed token)    -> G  (B, R + 1, D)  "shared" scene
                anchor queries q_k = TrajEnc(waypoints_k) attend to G             -> glance logit_k   (B, K)
    refine      q_k attends to its EVIDENCE E_k:                                  -> delta_k = (dd, dv)  (B, K, 2)
                    glance  E_k = G                    (every candidate sees the same compressed scene)
                    gaze    E_k = P masked by corridor(waypoints_k)    (B, K, N) boolean mask, ~17% of patches
                    random  E_k = P masked by a random mask with the same patch count as the corridor
                    all     E_k = P, no mask
    score       refined waypoints are RE-PROJECTED -> new corridor mask; q'_k attends to E'_k
                -> logits for NC (no collision), DAC (drivable), EP (progress)    (B, K, 3)
                score_k = sig(NC) * sig(DAC) * (0.75 sig(EP) + 0.25 C_k)          C_k = analytic comfort of (d, v)
    select      top-M (M = 8) anchors by glance logit -> refine -> pick argmax score_k
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .scene import (ANCHORS, CAM_H, CX, CY, F_PX, GH, GW, H, PATCH, REF_D, REF_V, W, comfort, traj_points)

MODES = ("glance", "gaze", "random", "all")


# ------------------------------------------------------------------------------------------------ corridor -> mask
def corridor_mask(pts: torch.Tensor, half_width: float = 1.3, height: float = 2.0, z_min: float = 6.0) -> torch.Tensor:
    """Project a 3-D corridor around each trajectory into the camera; return the patches it touches.

    pts (B, K, 16, 2) waypoints [X, Z] -> mask (B, K, N = 240) bool.
    The corridor is the ego path from Z = 0 densified to 4 samples per waypoint segment, swept laterally over
    [-hw, +hw] (5 samples) and vertically over [0, height] (4 samples) - a box tube the ego body will occupy.
    Each 3-D sample (X, Y, Z) projects to u = cx + f X / Z, v = cy + f (CAM_H - Y) / Z with only the calibration.
    """
    B, K, T, _ = pts.shape
    p = torch.cat([torch.zeros_like(pts[:, :, :1]), pts], 2)                      # prepend the ego origin
    a = torch.linspace(0, 1, 5, device=pts.device)[:-1]                            # 4 samples per segment
    dense = (p[:, :, :-1, None] * (1 - a[:, None]) + p[:, :, 1:, None] * a[:, None]).flatten(2, 3)  # (B, K, 64, 2)
    # extend 4 m beyond the last waypoint along the final heading, so the area the ego body reaches is covered
    head = F.normalize(p[:, :, -1] - p[:, :, -2] + torch.tensor([0.0, 1e-3], device=pts.device), dim=-1)
    ext = p[:, :, -1:, :] + head[:, :, None] * torch.tensor([1.0, 2.0, 3.0, 4.0], device=pts.device)[:, None]
    dense = torch.cat([dense, ext], 2)                                             # (B, K, 68, 2)
    lat = torch.linspace(-half_width, half_width, 5, device=pts.device)
    hgt = torch.linspace(0, height, 4, device=pts.device)
    X = dense[..., 0, None, None] + lat[:, None]                                  # (B, K, S, 5, 1)
    Z = dense[..., 1, None, None].expand_as(X)
    Y = hgt.view(1, 1, 1, 1, 4)
    Zc = Z.clamp(min=1e-3)
    u = CX + F_PX * X / Zc                                                         # (B, K, S, 5, 1)
    v = CY + F_PX * (CAM_H - Y) / Zc                                               # (B, K, S, 5, 4)
    u = u.expand_as(v)
    ok = (Z.expand_as(v) > z_min) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = (v.clamp(0, H - 1) // PATCH).long() * GW + (u.clamp(0, W - 1) // PATCH).long()
    idx = torch.where(ok, idx, torch.full_like(idx, GH * GW))                      # dump bucket for invalid
    m = torch.zeros(B, K, GH * GW + 1, dtype=torch.bool, device=pts.device)
    m.scatter_(2, idx.flatten(2), True)
    m = m[..., :-1]
    # dilate by one patch horizontally so thin far corridors keep their neighbours (objects straddle patch borders)
    g = m.view(B, K, GH, GW).float()
    g = F.max_pool2d(g.view(B * K, 1, GH, GW), (1, 3), 1, (0, 1)).view(B, K, GH * GW) > 0
    return g


def random_mask_like(mask: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    """Random patch mask with exactly the same number of patches per (b, k) as `mask`."""
    n = mask.sum(-1, keepdim=True)                                                # (B, K, 1)
    r = torch.rand(mask.shape, generator=gen, device=mask.device)
    rank = r.argsort(-1).argsort(-1)
    return rank < n


# ------------------------------------------------------------------------------------------------ blocks
class Backbone(nn.Module):
    """(B, 3, 96, 160) -> native patch tokens (B, N = 240, D) at stride 8, with a learned 2-D position table."""

    def __init__(self, d: int = 96):
        super().__init__()
        def blk(ci, co, s):
            return nn.Sequential(nn.Conv2d(ci, co, 3, s, 1), nn.GroupNorm(8, co), nn.GELU())
        self.stem = nn.Sequential(blk(3, 32, 2), blk(32, 48, 2), blk(48, d, 2))          # 96x160 -> 12x20
        self.res = nn.ModuleList([nn.Sequential(nn.Conv2d(d, d, 3, 1, 1), nn.GroupNorm(8, d), nn.GELU(),
                                                nn.Conv2d(d, d, 3, 1, 1), nn.GroupNorm(8, d)) for _ in range(2)])
        self.pos = nn.Parameter(torch.randn(1, GH * GW, d) * 0.02)

    def forward(self, x):
        f = self.stem(x)
        for r in self.res:
            f = F.gelu(f + r(f))
        return f.flatten(2).transpose(1, 2) + self.pos                                    # (B, 240, D)


class CrossAttn(nn.Module):
    """Pre-norm multi-head cross-attention with an optional PER-QUERY key mask, plus an MLP.

    q (B, K, D), kv (B, N, D), mask (B, K, N) bool (True = may attend) or None.  The mask is applied to the logits
    (B, heads, K, N) directly, so K x N masks cost no copies of kv - this is how 45 candidates each gaze at their own
    subset of the same 240 patches in one batched op.
    """

    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.nq, self.nk, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.last_attn: torch.Tensor | None = None
        self.keep = False

    def forward(self, q, kv, mask=None):
        B, K, D = q.shape
        N = kv.shape[1]
        kvn = self.nk(kv)
        Q = self.q(self.nq(q)).view(B, K, self.h, self.dh).transpose(1, 2)               # (B, h, K, dh)
        Kt = self.k(kvn).view(B, N, self.h, self.dh).transpose(1, 2)                     # (B, h, N, dh)
        V = self.v(kvn).view(B, N, self.h, self.dh).transpose(1, 2)
        logit = Q @ Kt.transpose(-1, -2) / math.sqrt(self.dh)                            # (B, h, K, N)
        if mask is not None:
            mask = mask | ~mask.any(-1, keepdim=True)                                    # empty mask -> attend all
            logit = logit.masked_fill(~mask[:, None], float("-inf"))
        a = logit.softmax(-1)
        if self.keep:
            self.last_attn = a.detach().mean(1)                                          # (B, K, N)
        y = (a @ V).transpose(1, 2).reshape(B, K, D)
        q = q + self.o(y)
        return q + self.mlp(self.n2(q))


class TrajEnc(nn.Module):
    """(d, v) + v0 -> waypoints -> query embedding (B, K, D)."""

    def __init__(self, d: int = 96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16 * 2 + 3, 128), nn.GELU(), nn.Linear(128, d))

    def forward(self, dv, v0, pts):
        x = torch.cat([pts[..., 0] / 7.0, pts[..., 1] / 60.0, dv[..., :1] / 7.0, dv[..., 1:] / 16.0,
                       (v0[:, None, None] / 16.0).expand_as(dv[..., :1])], -1)
        return self.net(x)


# ------------------------------------------------------------------------------------------------ planner
class GlanceGazePlanner(nn.Module):
    def __init__(self, refine: str = "gaze", score: str = "gaze", d: int = 96, n_reg: int = 8):
        super().__init__()
        assert refine in MODES and score in MODES
        self.refine_mode, self.score_mode = refine, score
        self.backbone = Backbone(d)
        self.reg = nn.Parameter(torch.randn(1, n_reg, d) * 0.02)
        self.reg_attn = nn.ModuleList([CrossAttn(d), CrossAttn(d)])                      # registers <- patches
        self.ego = nn.Linear(1, d)
        self.enc = TrajEnc(d)
        self.glance_attn = CrossAttn(d)
        self.glance_head = nn.Linear(d, 1)
        self.ref_attn = nn.ModuleList([CrossAttn(d), CrossAttn(d)])
        self.ref_head = nn.Linear(d, 2)
        self.score_ctx = CrossAttn(d)
        self.score_attn = nn.ModuleList([CrossAttn(d), CrossAttn(d)])
        self.score_head = nn.Linear(d, 3)
        self.register_buffer("anchors", torch.as_tensor(ANCHORS))
        nn.init.zeros_(self.ref_head.weight)
        nn.init.zeros_(self.ref_head.bias)

    def evidence(self, mode, patches, glance, pts):
        """-> (kv, mask, n_patches_used) for one stage."""
        if mode == "glance":
            return glance, None, None
        if mode == "all":
            return patches, None, None
        m = corridor_mask(pts)
        if mode == "random":
            m = random_mask_like(m)
        return patches, m, m

    def forward(self, img: torch.Tensor, v0: torch.Tensor, anchor_idx: torch.Tensor | None = None) -> dict:
        """img (B, 3, 96, 160) in [0, 1]; v0 (B,) m/s; anchor_idx (B, M) or None (= all 45 anchors)."""
        B = img.shape[0]
        P = self.backbone(img)                                                            # (B, 240, D)
        g = self.reg.expand(B, -1, -1)
        for a in self.reg_attn:
            g = a(g, P)                                                                   # (B, 8, D)
        G = torch.cat([g, self.ego(v0[:, None] / 16.0)[:, None]], 1)                      # (B, 9, D)

        A = self.anchors[None].expand(B, -1, -1)                                          # (B, 45, 2)
        if anchor_idx is not None:
            A = A.gather(1, anchor_idx[..., None].expand(-1, -1, 2))
        ptsA = traj_points(A, v0[:, None])                                                # (B, K, 16, 2)
        q = self.glance_attn(self.enc(A, v0, ptsA), G)                                    # (B, K, D)
        glance_logit = self.glance_head(q)[..., 0]                                        # (B, K)

        kv, m1, _ = self.evidence(self.refine_mode, P, G, ptsA)
        r = q
        for a in self.ref_attn:
            r = a(r, kv, m1)
        delta = torch.tanh(self.ref_head(r)) * torch.tensor([REF_D, REF_V], device=img.device)
        dv = A + delta
        dv = torch.stack([dv[..., 0], dv[..., 1].clamp(min=0)], -1)
        dv_s = dv.detach()                                                                # scorer sees a fixed proposal
        pts = traj_points(dv_s, v0[:, None])                                              # RE-PROJECT the proposal
        s = self.score_ctx(self.enc(dv_s, v0, pts), G)
        kv2, m2, _ = self.evidence(self.score_mode, P, G, pts)
        for a in self.score_attn:
            s = a(s, kv2, m2)
        sub = self.score_head(s)                                                          # (B, K, 3)
        p = torch.sigmoid(sub)
        score = p[..., 0] * p[..., 1] * (0.75 * p[..., 2] + 0.25 * comfort(dv_s, v0))
        return dict(glance_logit=glance_logit, delta=delta, dv=dv, pts=pts, sub=sub, score=score, anchors=A,
                    mask_refine=m1, mask_score=m2)

    @torch.no_grad()
    def plan(self, img, v0, top_m: int = 8) -> dict:
        """Glance -> top-M hypotheses -> refine -> re-project and score -> best proposal."""
        B = img.shape[0]
        full = self.forward(img, v0)
        idx = full["glance_logit"].topk(top_m, -1).indices                                 # (B, M)
        out = self.forward(img, v0, idx)
        best = out["score"].argmax(-1)
        pick = lambda t: t[torch.arange(B), best]
        return dict(dv=pick(out["dv"]), pts=pick(out["pts"]), score=pick(out["score"]), best=best, idx=idx, out=out,
                    full=full)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for r, s in (("glance", "glance"), ("gaze", "gaze"), ("random", "random"), ("all", "all")):
        m = GlanceGazePlanner(r, s)
        x = torch.rand(4, 3, 96, 160)
        v0 = torch.rand(4) * 10 + 4
        o = m(x, v0)
        (o["score"].mean() + o["delta"].mean() + o["glance_logit"].mean()).backward()
        frac = o["mask_score"].float().mean().item() if o["mask_score"] is not None else 1.0
        print(r, s, f"params {n_params(m):,}", tuple(o["score"].shape), f"patch fraction {frac:.3f}")
