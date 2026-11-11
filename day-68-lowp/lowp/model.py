"""
LOWP — "Look Only Where You Plan" (arXiv 2611.10264) — toy PyTorch re-implementation of the core.

What is implemented (from the paper's method description, as read from the arXiv HTML):
  * Shared image trunk (here: a tiny stride-8 CNN instead of ResNet-34).
  * Expert 1, DeformableAnchorReader: 16 learnable queries, each owning a metric 3D anchor in the ego
    frame. Anchors start on a grid ahead of the car, are shifted per scene by query content, and are
    refined each layer by a bounded step (tanh * max_step). Every anchor is lifted to 4 heights over
    3 m; at each height 8 points sit on a query-offset ring -> 32 candidates. Candidates are projected
    with K R^T (p - t), sampled bilinearly, and aggregated by multi-head attention together with a
    learned NULL candidate (so a query that sees nothing gets a learned state). The anchor position is
    fed back as an embedding; anchors invisible to every camera at init are re-seeded.
  * Expert 2, GlobalRayReader: 16 latent queries cross-attend to every feature cell of every camera.
    Each cell is labelled with the 3D points its viewing ray passes at 8 depths (1-60 m); that
    encoding is added to the cell feature (geometry-aware attention, no depth supervision).
  * Per-expert planning head: reader tokens + an ego-status token -> 2-layer transformer -> T poses.
  * Inference mix: tau = a * tau_def + (1 - a) * tau_glob, a = 0.5 straight, 0.8 turns.
  * L1 loss on each expert's trajectory; trunk shared and trained jointly.
No dense BEV is built and no object / map labels are used.

NOT implemented: nuScenes LiDAR-depth adapter, NAVSIM/Bench2Drive vocabulary scorers, control branch,
recovery / brake-release rules, temporal history frame. Hidden sizes are my choice (not in the paper).

Shape legend: B batch, N cameras, C/d channels, Hf x Wf feature map, Q queries (16),
              P candidates per anchor (4 heights x 8 ring points = 32), T waypoints.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import project_points, ray_points

CMD_STRAIGHT, CMD_LEFT, CMD_RIGHT = 0, 1, 2


# ============================================================================ shared pieces
class Trunk(nn.Module):
    """Tiny stride-8 CNN shared by both readers: (B, N, 3, H, W) -> (B, N, d, H/8, W/8)."""

    def __init__(self, d: int = 64):
        super().__init__()

        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.net = nn.Sequential(blk(3, 32, 2), blk(32, 48, 2), blk(48, 64, 2), blk(64, d, 1))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        B, N = img.shape[:2]
        f = self.net(img.flatten(0, 1))                              # (B*N, d, Hf, Wf)
        return f.view(B, N, *f.shape[1:])


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(inplace=True), nn.Linear(h, o))


def sample_multicam(feat: torch.Tensor, grid: torch.Tensor, valid: torch.Tensor):
    """Bilinear-sample N camera feature maps at projected points and average over the cameras that see them.

    feat: (B, N, C, Hf, Wf); grid: (B, N, M, 2); valid: (B, N, M)
    returns: (B, M, C) features, (B, M) visible-in-any-camera mask
    """
    B, N, C, Hf, Wf = feat.shape
    M = grid.shape[2]
    s = F.grid_sample(feat.flatten(0, 1), grid.flatten(0, 1)[:, :, None, :],
                      mode="bilinear", padding_mode="zeros", align_corners=False)   # (B*N, C, M, 1)
    s = s.view(B, N, C, M).permute(0, 3, 1, 2)                          # (B, M, N, C)
    w = valid.permute(0, 2, 1).float()                                  # (B, M, N)
    out = (s * w[..., None]).sum(2) / w.sum(2, keepdim=True).clamp(min=1.0)
    return out, valid.any(1)


class EgoToken(nn.Module):
    """Ego status (speed, accel) + one-hot driving command -> one token."""

    def __init__(self, d: int, n_status: int = 2):
        super().__init__()
        self.proj = mlp(n_status + 3, d, d)

    def forward(self, status: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([status, F.one_hot(cmd, 3).float()], -1))   # (B, d)


class PlanningHead(nn.Module):
    """Reader tokens (B, Q, d) + ego token (B, d) -> 2-layer transformer -> (B, T, 2) future (x, y) in metres."""

    def __init__(self, d: int, T: int, n_layers: int = 2, heads: int = 4, scale: float = 10.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out = mlp(d, d, T * 2)
        self.T, self.scale = T, scale

    def forward(self, tokens: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
        x = self.enc(torch.cat([ego[:, None], tokens], 1))              # (B, 1+Q, d)
        # regress per-step displacements and integrate -> smooth, monotone-in-time trajectories
        delta = self.out(x[:, 0]).view(-1, self.T, 2) * (self.scale / self.T)
        return delta.cumsum(1)                                          # (B, T, 2)


# ============================================================================ Expert 1
class DeformableAnchorReader(nn.Module):
    def __init__(self, d: int = 64, n_q: int = 16, n_layers: int = 3, heads: int = 4,
                 n_heights: int = 4, height_span: float = 3.0, n_ring: int = 8,
                 ring_radius: float = 1.5, max_step: float = 3.0, max_init_shift: float = 6.0):
        super().__init__()
        self.n_q, self.n_layers, self.max_step, self.max_init_shift = n_q, n_layers, max_step, max_init_shift
        self.ring_radius = ring_radius
        # --- initial anchor grid ahead of the car: 4 x 4 over x in [6, 30] m, y in [-6, 6] m
        gx, gy = torch.meshgrid(torch.linspace(6, 30, 4), torch.linspace(-6, 6, 4), indexing="ij")
        self.register_buffer("anchor_grid", torch.stack([gx, gy], -1).view(-1, 2))          # (Q, 2)
        self.register_buffer("heights", torch.linspace(0.0, height_span, n_heights))         # (4,)
        ang = torch.arange(n_ring) * 2 * math.pi / n_ring
        self.register_buffer("ring", torch.stack([ang.cos(), ang.sin()], -1) * ring_radius)  # (8, 2)

        self.query = nn.Embedding(n_q, d)
        self.scene_ctx = mlp(2 * d, d, d)          # pooled image feature + ego token -> scene content
        self.init_shift = nn.Linear(d, 2)          # per-scene initial anchor shift (bounded by tanh)
        self.anchor_emb = mlp(3, d, d)             # anchor position fed back as an embedding
        self.cand_emb = mlp(3, d, d)               # candidate offset (relative to anchor) embedding
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        P = n_heights * n_ring
        self.P = P
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict(dict(
                step=nn.Linear(d, 2),                         # bounded anchor refinement
                ring_off=nn.Linear(d, n_ring * 2),            # query-dependent ring offsets
                self_attn=nn.MultiheadAttention(d, heads, batch_first=True),
                cand_attn=nn.MultiheadAttention(d, heads, batch_first=True),
                n1=nn.LayerNorm(d), n2=nn.LayerNorm(d), n3=nn.LayerNorm(d),
                ffn=mlp(d, 2 * d, d),
            )))
        for l in self.layers:                         # start with zero refinement / zero offsets
            nn.init.zeros_(l["step"].weight); nn.init.zeros_(l["step"].bias)
            nn.init.zeros_(l["ring_off"].weight); nn.init.zeros_(l["ring_off"].bias)
        nn.init.zeros_(self.init_shift.weight); nn.init.zeros_(self.init_shift.bias)

    def candidates(self, anchor: torch.Tensor, ring_off: torch.Tensor) -> torch.Tensor:
        """anchor (B, Q, 2), ring_off (B, Q, 8, 2) -> candidates (B, Q, P=32, 3) in the ego frame."""
        B, Q = anchor.shape[:2]
        ring_xy = anchor[:, :, None, :] + self.ring[None, None] + ring_off               # (B, Q, 8, 2)
        xy = ring_xy[:, :, None].expand(B, Q, len(self.heights), -1, 2)                   # (B, Q, 4, 8, 2)
        z = self.heights.view(1, 1, -1, 1, 1).expand(B, Q, -1, ring_xy.shape[2], 1)       # (B, Q, 4, 8, 1)
        return torch.cat([xy, z], -1).reshape(B, Q, -1, 3)                                # (B, Q, 32, 3)

    def forward(self, feat, K, R, t, img_hw, ego_tok):
        """feat (B, N, d, Hf, Wf); K, R (B, N, 3, 3); t (B, N, 3); ego_tok (B, d) -> tokens (B, Q, d) + diagnostics."""
        B, N, d = feat.shape[:3]
        Q, P = self.n_q, self.P
        pooled = feat.mean(dim=(1, 3, 4))                                                  # (B, d)
        q = self.query.weight[None].expand(B, -1, -1) + self.scene_ctx(torch.cat([pooled, ego_tok], -1))[:, None]
        # (1) per-scene initial shift of the anchor grid
        anchor = self.anchor_grid[None] + torch.tanh(self.init_shift(q)) * self.max_init_shift   # (B, Q, 2)
        # (2) re-seed anchors whose candidates are invisible in every camera at initialisation
        zero_off = torch.zeros(B, Q, self.ring.shape[0], 2, device=feat.device)
        _, vis, _ = project_points(self.candidates(anchor, zero_off).view(B, Q * P, 3), K, R, t, img_hw)
        seen = vis.any(1).view(B, Q, P).any(-1, keepdim=True)                              # (B, Q, 1)
        anchor = torch.where(seen, anchor, self.anchor_grid[None].expand(B, -1, -1))
        n_reseeded = (~seen).sum()

        diag = dict(anchors=[anchor], cands=[], cand_valid=[], attn=[])
        for l in self.layers:
            a3 = torch.cat([anchor, torch.zeros_like(anchor[..., :1])], -1)                # (B, Q, 3)
            qp = q + self.anchor_emb(a3 / 30.0)                                            # position fed back
            # bounded refinement of the anchor (tanh * max_step metres)
            anchor = anchor + torch.tanh(l["step"](qp)) * self.max_step
            ring_off = torch.tanh(l["ring_off"](qp)).view(B, Q, -1, 2) * self.ring_radius
            cands = self.candidates(anchor, ring_off)                                      # (B, Q, 32, 3)
            grid, valid, _ = project_points(cands.view(B, Q * P, 3), K, R, t, img_hw)      # (B, N, Q*P, 2)
            sampled, vis_any = sample_multicam(feat, grid, valid)                          # (B, Q*P, d), (B, Q*P)
            rel = cands - torch.cat([anchor, torch.zeros_like(anchor[..., :1])], -1)[:, :, None]
            keys = sampled.view(B, Q, P, d) + self.cand_emb(rel / 3.0)                     # (B, Q, 32, d)
            keys = torch.cat([self.null.expand(B * Q, 1, d), keys.view(B * Q, P, d)], 1)  # (B*Q, 33, d)
            mask = torch.cat([torch.zeros(B * Q, 1, dtype=torch.bool, device=feat.device),
                              ~vis_any.view(B * Q, P)], 1)                                 # True = ignore
            # queries talk to each other, then each query attends over its own 32 candidates + NULL
            x = l["n1"](qp)
            q = q + l["self_attn"](x, x, x, need_weights=False)[0]
            x = l["n2"](q + self.anchor_emb(torch.cat([anchor, torch.zeros_like(anchor[..., :1])], -1) / 30.0))
            out, w = l["cand_attn"](x.reshape(B * Q, 1, d), keys, keys, key_padding_mask=mask)
            q = q + out.view(B, Q, d)
            q = q + l["ffn"](l["n3"](q))
            diag["anchors"].append(anchor); diag["cands"].append(cands)
            diag["cand_valid"].append(vis_any.view(B, Q, P)); diag["attn"].append(w.view(B, Q, P + 1))
        diag["n_reseeded"] = n_reseeded
        return q, diag


# ============================================================================ Expert 2
class GlobalRayReader(nn.Module):
    def __init__(self, d: int = 64, n_q: int = 16, n_layers: int = 3, heads: int = 4,
                 n_depths: int = 8, d_min: float = 1.0, d_max: float = 60.0):
        super().__init__()
        # 8 log-spaced depths between 1 and 60 m (spacing is my choice; the paper says "8 depths from 1 to 60 m")
        self.register_buffer("depths", torch.logspace(math.log10(d_min), math.log10(d_max), n_depths))
        self.d_max = d_max
        self.ray_enc = mlp(n_depths * 3, d, d)
        self.latents = nn.Parameter(torch.randn(1, n_q, d) * 0.02)
        self.ego_in = nn.Linear(d, d)
        self.layers = nn.ModuleList([nn.ModuleDict(dict(
            cross=nn.MultiheadAttention(d, heads, batch_first=True),
            self_attn=nn.MultiheadAttention(d, heads, batch_first=True),
            n1=nn.LayerNorm(d), nkv=nn.LayerNorm(d), n2=nn.LayerNorm(d), n3=nn.LayerNorm(d),
            ffn=mlp(d, 2 * d, d))) for _ in range(n_layers)])
        self._ray_cache = None

    def forward(self, feat, K, R, t, img_hw, ego_tok):
        B, N, d, Hf, Wf = feat.shape
        pts = ray_points(K, R, t, img_hw, (Hf, Wf), self.depths)                          # (B, N, Hf, Wf, 8, 3)
        geo = self.ray_enc(pts.flatten(-2) / self.d_max)                                   # (B, N, Hf, Wf, d)
        mem = feat.permute(0, 1, 3, 4, 2) + geo                                            # geometry-labelled cells
        mem = mem.reshape(B, N * Hf * Wf, d)                                               # (B, N*Hf*Wf, d)
        q = self.latents.expand(B, -1, -1) + self.ego_in(ego_tok)[:, None]
        attn = None
        for l in self.layers:
            kv = l["nkv"](mem)
            out, attn = l["cross"](l["n1"](q), kv, kv)                                     # attn: (B, Q, N*Hf*Wf)
            q = q + out
            x = l["n2"](q)
            q = q + l["self_attn"](x, x, x, need_weights=False)[0]
            q = q + l["ffn"](l["n3"](q))
        return q, dict(attn=attn.view(B, -1, N, Hf, Wf))


# ============================================================================ full planner
class LOWP(nn.Module):
    """Two sparse experts over a shared trunk; no dense BEV. mode='both' (LOWP) or 'global' (LOWP-S)."""

    def __init__(self, d: int = 64, T: int = 6, img_hw=(40, 80), mode: str = "both",
                 alpha_straight: float = 0.5, alpha_turn: float = 0.8):
        super().__init__()
        self.img_hw, self.mode = tuple(img_hw), mode
        self.trunk = Trunk(d)
        self.ego = EgoToken(d)
        self.glob = GlobalRayReader(d)
        self.head_glob = PlanningHead(d, T)
        if mode == "both":
            self.defo = DeformableAnchorReader(d)
            self.head_def = PlanningHead(d, T)
        self.alpha_straight, self.alpha_turn = alpha_straight, alpha_turn

    def forward(self, img, K, R, t, status, cmd):
        """img (B, N, 3, H, W) in [0, 1]; returns dict with tau (mixed), tau_def, tau_glob: (B, T, 2)."""
        feat = self.trunk(img)
        ego = self.ego(status, cmd)
        g_tok, g_diag = self.glob(feat, K, R, t, self.img_hw, ego)
        out = dict(tau_glob=self.head_glob(g_tok, ego), glob_diag=g_diag)
        if self.mode == "global":
            out["tau"] = out["tau_glob"]
            return out
        d_tok, d_diag = self.defo(feat, K, R, t, self.img_hw, ego)
        out["tau_def"] = self.head_def(d_tok, ego)
        out["def_diag"] = d_diag
        # command-dependent mixing: alpha = 0.5 when going straight, 0.8 on turns (inference rule)
        alpha = torch.where(cmd == CMD_STRAIGHT, self.alpha_straight, self.alpha_turn).to(img.dtype)[:, None, None]
        out["tau"] = alpha * out["tau_def"] + (1 - alpha) * out["tau_glob"]
        return out

    def loss(self, out, gt):
        """L1 on each expert's own trajectory (the mix is inference-only)."""
        l = F.l1_loss(out["tau_glob"], gt)
        if "tau_def" in out:
            l = l + F.l1_loss(out["tau_def"], gt)
        return l


# ============================================================================ baselines
class DenseIPMPlanner(nn.Module):
    """Dense counterpart: lift features onto a full BEV grid (2 heights) via calibration, CNN BEV encoder,
    then the same planning head. Stand-in for the dense-BEV planner LOWP is compared with (ResWorld in
    the paper) — NOT a re-implementation of ResWorld."""

    def __init__(self, d: int = 64, T: int = 6, img_hw=(40, 80), x_rng=(-10, 46), y_rng=(-20, 20),
                 cell: float = 1.0, heights=(0.0, 1.0)):
        super().__init__()
        self.img_hw = tuple(img_hw)
        self.trunk = Trunk(d)
        self.ego = EgoToken(d)
        xs = torch.arange(x_rng[0] + cell / 2, x_rng[1], cell)
        ys = torch.arange(y_rng[0] + cell / 2, y_rng[1], cell)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        pts = [torch.stack([gx, gy, torch.full_like(gx, h)], -1) for h in heights]
        self.register_buffer("bev_pts", torch.stack(pts, 0).view(-1, 3))                   # (nH*X*Y, 3)
        self.bev_shape = (len(heights), len(xs), len(ys))

        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.bev_enc = nn.Sequential(blk(len(heights) * d, d, 1), blk(d, d, 2),
                                     blk(d, d, 1), blk(d, d, 2))                             # stride 4
        nx, ny = math.ceil(len(xs) / 4), math.ceil(len(ys) / 4)
        self.pos = nn.Parameter(torch.zeros(1, nx * ny, d))
        self.head = PlanningHead(d, T)

    def forward(self, img, K, R, t, status, cmd):
        B = img.shape[0]
        feat = self.trunk(img)
        nH, X, Y = self.bev_shape
        grid, valid, _ = project_points(self.bev_pts[None].expand(B, -1, -1), K, R, t, self.img_hw)
        bev, _ = sample_multicam(feat, grid, valid)                                        # (B, nH*X*Y, d)
        bev = bev.view(B, nH, X, Y, -1).permute(0, 1, 4, 2, 3).flatten(1, 2)               # (B, nH*d, X, Y)
        tok = self.bev_enc(bev).flatten(2).transpose(1, 2) + self.pos                      # (B, X/4*Y/4, d)
        tau = self.head(tok, self.ego(status, cmd))
        return dict(tau=tau)

    def loss(self, out, gt):
        return F.l1_loss(out["tau"], gt)


class EgoStatusMLP(nn.Module):
    """Blind baseline: ego status + command only (no cameras)."""

    def __init__(self, d: int = 64, T: int = 6):
        super().__init__()
        self.net = mlp(5, 2 * d, T * 2)
        self.T = T

    def forward(self, img, K, R, t, status, cmd):
        x = torch.cat([status, F.one_hot(cmd, 3).float()], -1)
        return dict(tau=(self.net(x).view(-1, self.T, 2) * (10.0 / self.T)).cumsum(1))

    def loss(self, out, gt):
        return F.l1_loss(out["tau"], gt)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
