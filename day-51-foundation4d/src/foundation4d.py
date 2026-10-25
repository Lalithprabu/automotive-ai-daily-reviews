"""Foundation4D-Mini: a small PyTorch reconstruction of the *shape* described in the Foundation4D abstract
(arXiv:2610.30340): ONE feed-forward model, a shared backbone with FACTORIZED spatio-temporal attention
over (cameras x time), and four task heads: geometry (depth), 3D detection, semantic/3D occupancy, and
per-pixel Gaussians. Every dimension / head design here is this repo's own default (see SOURCING.md).

Tensor legend: B batch, V cameras, T frames, N patch tokens per image, D model dim.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .synthetic import BEV_N, BEV_R, N_Z


class Block(nn.Module):
    """Pre-norm transformer block: x + MHA(LN(x)); x + MLP(LN(x)). Operates on [*, L, D]."""

    def __init__(self, dim, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class FactorizedSTBlock(nn.Module):
    """Factorized spatio-temporal attention over tokens [B,V,T,N,D]:
       (1) spatial   : tokens attend within one (camera, frame) image              -> sequence length N
       (2) cross-view: tokens attend across cameras at the same frame & patch-slot  -> sequence length V*N
       (3) temporal  : tokens attend across frames for the same camera & patch-slot  -> sequence length T
    Cost per layer ~ B*V*T*N^2 + B*T*(V*N)^2 + B*V*N*T^2 instead of B*(V*T*N)^2 for joint attention."""

    def __init__(self, dim, heads):
        super().__init__()
        self.spatial, self.view, self.temporal = Block(dim, heads), Block(dim, heads), Block(dim, heads)

    def forward(self, x):
        B, V, T, N, D = x.shape
        x = self.spatial(x.reshape(B * V * T, N, D)).reshape(B, V, T, N, D)
        xv = x.permute(0, 2, 1, 3, 4).reshape(B * T, V * N, D)                     # same frame, all cameras
        x = self.view(xv).reshape(B, T, V, N, D).permute(0, 2, 1, 3, 4)
        xt = x.permute(0, 1, 3, 2, 4).reshape(B * V * N, T, D)                     # same cam/patch, all frames
        x = self.temporal(xt).reshape(B, V, N, T, D).permute(0, 1, 3, 2, 4)
        return x.contiguous()


class JointBlock(nn.Module):
    """'Old way' ablation: one dense attention over all V*T*N tokens (quadratic in V*T*N)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.blk = Block(dim, heads)

    def forward(self, x):
        B, V, T, N, D = x.shape
        return self.blk(x.reshape(B, V * T * N, D)).reshape(B, V, T, N, D)


def attention_pairs(V, T, N, factorized=True):
    """Number of query-key pairs per layer per sample (the quantity attention cost scales with)."""
    if not factorized:
        return (V * T * N) ** 2
    return V * T * N * N + T * (V * N) ** 2 + V * N * T * T


class Foundation4DMini(nn.Module):
    def __init__(self, dim=64, heads=4, depth=4, patch=4, n_views=4, n_frames=2, img_hw=(24, 32),
                 factorized=True, tasks=("depth", "det", "occ", "gs")):
        super().__init__()
        self.dim, self.patch, self.V, self.T, self.tasks = dim, patch, n_views, n_frames, tuple(tasks)
        H, W = img_hw
        self.gh, self.gw = H // patch, W // patch
        self.N = self.gh * self.gw
        self.embed = nn.Conv2d(3, dim, patch, patch)                                # patchify  [*,3,H,W]->[*,D,gh,gw]
        self.pos = nn.Parameter(torch.zeros(1, 1, 1, self.N, dim))                  # intra-image position
        self.view_emb = nn.Parameter(torch.zeros(1, n_views, 1, 1, dim))            # camera identity
        self.time_emb = nn.Parameter(torch.zeros(1, 1, n_frames, 1, dim))           # frame index
        for p in (self.pos, self.view_emb, self.time_emb):
            nn.init.trunc_normal_(p, std=0.02)
        Blk = FactorizedSTBlock if factorized else JointBlock
        self.blocks = nn.ModuleList([Blk(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        # --- geometry head: token -> patch*patch log-depth values
        self.depth_head = nn.Linear(dim, patch * patch)
        # --- Gaussian head: token -> per-pixel (opacity logit, log-scale, rgb) = 5 values / pixel
        self.gs_head = nn.Linear(dim, patch * patch * 5)
        # --- BEV head: geometry-guided lifting. Each BEV cell centre is projected into every camera (known pinhole
        # rig, same as src/synthetic.py) at two heights and the token feature map is bilinearly sampled there
        # (BEVFormer-style spatial sampling). Free learned queries / plain cross-attention were tried first and
        # did not train on this small synthetic set (see SOURCING.md).
        grid, valid = self._bev_projection(n_views, img_hw)
        self.register_buffer("bev_grid", grid)                                       # [V,Z,256,2] in [-1,1]
        self.register_buffer("bev_valid", valid)                                     # [V,Z,256]
        n_h = grid.shape[1]
        self.bev_proj = nn.Linear(dim * n_h, dim)
        ctr = (torch.arange(BEV_N) + 0.5) / BEV_N * 2 - 1
        yy, xx = torch.meshgrid(ctr, ctr, indexing="ij")
        self.register_buffer("bev_xy", torch.stack([xx, yy], -1).reshape(-1, 2))
        self.bev_coord = nn.Linear(2, dim)
        self.bev_n = nn.LayerNorm(dim)
        self.bev_self = Block(dim, heads)
        self.det_head = nn.Linear(dim, 1)                                           # centre heat-map logit
        self.occ_head = nn.Linear(dim, N_Z)                                         # N_Z height-layer occupancy logits

    @staticmethod
    def _bev_projection(V, img_hw, heights=(0.4, 1.2)):
        H, W = img_hw
        f = W / 2.0
        ctr = (torch.arange(BEV_N) + 0.5) / BEV_N * 2 * BEV_R - BEV_R
        yy, xx = torch.meshgrid(ctr, ctr, indexing="ij")
        grids, valids = [], []
        for k in range(V):
            th = k * math.pi / 2
            gz, vz = [], []
            for h in heights:
                px, py, pz = xx.reshape(-1), yy.reshape(-1), torch.full((BEV_N * BEV_N,), h - 1.5)
                zc = px * math.sin(th) + py * math.cos(th)
                xc = px * math.cos(th) - py * math.sin(th)
                u = f * xc / zc.clamp(min=0.1) + W / 2
                v = f * (-pz) / zc.clamp(min=0.1) + H / 2
                gz.append(torch.stack([u / W * 2 - 1, v / H * 2 - 1], -1))
                vz.append((zc > 0.5) & (u > 0) & (u < W) & (v > 0) & (v < H))
            grids.append(torch.stack(gz)); valids.append(torch.stack(vz))
        return torch.stack(grids), torch.stack(valids).float()

    def encode(self, rgb):
        """rgb [B,V,T,3,H,W] -> tokens [B,V,T,N,D]."""
        B, V, T, C, H, W = rgb.shape
        x = self.embed(rgb.reshape(B * V * T, C, H, W))                             # [BVT,D,gh,gw]
        x = x.flatten(2).transpose(1, 2).reshape(B, V, T, self.N, self.dim)
        x = x + self.pos + self.view_emb[:, :V] + self.time_emb[:, :, :T]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, rgb):
        x = self.encode(rgb)
        B, V, T, N, D = x.shape
        out = {}
        if "depth" in self.tasks or "gs" in self.tasks:
            p, gh, gw = self.patch, self.gh, self.gw
            def unpatch(t, c):                                                       # [B,V,T,N,c*p*p]->[B,V,T,c,H,W]
                t = t.reshape(B, V, T, gh, gw, c, p, p).permute(0, 1, 2, 5, 3, 6, 4, 7)
                return t.reshape(B, V, T, c, gh * p, gw * p)
            if "depth" in self.tasks:
                out["depth"] = unpatch(self.depth_head(x), 1)[:, :, :, 0].exp()      # metres, positive by construction
            if "gs" in self.tasks:
                g = self.gs_head(x).reshape(B, V, T, gh, gw, p, p, 5).permute(0, 1, 2, 7, 3, 5, 4, 6) \
                    .reshape(B, V, T, 5, gh * p, gw * p)                                 # [B,V,T,5,H,W]
                out["gs"] = dict(opacity=torch.sigmoid(g[:, :, :, 0]), scale=g[:, :, :, 1].exp(),
                                 rgb=torch.sigmoid(g[:, :, :, 2:5]))
        if "det" in self.tasks or "occ" in self.tasks:
            fm = x[:, :, -1].reshape(B * V, self.gh, self.gw, D).permute(0, 3, 1, 2)    # current-frame feature maps
            feats = []
            for zi in range(self.bev_grid.shape[1]):
                gr = self.bev_grid[:, zi].reshape(1, V, -1, 1, 2).expand(B, -1, -1, -1, -1).reshape(B * V, -1, 1, 2)
                smp = F.grid_sample(fm, gr, align_corners=False, padding_mode="zeros")  # [BV,D,256,1]
                smp = smp.reshape(B, V, D, -1) * self.bev_valid[:, zi][None, :, None]
                feats.append(smp.sum(1))                                                # fuse cameras by masked sum
            q = self.bev_proj(torch.cat(feats, 1).transpose(1, 2))                     # [B,256,D]
            q = self.bev_self(self.bev_n(q + self.bev_coord(self.bev_xy)))
            if "det" in self.tasks:
                out["det"] = self.det_head(q).transpose(1, 2).reshape(B, 1, BEV_N, BEV_N)
            if "occ" in self.tasks:
                out["occ"] = self.occ_head(q).transpose(1, 2).reshape(B, N_Z, BEV_N, BEV_N)
        return out


# ------------------------------- losses -------------------------------------------------------------------------
def class_balanced_bce(logit, target):
    """Positive / negative cells averaged separately then summed (Day 13 lesson: plain mean BCE collapses)."""
    pos = (target > 0.5).float()
    l = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    return (l * pos).sum() / pos.sum().clamp(min=1) + (l * (1 - pos)).sum() / (1 - pos).sum().clamp(min=1)


def compute_loss(out, batch, w=None):
    w = w or dict(depth=1.0, det=1.0, occ=1.0, gs=0.5)
    parts = {}
    if "depth" in out:
        parts["depth"] = F.l1_loss(out["depth"].log(), batch["depth"].clamp(min=0.1).log())
    if "det" in out:
        parts["det"] = class_balanced_bce(out["det"], batch["heat"])
    if "occ" in out:
        parts["occ"] = class_balanced_bce(out["occ"], batch["occ"])
    if "gs" in out:
        gs = out["gs"]                                                              # splat each Gaussian onto its own pixel
        parts["gs"] = F.l1_loss(gs["rgb"], batch["rgb"]) + F.l1_loss(gs["opacity"], torch.ones_like(gs["opacity"])) * 0.1
    total = sum(w[k] * v for k, v in parts.items())
    return total, parts


def unproject_gaussian_centres(depth, rays):
    """Gaussian means = camera origin + ray * predicted depth. depth [B,V,T,H,W], rays [V,H*W,3] -> [B,V,T,H*W,3]."""
    B, V, T, H, W = depth.shape
    o = torch.tensor([0.0, 0.0, 1.5])
    return o + rays[None, :, None] * depth.reshape(B, V, T, H * W, 1)
