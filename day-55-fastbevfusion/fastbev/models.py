"""FastBEVFusion-style detector (my reconstruction from the paper's abstract + public summary).

Paper (arXiv 2610.37153, Sanitz, Gulsoylu, Frintrop, Wilms): LiDAR -> LiDAR-BEV, cameras -> image-BEV with a
UNIFORM depth distribution (no depth estimation), the two BEVs are only SEMI-ALIGNED (orientation matches, depth
unknown), and ONE lightweight multi-head cross-attention block (LiDAR tokens = queries, image tokens = keys/values)
+ MLP + residual fuses them over the whole BEV.  Dimensions below are this repo's toy defaults.

Tensor legend: B batch, C channels, G BEV size, T = (G/2)^2 tokens, W image columns.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .synthetic import G, W_IMG, FOV, azimuth_map

FUSIONS = ("lidar_only", "camera_only", "add", "concat", "xattn")


class UniformDepthProjector(nn.Module):
    """Image features [B,C,W] -> semi-aligned BEV [B,C,G,G] with NO depth estimate.
    Every BEV cell reads the image column at its azimuth, i.e. each feature is copied along its whole camera ray
    (= a uniform depth distribution along the ray).  `yaw` [B] (rad) perturbs the extrinsics (calibration error)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("az", torch.tensor(azimuth_map(), dtype=torch.float32))      # [G,G]

    def forward(self, feat, yaw=None):
        B, C, W = feat.shape
        az = self.az[None].expand(B, G, G) + (yaw[:, None, None] if yaw is not None else 0.0)    # [B,G,G]
        pos = (az + FOV / 2) / FOV * W - 0.5                                                # fractional column
        inside = ((pos > -0.5) & (pos < W - 0.5)).float()[:, None]                         # cell inside the camera FOV
        pos = pos.clamp(0, W - 1)
        lo = pos.floor().long().clamp(max=W - 2); w = (pos - lo)[:, None]                   # linear interpolation
        flat = lambda idx: torch.gather(feat, 2, idx.reshape(B, 1, -1).expand(B, C, -1)).view(B, C, G, G)
        return (flat(lo) * (1 - w) + flat(lo + 1) * w) * inside                            # [B,C,G,G]


def pos_enc_2d(h, w, dim):
    """Fixed 2-D sin/cos position encoding [h*w, dim]."""
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    d = dim // 4
    freq = torch.exp(-math.log(100.0) * torch.arange(d) / d)
    parts = [f(c.reshape(-1, 1).float() * freq) for c in (y, x) for f in (torch.sin, torch.cos)]
    return torch.cat(parts, 1)


class CrossAttnFusion(nn.Module):
    """LiDAR tokens (queries) attend to image-BEV tokens (keys/values) over the whole BEV, then MLP, upsample,
    residual from the LiDAR BEV.  The attention matrix [T,T] is what lets a LiDAR cell 'find' the right depth along a ray."""

    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.down = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.down_i = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n1, self.n1k, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.register_buffer("pe", pos_enc_2d(G // 2, G // 2, dim))
        self.last_attn = None

    def forward(self, bev_l, bev_i, keep_attn=False):
        B, C = bev_l.shape[:2]
        q = self.down(bev_l).flatten(2).transpose(1, 2) + self.pe          # [B,T,C]  LiDAR tokens
        kv = self.down_i(bev_i).flatten(2).transpose(1, 2) + self.pe       # [B,T,C]  image tokens
        kv = self.n1k(kv)
        a, w = self.attn(self.n1(q), kv, kv, need_weights=keep_attn, average_attn_weights=True)
        if keep_attn:
            self.last_attn = w.detach()                                    # [B,T,T]
        x = q + a
        x = x + self.mlp(self.n2(x))
        x = x.transpose(1, 2).reshape(B, C, G // 2, G // 2)
        return bev_l + self.up(x)                                          # residual from LiDAR BEV


class FastBEVFusion(nn.Module):
    def __init__(self, fusion="xattn", dim=32, heads=4):
        super().__init__()
        assert fusion in FUSIONS
        self.fusion, self.dim = fusion, dim
        self.lidar_enc = nn.Sequential(nn.Conv2d(2, dim, 3, padding=1), nn.ReLU(), nn.Conv2d(dim, dim, 3, padding=1), nn.ReLU())
        self.img_enc = nn.Sequential(nn.Conv1d(3, dim, 5, padding=2), nn.ReLU(), nn.Conv1d(dim, dim, 5, padding=2), nn.ReLU())
        self.proj = UniformDepthProjector()
        if fusion == "xattn":
            self.fuse = CrossAttnFusion(dim, heads)
        elif fusion == "concat":
            self.fuse = nn.Conv2d(2 * dim, dim, 1)
        self.head = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.ReLU(), nn.Conv2d(dim, 1, 1))

    def forward(self, lidar, img, yaw=None, keep_attn=False):
        """lidar [B,2,G,G], img [B,3,W_IMG], yaw [B] extrinsic error (rad) -> centre logits [B,1,G,G]."""
        bl = self.lidar_enc(lidar)                                         # [B,C,G,G]
        bi = self.proj(self.img_enc(img), yaw)                             # [B,C,G,G] semi-aligned image BEV
        if self.fusion == "lidar_only": x = bl
        elif self.fusion == "camera_only": x = bi
        elif self.fusion == "add": x = bl + bi
        elif self.fusion == "concat": x = self.fuse(torch.cat([bl, bi], 1))
        else: x = self.fuse(bl, bi, keep_attn)
        return self.head(x)
