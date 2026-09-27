"""
S2Planner reconstruction (arXiv:2609.29813, "S2Planner: Multi-Scale Semantic
Planner for End-to-End Autonomous Driving", Lu et al., TU Munich / Huaibei
Normal University, submitted 2026-09-24).

PAPER-SOURCED (via alphaxiv.org mirror -- direct arXiv fetch was HTTP 429'd,
the same recurring wall hit on most days since this project's Day 9):
  - Three front-facing cameras + ego-motion history + driving command as input.
  - A fine-tuned DINOv3 visual backbone with a "Spatial Tuning Adapter" producing
    multi-scale image representations.
  - A coarse-to-fine decoder using trajectory self-attention and
    camera-projected cross-attention.
  - Ego-conditioned trajectory initialization + iterative, geometry-guided
    sampling of multi-scale image features as the core refinement idea.
  - One headline number: 88.03 PDMS on NAVSIM v1 navtest -- which the authors
    themselves flag as an exploratory, run-selected result, not an unbiased
    estimate. Quoted here with that same caveat attached.

THIS PROJECT'S OWN RECONSTRUCTION (everything below is a disclosed default,
not paper-sourced -- no architecture diagram, dimensions, loss function, or
results table beyond the abstract was recoverable):
  - DINOv3 itself is a multi-billion-parameter foundation model this project
    cannot download or fine-tune in a scheduled run. `SyntheticBackboneStem`
    is a small from-scratch CNN standing in for "the fine-tuned DINOv3
    backbone" -- it is NOT DINOv3 and makes no claim to reproduce its
    features, only to produce *some* multi-channel feature grid for the
    adapter and cross-attention mechanism to operate on.
  - All layer widths, the exact adapter bottleneck ratio, the initializer's
    unicycle parameterization, and the cross-attention's per-camera softmax
    formulation are this project's own design choices.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geometry import make_camera_rig, project_points, normalize_pixels_for_grid_sample
from src.dataset import N_CAM, FEAT_CHANNELS, HIST_LEN, FUT_LEN, DT, IMG_SIZE


class SyntheticBackboneStem(nn.Module):
    """Small CNN stand-in for "a fine-tuned DINOv3 backbone". NOT DINOv3."""

    def __init__(self, in_ch: int = FEAT_CHANNELS, d_feat: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, d_feat, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_feat, d_feat, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: (B * n_cam, in_ch, S, S) -> (B * n_cam, d_feat, S, S)
        return self.net(x)


class SpatialTuningAdapter(nn.Module):
    """
    Lightweight bottleneck adapter (down-proj -> GELU -> up-proj -> scaled
    residual), applied independently at each pyramid scale. Stands in for
    "fine-tuning DINOv3 with a Spatial Tuning Adapter" instead of full
    fine-tuning -- one adapter instance per scale, per this reconstruction.
    """

    def __init__(self, d_feat: int = 32, bottleneck: int = 8):
        super().__init__()
        self.down = nn.Conv2d(d_feat, bottleneck, kernel_size=1)
        self.up = nn.Conv2d(bottleneck, d_feat, kernel_size=1)
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        delta = self.up(self.act(self.down(x)))
        return x + self.scale * delta


class MultiScaleCameraEncoder(nn.Module):
    """
    Backbone stem + per-scale Spatial Tuning Adapters, producing a 3-level
    feature pyramid per camera: fine (S), mid (S/2), coarse (S/4).
    """

    def __init__(self, d_feat: int = 32, n_scales: int = 3):
        super().__init__()
        self.stem = SyntheticBackboneStem(FEAT_CHANNELS, d_feat)
        self.adapters = nn.ModuleList([SpatialTuningAdapter(d_feat) for _ in range(n_scales)])
        self.n_scales = n_scales
        self.d_feat = d_feat

    def forward(self, cam_features):
        # cam_features: (B, n_cam, C, S, S)
        B, n_cam, C, S, _ = cam_features.shape
        x = cam_features.reshape(B * n_cam, C, S, S)
        feat_full = self.stem(x)  # (B*n_cam, d_feat, S, S)

        pyramid = []
        cur = feat_full
        for i in range(self.n_scales):
            adapted = self.adapters[i](cur)
            pyramid.append(adapted.reshape(B, n_cam, self.d_feat, *adapted.shape[-2:]))
            if i < self.n_scales - 1:
                cur = F.avg_pool2d(cur, kernel_size=2)
        # pyramid[0] = finest (S x S) ... pyramid[-1] = coarsest (S/4 x S/4)
        return pyramid


class EgoConditionedInitializer(nn.Module):
    """
    Ego-conditioned, geometry-aware coarse trajectory initialization: predicts
    a per-step (speed, curvature) profile from ego-motion history + driving
    command, then differentiably integrates a unicycle model to obtain an
    initial (x, y) waypoint sequence -- a real kinematic rollout, not a raw
    coordinate regression.
    """

    def __init__(self, d_hidden: int = 64, fut_len: int = FUT_LEN, dt: float = DT):
        super().__init__()
        self.fut_len = fut_len
        self.dt = dt
        in_dim = HIST_LEN * 2 + 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, fut_len * 2),  # (speed_t, kappa_t) per step
        )

    def forward(self, ego_history, command_onehot):
        B = ego_history.shape[0]
        flat_hist = ego_history.reshape(B, -1)
        inp = torch.cat([flat_hist, command_onehot], dim=-1)
        params = self.mlp(inp).view(B, self.fut_len, 2)
        speed = 2.0 + 10.0 * torch.sigmoid(params[..., 0])  # keep speeds in a plausible range
        kappa = 0.15 * torch.tanh(params[..., 1])  # bounded curvature

        x = torch.zeros(B, device=ego_history.device)
        y = torch.zeros(B, device=ego_history.device)
        theta = torch.zeros(B, device=ego_history.device)
        waypoints = []
        for t in range(self.fut_len):
            theta = theta + kappa[:, t] * speed[:, t] * self.dt
            x = x + speed[:, t] * self.dt * torch.cos(theta)
            y = y + speed[:, t] * self.dt * torch.sin(theta)
            waypoints.append(torch.stack([x, y], dim=-1))
        return torch.stack(waypoints, dim=1)  # (B, fut_len, 2)


class CameraProjectedCrossAttention(nn.Module):
    """
    For every trajectory point's current (x, y) estimate, project into all
    `n_cam` cameras via the pinhole rig, bilinearly sample that scale's
    per-camera feature map at the projected pixel, then run a small
    attention over the (up to n_cam) camera observations per point.
    Points that project outside every camera's view fall back to a learned
    "unobserved" embedding rather than a zero vector.
    """

    def __init__(self, d_feat: int, rig: dict):
        super().__init__()
        self.rig = rig
        self.d_feat = d_feat
        self.q_proj = nn.Linear(d_feat, d_feat)
        self.k_proj = nn.Linear(d_feat, d_feat)
        self.v_proj = nn.Linear(d_feat, d_feat)
        self.out_proj = nn.Linear(d_feat, d_feat)
        self.unobserved = nn.Parameter(torch.randn(d_feat) * 0.02)

    def forward(self, point_feat, point_xy, scale_pyramid_level):
        """
        point_feat: (B, T, D)
        point_xy:   (B, T, 2) current BEV coordinate estimate
        scale_pyramid_level: (B, n_cam, D, Hs, Ws) feature maps for this scale
        """
        B, T, D = point_feat.shape
        n_cam = scale_pyramid_level.shape[1]
        Hs = scale_pyramid_level.shape[-2]

        xyz = torch.cat([point_xy, torch.zeros(B, T, 1, device=point_xy.device)], dim=-1)
        pixel_uv, valid = project_points(self.rig, xyz)  # (n_cam, B, T, 2), (n_cam, B, T)
        grid = normalize_pixels_for_grid_sample(pixel_uv, Hs)  # (n_cam, B, T, 2)
        in_bounds = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)
        valid = valid & in_bounds  # (n_cam, B, T)

        sampled = []
        for c in range(n_cam):
            g = grid[c].unsqueeze(1)  # (B, 1, T, 2)
            feat_map = scale_pyramid_level[:, c]  # (B, D, Hs, Ws)
            s = F.grid_sample(feat_map, g, mode="bilinear", padding_mode="zeros", align_corners=True)
            s = s.squeeze(2).permute(0, 2, 1)  # (B, T, D)
            mask = valid[c].unsqueeze(-1)  # (B, T, 1)
            s = torch.where(mask, s, self.unobserved.view(1, 1, -1).expand_as(s))
            sampled.append(s)
        sampled = torch.stack(sampled, dim=2)  # (B, T, n_cam, D)
        cam_valid = valid.permute(1, 2, 0)  # (B, T, n_cam)

        q = self.q_proj(point_feat)  # (B, T, D)
        k = self.k_proj(sampled)  # (B, T, n_cam, D)
        v = self.v_proj(sampled)  # (B, T, n_cam, D)

        logits = (q.unsqueeze(2) * k).sum(-1) / math.sqrt(D)  # (B, T, n_cam)
        # If a point is invisible to every camera, attend uniformly rather than div-by-zero.
        any_valid = cam_valid.any(dim=-1, keepdim=True)
        logits = logits.masked_fill(~cam_valid & any_valid, float("-1e4"))
        attn = torch.softmax(logits, dim=-1)
        out = (attn.unsqueeze(-1) * v).sum(2)  # (B, T, D)
        return self.out_proj(out), attn


class CoarseToFineDecoderLayer(nn.Module):
    def __init__(self, d_feat: int, rig: dict, n_heads: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_feat, n_heads, batch_first=True)
        self.cross_attn = CameraProjectedCrossAttention(d_feat, rig)
        self.norm1 = nn.LayerNorm(d_feat)
        self.norm2 = nn.LayerNorm(d_feat)
        self.norm3 = nn.LayerNorm(d_feat)
        self.ffn = nn.Sequential(nn.Linear(d_feat, d_feat * 2), nn.GELU(), nn.Linear(d_feat * 2, d_feat))
        self.offset_head = nn.Sequential(nn.Linear(d_feat, d_feat // 2), nn.GELU(), nn.Linear(d_feat // 2, 2))

    def forward(self, point_feat, point_xy, scale_level):
        sa_out, _ = self.self_attn(point_feat, point_feat, point_feat)
        point_feat = self.norm1(point_feat + sa_out)

        ca_out, attn_weights = self.cross_attn(point_feat, point_xy, scale_level)
        point_feat = self.norm2(point_feat + ca_out)

        point_feat = self.norm3(point_feat + self.ffn(point_feat))

        offset = self.offset_head(point_feat)  # (B, T, 2), coarse-to-fine refinement step
        new_xy = point_xy + offset
        return point_feat, new_xy, attn_weights


class S2Planner(nn.Module):
    def __init__(self, d_feat: int = 32, n_scales: int = 3, fut_len: int = FUT_LEN):
        super().__init__()
        self.rig = make_camera_rig(image_size=IMG_SIZE)
        self.encoder = MultiScaleCameraEncoder(d_feat=d_feat, n_scales=n_scales)
        self.initializer = EgoConditionedInitializer(fut_len=fut_len)
        # One decoder layer per pyramid level, coarse (index -1) -> fine (index 0).
        self.decoder_layers = nn.ModuleList([CoarseToFineDecoderLayer(d_feat, self.rig) for _ in range(n_scales)])
        self.point_pos_emb = nn.Parameter(torch.randn(fut_len, d_feat) * 0.02)
        self.xy_to_feat = nn.Linear(2, d_feat)
        self.fut_len = fut_len
        self.d_feat = d_feat

    def forward(self, cam_features, ego_history, command_onehot):
        B = cam_features.shape[0]
        pyramid = self.encoder(cam_features)  # fine -> coarse
        coarse_first = list(reversed(pyramid))  # coarse -> fine, matches "coarse-to-fine"

        init_xy = self.initializer(ego_history, command_onehot)  # (B, T, 2)
        point_feat = self.point_pos_emb.unsqueeze(0).expand(B, -1, -1) + self.xy_to_feat(init_xy)

        xy = init_xy
        stage_trajectories = [init_xy]
        last_attn = None
        for layer, scale_level in zip(self.decoder_layers, coarse_first):
            point_feat, xy, last_attn = layer(point_feat, xy, scale_level)
            stage_trajectories.append(xy)

        return {
            "trajectory": xy,  # (B, T, 2) final, fine-stage prediction
            "stage_trajectories": stage_trajectories,  # coarse init -> ... -> final
            "last_cross_attn": last_attn,
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
