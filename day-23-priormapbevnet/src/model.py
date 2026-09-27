"""
PriorMapBEVNet -- this project's own disclosed PyTorch reconstruction of the
mechanism described in arXiv:2609.26325 ("Leveraging Vision-Based Point
Cloud Map Priors for Camera-Based 3D Object Detection and Online Vectorized
HD Mapping", Kappeler, Mohan, Valada, Univ. of Freiburg, IROS 2026 Workshop
on Long-Term Perception for Human-Centric Autonomy).

Only the abstract-level description was retrievable for this paper (see
SOURCING.md): a static point-cloud prior map built from previous camera
traversals via Pi3X and augmented with DINOv3 features, retrieved and
encoded with a sparse voxel backbone, fused in BEV with live multi-camera
features, and decoded by task-specific transformer heads for 3D detection
and vectorized HD mapping. Every dimension, the exact fusion mechanism, and
the head design below are this repo's own reconstruction default, not
paper-sourced.

Pipeline
--------
1. SparseVoxelPriorEncoder  : prior point cloud  -> dense prior BEV features
2. CameraBEVLifter          : live multi-camera features -> dense live BEV features
3. PriorMapBEVFusion        : gated fusion of (1) and (2) -> fused BEV features
4. DetectionHead            : fused BEV -> per-cell object heatmap + box regression
5. MapHead                  : fused BEV -> per-cell map-class heatmap (lane / curb)

All BEV tensors are [B, C, H, W] with H = grid.grid_h, W = grid.grid_w.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import CAM_FEAT_DIM, FEAT_DIM, NUM_MAP_CLASSES
from .geometry import BEVGrid, PinholeCamera


class SparseVoxelPriorEncoder(nn.Module):
    """Voxelizes a sparse prior point cloud into a dense BEV feature grid.

    A real sparse-conv backbone (e.g. spconv/MinkowskiEngine) would keep the
    representation sparse through several conv layers. This reconstruction
    keeps the *spirit* (only occupied voxels do any work in the pooling
    step) but uses a plain per-voxel MLP + one dense smoothing conv to fill
    gaps between sparse cells, which is enough to demonstrate the
    memory-prior mechanism at this project's compute budget.
    """

    def __init__(self, feat_dim: int = FEAT_DIM, out_channels: int = 32, grid: BEVGrid = BEVGrid()):
        super().__init__()
        self.grid = grid
        self.point_mlp = nn.Sequential(
            nn.Linear(feat_dim, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels + 1, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, prior_points: list[torch.Tensor]) -> torch.Tensor:
        """prior_points: list of length B, each [N_i, 2 + feat_dim] (x, y, feat...).

        Returns dense prior BEV features, shape [B, out_channels, H, W].
        """
        grid = self.grid
        device = self.point_mlp[0].weight.device
        b = len(prior_points)
        h, w = grid.grid_h, grid.grid_w
        out = torch.zeros(b, self.out_channels, h, w, device=device)
        occ = torch.zeros(b, 1, h, w, device=device)

        for i, pts in enumerate(prior_points):
            if pts.numel() == 0:
                continue
            xy = pts[:, :2].to(device)
            feat = pts[:, 2:].to(device)
            rc = grid.world_to_cell(xy).round().long()  # [N, 2] -> (row, col)
            r, c = rc[:, 0], rc[:, 1]
            valid = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            r, c, feat = r[valid], c[valid], feat[valid]
            if r.numel() == 0:
                continue
            enc = self.point_mlp(feat)  # [N_valid, out_channels]  -- Shape: [N, C_voxel]
            flat_idx = r * w + c  # scatter target per occupied voxel
            voxel_sum = torch.zeros(h * w, self.out_channels, device=device)
            voxel_cnt = torch.zeros(h * w, 1, device=device)
            voxel_sum.index_add_(0, flat_idx, enc)
            voxel_cnt.index_add_(0, flat_idx, torch.ones(r.numel(), 1, device=device))
            voxel_mean = voxel_sum / voxel_cnt.clamp(min=1.0)
            out[i] = voxel_mean.view(h, w, self.out_channels).permute(2, 0, 1)
            occ[i] = (voxel_cnt.view(h, w) > 0).float().unsqueeze(0)

        return self.smooth(torch.cat([out, occ], dim=1))  # Shape: [B, C_voxel, H, W]


class CameraBEVLifter(nn.Module):
    """Lifts live multi-camera feature maps into BEV via known pinhole geometry
    (geometry-guided sampling, echoing the paper's own use of a known camera
    rig to fuse map priors -- but here used to lift the *live* branch)."""

    def __init__(
        self,
        cameras: list[PinholeCamera],
        cam_feat_dim: int = CAM_FEAT_DIM,
        out_channels: int = 32,
        grid: BEVGrid = BEVGrid(),
    ):
        super().__init__()
        self.cameras = cameras
        self.grid = grid
        self.proj = nn.Sequential(
            nn.Linear(cam_feat_dim, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )
        self.out_channels = out_channels
        centers = grid.cell_centers()  # [H, W, 2]
        self.register_buffer("cell_centers", centers, persistent=False)

    def forward(self, camera_feats: list[list[torch.Tensor]]) -> torch.Tensor:
        """camera_feats: list of length B, each a list of per-camera tensors
        [cam_feat_dim, img_h, img_w].

        Returns dense live BEV features, shape [B, out_channels, H, W].
        """
        grid = self.grid
        h, w = grid.grid_h, grid.grid_w
        device = self.proj[0].weight.device
        centers = self.cell_centers.to(device).view(-1, 2)  # [H*W, 2]
        b = len(camera_feats)
        out = torch.zeros(b, self.out_channels, h, w, device=device)

        for i, cams_feats in enumerate(camera_feats):
            accum = torch.zeros(h * w, self.out_channels, device=device)
            weight = torch.zeros(h * w, 1, device=device)
            for cam, feat in zip(self.cameras, cams_feats):
                feat = feat.to(device)
                u, v, valid = cam.world_to_pixel(centers)  # each [H*W]
                # normalize to [-1, 1] for grid_sample
                gx = (u / (cam.img_w - 1)) * 2 - 1
                gy = (v / (cam.img_h - 1)) * 2 - 1
                grid_xy = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
                sampled = F.grid_sample(
                    feat.unsqueeze(0), grid_xy, align_corners=True, mode="bilinear",
                    padding_mode="zeros",
                )  # [1, cam_feat_dim, H*W, 1]
                sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)  # [H*W, cam_feat_dim]
                v_mask = valid.float().unsqueeze(-1)
                accum += self.proj(sampled) * v_mask
                weight += v_mask
            mean = accum / weight.clamp(min=1.0)
            out[i] = mean.view(h, w, self.out_channels).permute(2, 0, 1)

        return out  # Shape: [B, C_cam_bev, H, W]


class PriorMapBEVFusion(nn.Module):
    """Gated fusion of live and prior-map BEV features.

    A learned per-cell gate decides, cell by cell, how much to trust the
    (potentially stale, but complete) prior versus the (fresh, but
    occluded) live observation -- this repo's own reconstruction of the
    paper's "fuse ... in bird's-eye view" step.
    """

    def __init__(self, channels: int = 32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, live_bev: torch.Tensor, prior_bev: torch.Tensor, prior_enabled: bool = True) -> torch.Tensor:
        """live_bev, prior_bev: [B, C, H, W]. Returns fused [B, C, H, W]."""
        if not prior_enabled:
            prior_bev = torch.zeros_like(prior_bev)
        cat = torch.cat([live_bev, prior_bev], dim=1)  # Shape: [B, 2C, H, W]
        g = self.gate(cat)  # Shape: [B, 1, H, W] -- per-cell trust-the-prior weight
        gated_prior = g * prior_bev
        fused_in = torch.cat([live_bev, gated_prior], dim=1)
        residual = self.fuse(fused_in)
        return self.out_relu(live_bev + residual)  # Shape: [B, C, H, W]


class DetectionHead(nn.Module):
    """CenterNet-style per-cell detection head: 1-channel objectness heatmap
    + 6-channel box regression (dx, dy, w, l, sin(heading), cos(heading))."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.heatmap = nn.Conv2d(channels, 1, 1)
        self.boxes = nn.Conv2d(channels, 6, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.trunk(x)
        return torch.sigmoid(self.heatmap(t)), self.boxes(t)  # [B,1,H,W], [B,6,H,W]


class MapHead(nn.Module):
    """Per-cell map-element classification head (lane / curb), standing in
    for the paper's vectorized-polyline transformer head. `simulate.py`
    extracts approximate polylines from this heatmap for visualization."""

    def __init__(self, channels: int = 32, num_classes: int = NUM_MAP_CLASSES):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.cls(self.trunk(x)))  # [B, num_classes, H, W]


class PriorMapBEVNet(nn.Module):
    def __init__(self, cameras: list[PinholeCamera], grid: BEVGrid = BEVGrid(), channels: int = 32):
        super().__init__()
        self.grid = grid
        self.prior_encoder = SparseVoxelPriorEncoder(out_channels=channels, grid=grid)
        self.cam_lifter = CameraBEVLifter(cameras, out_channels=channels, grid=grid)
        self.fusion = PriorMapBEVFusion(channels=channels)
        self.det_head = DetectionHead(channels=channels)
        self.map_head = MapHead(channels=channels)

    def forward(
        self,
        prior_points: list[torch.Tensor],
        camera_feats: list[list[torch.Tensor]],
        prior_enabled: bool = True,
    ):
        live_bev = self.cam_lifter(camera_feats)  # [B, C, H, W]
        prior_bev = self.prior_encoder(prior_points)  # [B, C, H, W]
        fused = self.fusion(live_bev, prior_bev, prior_enabled=prior_enabled)  # [B, C, H, W]
        det_heatmap, det_boxes = self.det_head(fused)
        map_heatmap = self.map_head(fused)
        return {
            "det_heatmap": det_heatmap,
            "det_boxes": det_boxes,
            "map_heatmap": map_heatmap,
            "fused_bev": fused,
        }
