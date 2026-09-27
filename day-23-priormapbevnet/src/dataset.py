"""
Synthetic ego-centric driving-scene dataset for PriorMapBEVNet.

Every element below (world layout, occlusion model, feature dimensions,
class embeddings) is this project's own disclosed synthetic reconstruction,
NOT drawn from the source paper (arXiv:2609.26325) -- see SOURCING.md.

Scene design, deliberately built to test the paper's actual claim:
  - A "prior traversal" of the scene produced a *static* point-cloud map
    (lane boundaries + curbs only -- never the dynamic vehicles, since a
    static map cannot contain them) with DINOv3-like per-point semantic
    features. This map is available to the model as `prior_points`.
  - The *live* pass through the same scene sees the world through three
    noisy cameras. Camera-only depth/parallax ambiguity is simulated by
    randomly occluding a large fraction of the static lane/curb geometry
    in the live camera views each scene (a live-only model must guess the
    missing structure). Dynamic vehicles are always fully visible live
    (they cannot appear in the static prior map at all).

This directly reproduces the paper's qualitative pattern: fusing the prior
should help *mapping* a lot (the prior supplies geometry the live cameras
occluded) and help *detection* only a little (the prior never contains the
moving objects the detector must find).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .geometry import BEVGrid, default_camera_rig

MAP_CLASSES = ["lane", "curb"]
NUM_MAP_CLASSES = len(MAP_CLASSES)
FEAT_DIM = 16  # stand-in for a reduced DINOv3 descriptor (real DINOv3 is 768-d+)
CAM_FEAT_DIM = 16  # stand-in for a live-image backbone's per-pixel feature channels


def _class_embedding(seed: int, dim: int = FEAT_DIM) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


LANE_EMB = _class_embedding(1)
CURB_EMB = _class_embedding(2)
VEHICLE_EMB = _class_embedding(3)


def _polyline_points(x0: float, curvature: float, n: int, y_max: float) -> torch.Tensor:
    """A gently curved polyline running forward in y, offset in x. Shape [n, 2]."""
    y = torch.linspace(0.5, y_max, n)
    x = x0 + curvature * (y ** 2) / (y_max ** 2)
    return torch.stack([x, y], dim=-1)


@dataclass
class Scene:
    prior_points: torch.Tensor  # [N, 2 + FEAT_DIM]  (x, y, feat...)
    camera_feats: list[torch.Tensor]  # per-camera [CAM_FEAT_DIM, img_h, img_w]
    vehicles: torch.Tensor  # [M, 5] -> (x, y, w, l, heading_rad)
    det_heatmap: torch.Tensor  # [1, H, W]
    det_boxes: torch.Tensor  # [5, H, W] -> dx, dy, w, l, (sin,cos) packed as 6? see below
    det_mask: torch.Tensor  # [1, H, W] 1 at object center cells
    map_heatmap: torch.Tensor  # [NUM_MAP_CLASSES, H, W] full (unoccluded) ground truth


class SyntheticSceneDataset(Dataset):
    def __init__(self, num_scenes: int, grid: BEVGrid, seed: int = 0, occlusion: float = 0.65):
        self.num_scenes = num_scenes
        self.grid = grid
        self.seed = seed
        self.occlusion = occlusion
        self.cameras = default_camera_rig()

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, idx: int) -> Scene:
        g = torch.Generator().manual_seed(self.seed * 100_003 + idx)
        grid = self.grid
        y_max = grid.ego_row * grid.cell_size

        # --- static map layout: 2 lane boundaries + 1 curb, mildly curved ---
        curvature = (torch.rand(1, generator=g).item() - 0.5) * 10.0
        lane_left = _polyline_points(-3.5, curvature, 60, y_max)
        lane_right = _polyline_points(3.5, curvature, 60, y_max)
        curb = _polyline_points(-7.0, curvature, 60, y_max)
        polylines = [(lane_left, LANE_EMB), (lane_right, LANE_EMB), (curb, CURB_EMB)]

        # Full (unoccluded) ground-truth map raster -- used only as supervision target.
        map_heatmap = torch.zeros(NUM_MAP_CLASSES, grid.grid_h, grid.grid_w)
        for pts, emb in polylines:
            cls = 0 if torch.allclose(emb, LANE_EMB) else 1
            rc = grid.world_to_cell(pts).round().long()
            for r, c in rc.tolist():
                if 0 <= r < grid.grid_h and 0 <= c < grid.grid_w:
                    map_heatmap[cls, r, c] = 1.0

        # --- prior point cloud: ALL static points, small SfM-style noise, never occluded ---
        prior_pts_list = []
        for pts, emb in polylines:
            noise = torch.randn_like(pts, generator=None) * 0.05  # small reconstruction noise
            noisy = pts + noise
            feat = emb.unsqueeze(0).expand(noisy.shape[0], -1) + 0.02 * torch.randn(
                noisy.shape[0], FEAT_DIM
            )
            prior_pts_list.append(torch.cat([noisy, feat], dim=-1))
        prior_points = torch.cat(prior_pts_list, dim=0)

        # --- live-visible static points: same polylines but heavily occluded ---
        live_static_list = []
        for pts, _emb in polylines:
            keep = torch.rand(pts.shape[0], generator=g) > self.occlusion
            live_static_list.append(pts[keep])

        # --- dynamic vehicles: random boxes, never in the prior map ---
        n_veh = int(torch.randint(2, 5, (1,), generator=g).item())
        vx = (torch.rand(n_veh, generator=g) - 0.5) * (grid.grid_w * grid.cell_size * 0.8)
        vy = torch.rand(n_veh, generator=g) * y_max * 0.85 + 1.0
        vw = torch.full((n_veh,), 1.8)
        vl = torch.full((n_veh,), 4.2)
        vh = torch.rand(n_veh, generator=g) * math.pi - math.pi / 2
        vehicles = torch.stack([vx, vy, vw, vl, vh], dim=-1)

        det_heatmap = torch.zeros(1, grid.grid_h, grid.grid_w)
        det_mask = torch.zeros(1, grid.grid_h, grid.grid_w)
        det_boxes = torch.zeros(6, grid.grid_h, grid.grid_w)  # dx, dy, w, l, sin(h), cos(h)
        yy, xx = torch.meshgrid(
            torch.arange(grid.grid_h).float(), torch.arange(grid.grid_w).float(), indexing="ij"
        )
        for x, y, w, length, heading in vehicles.tolist():
            rc = grid.world_to_cell(torch.tensor([x, y]))
            r0, c0 = rc[0].item(), rc[1].item()
            r_i, c_i = int(round(r0)), int(round(c0))
            dist2 = (yy - r0) ** 2 + (xx - c0) ** 2
            det_heatmap[0] = torch.maximum(det_heatmap[0], torch.exp(-dist2 / (2 * 1.2 ** 2)))
            if 0 <= r_i < grid.grid_h and 0 <= c_i < grid.grid_w:
                det_heatmap[0, r_i, c_i] = 1.0  # force exact center to 1.0 (CenterNet convention)
                det_mask[0, r_i, c_i] = 1.0
                det_boxes[0, r_i, c_i] = r0 - r_i
                det_boxes[1, r_i, c_i] = c0 - c_i
                det_boxes[2, r_i, c_i] = w
                det_boxes[3, r_i, c_i] = length
                det_boxes[4, r_i, c_i] = math.sin(heading)
                det_boxes[5, r_i, c_i] = math.cos(heading)

        # --- render live camera feature maps by splatting visible static + dynamic points ---
        camera_feats = []
        for cam in self.cameras:
            feat = torch.zeros(CAM_FEAT_DIM, cam.img_h, cam.img_w)
            for pts in live_static_list:
                if pts.numel() == 0:
                    continue
                u, v, valid = cam.world_to_pixel(pts)
                _splat(feat, u[valid], v[valid], LANE_EMB)
            if n_veh > 0:
                u, v, valid = cam.world_to_pixel(vehicles[:, :2])
                _splat(feat, u[valid], v[valid], VEHICLE_EMB)
            feat = feat + 0.05 * torch.randn_like(feat)  # sensor noise
            camera_feats.append(feat)

        return Scene(
            prior_points=prior_points,
            camera_feats=camera_feats,
            vehicles=vehicles,
            det_heatmap=det_heatmap,
            det_boxes=det_boxes,
            det_mask=det_mask,
            map_heatmap=map_heatmap,
        )


def _splat(feat: torch.Tensor, u: torch.Tensor, v: torch.Tensor, emb: torch.Tensor, radius: int = 1):
    c, h, w = feat.shape
    for uu, vv in zip(u.tolist(), v.tolist()):
        ui, vi = int(round(uu)), int(round(vv))
        for dv in range(-radius, radius + 1):
            for du in range(-radius, radius + 1):
                pu, pv = ui + du, vi + dv
                if 0 <= pu < w and 0 <= pv < h:
                    feat[:, pv, pu] = torch.maximum(feat[:, pv, pu], emb)


def collate_scenes(batch: list[Scene]) -> dict:
    """Collates a list of Scene objects. Point-cloud sizes vary, so prior
    points are kept as a python list (per-scene) rather than stacked."""
    return {
        "prior_points": [s.prior_points for s in batch],
        "camera_feats": [s.camera_feats for s in batch],
        "det_heatmap": torch.stack([s.det_heatmap for s in batch]),
        "det_boxes": torch.stack([s.det_boxes for s in batch]),
        "det_mask": torch.stack([s.det_mask for s in batch]),
        "map_heatmap": torch.stack([s.map_heatmap for s in batch]),
        "vehicles": [s.vehicles for s in batch],
    }
