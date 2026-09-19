"""
Synthetic multi-view-camera -> BEV 3D detection task.

There is no public nuScenes-scale multi-camera rig available in this
CPU-only reconstruction environment, so a lightweight synthetic generator
stands in: a handful of car-like boxes are scattered in a BEV plane, then
projected into ``num_cameras`` synthetic camera views arranged in a ring
around the ego vehicle (an evenly-spaced-azimuth simplification of a real
surround-camera rig, NOT a physically calibrated camera model). Ground
truth BEV grid targets (per-cell objectness + box regression) are derived
directly from the same object list used to render the images, so the task
is internally consistent and learnable -- which is exactly what the
overfit sanity check in train.py verifies.

All dimensions/constants here (image size, FOV, anchor box size, etc.)
are this project's own reconstruction defaults, not paper-sourced.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

CAR_W = 2.0  # anchor width (m), reconstruction default
CAR_L = 4.0  # anchor length (m), reconstruction default


def make_scene(bev_range: float, num_objects_min: int, num_objects_max: int, rng: np.random.Generator):
    n = int(rng.integers(num_objects_min, num_objects_max + 1))
    objs = []
    for _ in range(n):
        x = float(rng.uniform(-bev_range * 0.9, bev_range * 0.9))
        y = float(rng.uniform(-bev_range * 0.9, bev_range * 0.9))
        heading = float(rng.uniform(-np.pi, np.pi))
        w = CAR_W * float(rng.uniform(0.8, 1.2))
        l = CAR_L * float(rng.uniform(0.8, 1.2))
        objs.append(dict(x=x, y=y, heading=heading, w=w, l=l))
    return objs


def render_cameras(objs, num_cameras: int, image_size: int, rng: np.random.Generator, noise_std: float = 0.05):
    """Render `num_cameras` synthetic grayscale-ish RGB views of the scene.

    Each camera sits at a fixed azimuth around the ego vehicle (ring
    layout). Objects in front of a camera and within its FOV are drawn as
    a solid, depth-scaled patch; objects behind or outside the FOV are
    simply not visible to that camera (this is what makes camera-view
    routing meaningful -- not every object is visible in every view).
    """
    imgs = np.full((num_cameras, 3, image_size, image_size), 0.1, dtype=np.float32)
    imgs += rng.normal(0, noise_std, imgs.shape).astype(np.float32)
    focal = image_size * 0.8
    fov_half = np.deg2rad(50)

    for c in range(num_cameras):
        azimuth = c * (2 * np.pi / num_cameras)
        cos_a, sin_a = np.cos(-azimuth), np.sin(-azimuth)
        # Sort by depth (far to near) so nearer objects draw on top -- cheap z-buffer.
        cam_objs = []
        for obj in objs:
            xc = obj["x"] * cos_a - obj["y"] * sin_a
            yc = obj["x"] * sin_a + obj["y"] * cos_a
            if yc <= 0.5:
                continue
            angle = np.arctan2(xc, yc)
            if abs(angle) > fov_half:
                continue
            cam_objs.append((yc, xc, obj))
        cam_objs.sort(key=lambda t: -t[0])

        for yc, xc, obj in cam_objs:
            u = image_size / 2 + (xc / yc) * focal
            v = image_size / 2 + (2.0 / yc) * focal * 0.3
            size = max(3, int(min(24, focal * obj["w"] / yc)))
            u0, u1 = int(u - size / 2), int(u + size / 2)
            v0, v1 = int(v - size / 2), int(v + size / 2)
            u0c, u1c = max(0, u0), min(image_size, u1)
            v0c, v1c = max(0, v0), min(image_size, v1)
            if u1c > u0c and v1c > v0c:
                intensity = float(min(1.0, 1.5 / yc + 0.3))
                imgs[c, 0, v0c:v1c, u0c:u1c] = intensity
                imgs[c, 1, v0c:v1c, u0c:u1c] = intensity * 0.6
                imgs[c, 2, v0c:v1c, u0c:u1c] = intensity * 0.3

    return np.clip(imgs, 0.0, 1.0)


def make_bev_target(objs, bev_size: int, bev_range: float):
    """Derive (objectness_grid, box_regression_grid) from the object list.

    objectness_grid: (bev_size, bev_size) in {0, 1}
    box_grid:        (bev_size, bev_size, 6) = [dx, dy, log(w/W0), log(l/L0), sin(h), cos(h)]
                      dx, dy are the object center's offset from its cell
                      center, normalized by cell size (roughly in [-0.5, 0.5]).
    """
    cell = 2 * bev_range / bev_size
    obj_grid = np.zeros((bev_size, bev_size), dtype=np.float32)
    box_grid = np.zeros((bev_size, bev_size, 6), dtype=np.float32)
    for obj in objs:
        gx = int((obj["x"] + bev_range) / cell)
        gy = int((obj["y"] + bev_range) / cell)
        if 0 <= gx < bev_size and 0 <= gy < bev_size:
            obj_grid[gy, gx] = 1.0
            cx = -bev_range + (gx + 0.5) * cell
            cy = -bev_range + (gy + 0.5) * cell
            dx = (obj["x"] - cx) / cell
            dy = (obj["y"] - cy) / cell
            dw = np.log(obj["w"] / CAR_W)
            dl = np.log(obj["l"] / CAR_L)
            box_grid[gy, gx] = [dx, dy, dw, dl, np.sin(obj["heading"]), np.cos(obj["heading"])]
    return obj_grid, box_grid


class SparseBEVDataset(Dataset):
    """Deterministic (seeded per-index) synthetic multi-view BEV dataset."""

    def __init__(self, num_samples: int, config: dict, seed: int = 0):
        self.num_samples = num_samples
        self.config = config
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        objs = make_scene(
            self.config["bev_range"], self.config["num_objects_min"], self.config["num_objects_max"], rng
        )
        imgs = render_cameras(objs, self.config["num_cameras"], self.config["image_size"], rng)
        obj_grid, box_grid = make_bev_target(objs, self.config["bev_size"], self.config["bev_range"])
        return {
            "images": torch.from_numpy(imgs).float(),
            "obj_grid": torch.from_numpy(obj_grid).float(),
            "box_grid": torch.from_numpy(box_grid).float(),
        }


def generate_moving_sequence(num_frames: int, config: dict, seed: int = 12345):
    """Generate a temporally-coherent sequence for simulate.py: one base
    scene whose objects drift with small per-frame velocities, re-rendered
    (with fresh camera noise each frame) at every step.

    Returns a list of dicts, one per frame, each shaped like a
    SparseBEVDataset item (images / obj_grid / box_grid), plus the raw
    object list for that frame (used to overlay ground-truth boxes).
    """
    rng = np.random.default_rng(seed)
    objs = make_scene(config["bev_range"], config["num_objects_min"], config["num_objects_max"], rng)
    velocities = [
        dict(vx=rng.uniform(-0.4, 0.4), vy=rng.uniform(-0.4, 0.4), vh=rng.uniform(-0.05, 0.05))
        for _ in objs
    ]

    frames = []
    bev_range = config["bev_range"]
    for _t in range(num_frames):
        # Advance positions, clip to stay inside the BEV range.
        for obj, vel in zip(objs, velocities):
            obj["x"] = float(np.clip(obj["x"] + vel["vx"], -bev_range * 0.9, bev_range * 0.9))
            obj["y"] = float(np.clip(obj["y"] + vel["vy"], -bev_range * 0.9, bev_range * 0.9))
            obj["heading"] = float(obj["heading"] + vel["vh"])

        imgs = render_cameras(objs, config["num_cameras"], config["image_size"], rng, noise_std=0.07)
        obj_grid, box_grid = make_bev_target(objs, config["bev_size"], config["bev_range"])
        frames.append(
            {
                "images": torch.from_numpy(imgs).float(),
                "obj_grid": torch.from_numpy(obj_grid).float(),
                "box_grid": torch.from_numpy(box_grid).float(),
                "objects": [dict(o) for o in objs],
            }
        )
    return frames
