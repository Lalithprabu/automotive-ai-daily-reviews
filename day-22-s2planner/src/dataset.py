"""
Synthetic multi-camera driving-scene dataset for the S2Planner reconstruction.

Disclosure: the paper (arXiv:2609.29813) trains on real front-camera video
from a NAVSIM-style benchmark, run through a fine-tuned DINOv3 backbone. This
project cannot download that data or those weights in a scheduled run, so we
synthesize scenes directly at the *feature-map* level: instead of rendering
photorealistic camera pixels and running them through DINOv3, we place known
3D landmarks (lane lines + obstacle vehicles) in the ego frame and splat them
into per-camera feature grids using the SAME pinhole projection the model's
cross-attention uses to sample them. This keeps the camera <-> BEV geometric
relationship genuine (the actual mechanism under test) while replacing the
vision-foundation-model backbone with a synthetic stand-in. Every number here
is this project's own default, not paper-sourced.
"""
from __future__ import annotations
import math
import torch
from torch.utils.data import Dataset

from src.geometry import make_camera_rig, project_points

N_CAM = 3
IMG_SIZE = 64
FEAT_CHANNELS = 4  # [ground, left-lane-line, right-lane-line, obstacle]
HIST_LEN = 8
FUT_LEN = 12
DT = 0.5  # seconds per step
LANE_HALF_WIDTH = 1.75

COMMAND_STRAIGHT, COMMAND_LEFT, COMMAND_RIGHT = 0, 1, 2
COMMAND_NAMES = {COMMAND_STRAIGHT: "keep_lane", COMMAND_LEFT: "lane_change_left", COMMAND_RIGHT: "lane_change_right"}


def _unicycle_rollout(x0, y0, theta0, speed, kappa, n_steps, dt):
    """Differentiable-friendly constant-(speed,curvature) unicycle rollout."""
    xs, ys, thetas = [x0], [y0], [theta0]
    x, y, th = x0, y0, theta0
    for _ in range(n_steps):
        th = th + kappa * speed * dt
        x = x + speed * dt * math.cos(th)
        y = y + speed * dt * math.sin(th)
        xs.append(x)
        ys.append(y)
        thetas.append(th)
    return xs, ys, thetas


class SyntheticDrivingSceneDataset(Dataset):
    """
    Each sample is one ego-centric driving scene: a curved lane corridor with
    lane-boundary landmarks and 1-3 obstacle vehicles, an ego motion history,
    a driving command, and a future ground-truth trajectory consistent with
    that command and the lane curvature.
    """

    def __init__(self, n_samples: int, seed: int = 0, image_size: int = IMG_SIZE):
        self.n_samples = n_samples
        self.image_size = image_size
        self.rig = make_camera_rig(image_size=image_size)
        self._rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.n_samples

    def _sample_scene(self, idx: int):
        g = torch.Generator().manual_seed(1000 + idx)

        kappa = (torch.rand((), generator=g).item() - 0.5) * 0.12  # lane curvature (1/m)
        speed = 6.0 + torch.rand((), generator=g).item() * 6.0  # 6-12 m/s
        command = int(torch.randint(0, 3, (1,), generator=g).item())

        # --- past (ego history): roll BACKWARD from the origin so history[0]
        # is furthest in the past, history[-1] == current pose (0,0,0).
        xs, ys, thetas = _unicycle_rollout(0.0, 0.0, 0.0, -speed, -kappa, HIST_LEN, DT)
        hist_xy = torch.tensor(list(zip(xs, ys))[::-1][: HIST_LEN + 1], dtype=torch.float32)
        ego_history = hist_xy[: HIST_LEN]  # (HIST_LEN, 2), oldest..just-before-now

        # --- future ground truth, forward rollout consistent with `command`.
        lane_shift = {COMMAND_STRAIGHT: 0.0, COMMAND_LEFT: 2 * LANE_HALF_WIDTH, COMMAND_RIGHT: -2 * LANE_HALF_WIDTH}[command]
        xs_f, ys_f, thetas_f = _unicycle_rollout(0.0, 0.0, 0.0, speed, kappa, FUT_LEN, DT)
        future = []
        for t in range(1, FUT_LEN + 1):
            # smooth half-cosine lateral-offset profile toward the target lane
            frac = t / FUT_LEN
            shift = lane_shift * 0.5 * (1 - math.cos(math.pi * frac))
            future.append((xs_f[t], ys_f[t] + shift))
        future_gt = torch.tensor(future, dtype=torch.float32)  # (FUT_LEN, 2)

        # --- lane-boundary landmarks along the corridor, from -20m to +40m.
        landmarks, classes, extents = [], [], []
        s_vals = torch.linspace(-20.0, 40.0, 30)
        for s in s_vals.tolist():
            # centerline point at arclength s under constant curvature kappa
            th = kappa * s
            cx = math.sin(th) / kappa if abs(kappa) > 1e-4 else s
            cy = (1 - math.cos(th)) / kappa if abs(kappa) > 1e-4 else 0.0
            nx, ny = -math.sin(th), math.cos(th)  # left-normal direction
            landmarks.append((cx + nx * LANE_HALF_WIDTH, cy + ny * LANE_HALF_WIDTH, 0.0))
            classes.append(1)  # left lane line
            extents.append((0.1, 0.1))
            landmarks.append((cx - nx * LANE_HALF_WIDTH, cy - ny * LANE_HALF_WIDTH, 0.0))
            classes.append(2)  # right lane line
            extents.append((0.1, 0.1))

        # --- obstacle vehicles: place ahead, roughly in-lane, with some jitter.
        n_obstacles = int(torch.randint(1, 4, (1,), generator=g).item())
        for _ in range(n_obstacles):
            s = 8.0 + torch.rand((), generator=g).item() * 28.0
            th = kappa * s
            cx = math.sin(th) / kappa if abs(kappa) > 1e-4 else s
            cy = (1 - math.cos(th)) / kappa if abs(kappa) > 1e-4 else 0.0
            lateral_jitter = (torch.rand((), generator=g).item() - 0.5) * LANE_HALF_WIDTH
            landmarks.append((cx, cy + lateral_jitter, 0.75))  # obstacle "center", z~mid-height
            classes.append(3)
            extents.append((0.9, 2.0))  # half-width, half-length (m) for bbox drawing

        landmarks_xyz = torch.tensor(landmarks, dtype=torch.float32)
        landmark_class = torch.tensor(classes, dtype=torch.long)
        landmark_extent = torch.tensor(extents, dtype=torch.float32)

        return {
            "ego_history": ego_history,
            "command": command,
            "future_gt": future_gt,
            "landmarks_xyz": landmarks_xyz,
            "landmark_class": landmark_class,
            "landmark_extent": landmark_extent,
            "kappa": kappa,
            "speed": speed,
        }

    def _rasterize_cameras(self, landmarks_xyz, landmark_class):
        """Splat landmarks into (n_cam, FEAT_CHANNELS, S, S) synthetic feature grids."""
        S = self.image_size
        pixel_uv, valid = project_points(self.rig, landmarks_xyz)  # (n_cam, N, 2), (n_cam, N)
        grids = torch.zeros(N_CAM, FEAT_CHANNELS, S, S)
        grids[:, 0, :, :] = 0.15  # "ground/sky" background channel, constant prior

        yy, xx = torch.meshgrid(torch.arange(S, dtype=torch.float32), torch.arange(S, dtype=torch.float32), indexing="ij")
        for cam in range(N_CAM):
            for n in range(landmarks_xyz.shape[0]):
                if not valid[cam, n]:
                    continue
                u, v = pixel_uv[cam, n, 0].item(), pixel_uv[cam, n, 1].item()
                if u < -8 or u > S + 8 or v < -8 or v > S + 8:
                    continue
                cls = landmark_class[n].item()
                sigma = 3.0 if cls == 3 else 1.4
                amp = 1.0
                gauss = torch.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * sigma ** 2))
                grids[cam, cls, :, :] = torch.maximum(grids[cam, cls, :, :], amp * gauss)
        return grids

    def __getitem__(self, idx: int):
        scene = self._sample_scene(idx)
        cam_features = self._rasterize_cameras(scene["landmarks_xyz"], scene["landmark_class"])
        command_onehot = torch.zeros(3)
        command_onehot[scene["command"]] = 1.0
        return {
            "cam_features": cam_features,  # (n_cam, C, S, S)
            "ego_history": scene["ego_history"],  # (HIST_LEN, 2)
            "command": command_onehot,  # (3,)
            "future_gt": scene["future_gt"],  # (FUT_LEN, 2)
            # kept for simulate.py visualization only, not used by the model:
            "landmarks_xyz": scene["landmarks_xyz"],
            "landmark_class": scene["landmark_class"],
            "landmark_extent": scene["landmark_extent"],
        }


def collate_scenes(batch):
    out = {}
    for k in ["cam_features", "ego_history", "command", "future_gt"]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["landmarks_xyz"] = [b["landmarks_xyz"] for b in batch]
    out["landmark_class"] = [b["landmark_class"] for b in batch]
    out["landmark_extent"] = [b["landmark_extent"] for b in batch]
    return out
