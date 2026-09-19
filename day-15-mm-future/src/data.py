"""Synthetic driving-scene dataset for the MM-Future reconstruction.

MM-Future (arXiv:2609.20377) is trained on real multi-camera nuScenes-style
video against the NAVSIM / HUGSIM benchmarks. Neither the data nor those
benchmarks are available inside this project's build sessions, so this
module generates a small synthetic stand-in: a top-down bird's-eye-view
(BEV) occupancy grid containing an ego vehicle, a "blocker" agent placed
directly ahead of ego (forcing a genuinely bimodal go-left / go-right
decision — the kind of multi-modal structure the paper's action prior is
built to capture), and several background agents with constant-velocity
motion.

Every scene yields:
  - `history_frames`  : BEV occupancy grids for the past `history_frames` steps
  - `future_frames`   : BEV occupancy grids for the future `future_frames` steps
                         (used only as the flow-matching *scene* target — the
                         paper's own scene representation is "implicit", so we
                         only need something the tokenizer can also encode)
  - `history_cameras` : 4-camera-crop stand-ins for each history frame
  - `future_cameras`  : 4-camera-crop stand-ins for each future frame
  - `action_deltas`   : ground-truth normalized differential ego motion
                         [dx, dy, sin(psi), cos(psi)] for each future step
  - `gt_trajectory`   : cumulative (x, y) ego trajectory for the future horizon
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def _gaussian_blob(grid_size: int, center: np.ndarray, sigma: float) -> np.ndarray:
    """Render a single 2D Gaussian blob on a grid_size x grid_size grid."""
    yy, xx = np.mgrid[0:grid_size, 0:grid_size]
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    return np.exp(-d2 / (2.0 * sigma ** 2)).astype(np.float32)


class SyntheticDrivingScenes:
    """Generates one bimodal go-around-the-blocker driving scene."""

    def __init__(self, grid_size: int, num_agents: int, history_frames: int,
                 future_frames: int, rng: np.random.Generator):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.rng = rng

    def sample(self):
        g = self.grid_size
        total_frames = self.history_frames + self.future_frames
        rng = self.rng

        # Ego starts near the bottom, drives "up" (increasing row index).
        ego_x0 = g / 2.0 + rng.uniform(-1.0, 1.0)
        ego_y0 = 3.0
        speed = rng.uniform(0.9, 1.3)

        # Bimodal decision: a blocker sits ahead in ego's lane. Ego swerves
        # left or right around it starting partway through the future horizon.
        go_left = rng.random() < 0.5
        swerve_sign = -1.0 if go_left else 1.0
        blocker_y = ego_y0 + self.history_frames * speed + self.future_frames * speed * 0.55
        blocker_x = ego_x0

        swerve_start = int(self.future_frames * 0.35)
        swerve_end = int(self.future_frames * 0.85)
        lateral_amp = rng.uniform(3.0, 5.0)

        ego_positions = []
        x, y = ego_x0, ego_y0
        for t in range(total_frames):
            future_t = t - self.history_frames
            if 0 <= future_t < self.future_frames:
                if future_t < swerve_start:
                    lat = 0.0
                elif future_t < swerve_end:
                    frac = (future_t - swerve_start) / max(1, swerve_end - swerve_start)
                    lat = swerve_sign * lateral_amp * (frac ** 1.5)
                else:
                    lat = swerve_sign * lateral_amp
                x = ego_x0 + lat
                y = ego_y0 + speed * (t + 1) + rng.normal(0, 0.03)
            else:
                x = ego_x0
                y = ego_y0 + speed * (t + 1)
            ego_positions.append((x, y))
        ego_positions = np.array(ego_positions, dtype=np.float32)

        # Background agents: constant-velocity, roughly lane-following.
        agent_tracks = []
        for a in range(self.num_agents - 1):
            ax0 = rng.uniform(2, g - 2)
            ay0 = rng.uniform(2, g - 2)
            vx = rng.normal(0, 0.15)
            vy = rng.uniform(-0.6, 0.6)
            track = np.stack([
                np.clip(ax0 + vx * np.arange(total_frames), 0, g - 1),
                np.clip(ay0 + vy * np.arange(total_frames), 0, g - 1),
            ], axis=1).astype(np.float32)
            agent_tracks.append(track)
        # The blocker agent is (almost) static, directly ahead of ego.
        blocker_track = np.tile(
            np.array([blocker_x, blocker_y], dtype=np.float32), (total_frames, 1)
        )
        agent_tracks.append(blocker_track)
        agent_tracks = np.stack(agent_tracks, axis=0)  # [num_agents, T, 2]

        frames = np.zeros((total_frames, 2, g, g), dtype=np.float32)
        for t in range(total_frames):
            other = np.zeros((g, g), dtype=np.float32)
            for a in range(agent_tracks.shape[0]):
                other += _gaussian_blob(g, agent_tracks[a, t], sigma=1.2)
            ego_blob = _gaussian_blob(g, ego_positions[t], sigma=1.0)
            frames[t, 0] = np.clip(other, 0, 1)
            frames[t, 1] = ego_blob

        # Ground-truth differential ego motion over the future horizon.
        full_pos = np.concatenate([[[ego_x0, ego_y0 - speed]], ego_positions], axis=0)
        deltas = np.diff(full_pos, axis=0)[self.history_frames:]  # [future_frames, 2]
        psi = np.arctan2(deltas[:, 1], deltas[:, 0] + 1e-6)
        action_deltas = np.stack(
            [deltas[:, 0], deltas[:, 1], np.sin(psi), np.cos(psi)], axis=1
        ).astype(np.float32)
        gt_trajectory = ego_positions[self.history_frames:].astype(np.float32)
        ego_history = ego_positions[: self.history_frames].astype(np.float32)

        return {
            "frames": frames,                # [T, 2, g, g]
            "action_deltas": action_deltas,  # [future_frames, 4]
            "gt_trajectory": gt_trajectory,   # [future_frames, 2]
            "ego_history": ego_history,       # [history_frames, 2]
            "go_left": go_left,
        }


def extract_camera_crops(frame: np.ndarray, ego_pos: np.ndarray, crop: int) -> np.ndarray:
    """Extract 4 fixed-offset crops (front/rear/left/right) around ego, as a
    deterministic stand-in for the paper's 4 real camera views.

    Args:
        frame: [C, grid_size, grid_size]
        ego_pos: [2] (x, y) in grid coordinates
        crop: crop side length
    Returns:
        [4, C, crop, crop]
    """
    c, g, _ = frame.shape
    pad = crop
    padded = np.pad(frame, ((0, 0), (pad, pad), (pad, pad)), mode="constant")
    ex, ey = int(round(ego_pos[0])) + pad, int(round(ego_pos[1])) + pad
    half = crop // 2
    offsets = {
        "front": (0, crop // 2),
        "rear": (0, -crop // 2),
        "left": (-crop // 2, 0),
        "right": (crop // 2, 0),
    }
    crops = []
    for _, (ox, oy) in offsets.items():
        cx, cy = ex + ox, ey + oy
        patch = padded[:, cy - half: cy + half, cx - half: cx + half]
        if patch.shape[1] != crop or patch.shape[2] != crop:
            fixed = np.zeros((c, crop, crop), dtype=np.float32)
            fixed[:, : patch.shape[1], : patch.shape[2]] = patch
            patch = fixed
        crops.append(patch)
    return np.stack(crops, axis=0)


class MMFutureDataset(Dataset):
    """Pre-generates `num_scenes` synthetic scenes at construction time so
    every epoch trains/evaluates on a fixed, reproducible set."""

    def __init__(self, num_scenes: int, grid_size: int, num_agents: int,
                 history_frames: int, future_frames: int, camera_crop: int,
                 seed: int):
        rng = np.random.default_rng(seed)
        gen = SyntheticDrivingScenes(grid_size, num_agents, history_frames,
                                      future_frames, rng)
        self.camera_crop = camera_crop
        self.history_frames = history_frames
        self.future_frames = future_frames
        self.scenes = [gen.sample() for _ in range(num_scenes)]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        s = self.scenes[idx]
        frames = s["frames"]
        ego_positions = np.concatenate([s["ego_history"], s["gt_trajectory"]], axis=0)

        hist_cams = np.stack([
            extract_camera_crops(frames[t], ego_positions[t], self.camera_crop)
            for t in range(self.history_frames)
        ])  # [Th, 4, 2, crop, crop]
        fut_cams = np.stack([
            extract_camera_crops(frames[self.history_frames + t], ego_positions[self.history_frames + t], self.camera_crop)
            for t in range(self.future_frames)
        ])  # [Tf, 4, 2, crop, crop]

        return {
            "history_cameras": torch.from_numpy(hist_cams).float(),
            "future_cameras": torch.from_numpy(fut_cams).float(),
            "action_deltas": torch.from_numpy(s["action_deltas"]).float(),
            "gt_trajectory": torch.from_numpy(s["gt_trajectory"]).float(),
            "ego_history": torch.from_numpy(s["ego_history"]).float(),
            "frames": torch.from_numpy(frames).float(),
            "go_left": torch.tensor(float(s["go_left"])),
        }
