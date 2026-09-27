"""
Synthetic agent-trajectory generator standing in for the nuScenes prediction
split.

IMPORTANT: We do NOT have access to nuScenes (or any real dataset) in this
environment. Every trajectory produced here is procedurally generated with
NumPy -- it is NOT real nuScenes data, and no numbers derived from it should
ever be presented as reproducing the paper's reported metrics.

The abstract explicitly calls out three motion regimes where neural
predictors "may fail to reflect physically feasible trajectories":
acceleration, deceleration, and turning. To let `train.py` actually
demonstrate the paper's qualitative claim (EKF candidates complement the
neural predictor most in these regimes), we generate scenes from exactly
three parametric regimes:

    "straight"     - constant velocity, near-zero yaw rate
    "accel_decel"  - constant longitudinal acceleration or deceleration
    "turning"      - constant speed, nonzero constant yaw rate (intersection)

Each ground-truth trajectory is generated analytically from the regime's
kinematic equations (i.e. the *true* generating process really is CV / CA /
CTRV motion), and Gaussian sensor noise is added only to the observed
history window, matching how a real perception stack would report noisy
past positions while the "true" future is clean for evaluation purposes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

REGIMES = ["straight", "accel_decel", "turning"]
REGIME_TO_IDX = {r: i for i, r in enumerate(REGIMES)}


def _generate_positions(regime: str, rng: np.random.Generator, total_len: int, dt: float) -> np.ndarray:
    """Analytically generate a [total_len, 2] ground-truth position sequence
    for the given regime, starting at the origin with a random heading.
    """
    theta0 = rng.uniform(-np.pi, np.pi)
    v0 = rng.uniform(3.0, 12.0)  # m/s (~11-43 km/h), plausible urban speeds
    positions = np.zeros((total_len, 2))
    x, y, theta, v = 0.0, 0.0, theta0, v0

    if regime == "straight":
        for t in range(total_len):
            positions[t] = [x, y]
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt

    elif regime == "accel_decel":
        a = rng.uniform(0.5, 2.5) * rng.choice([-1.0, 1.0])
        for t in range(total_len):
            positions[t] = [x, y]
            x += v * np.cos(theta) * dt + 0.5 * a * np.cos(theta) * dt ** 2
            y += v * np.sin(theta) * dt + 0.5 * a * np.sin(theta) * dt ** 2
            v = max(v + a * dt, 0.5)  # keep speed positive (no reversing)

    elif regime == "turning":
        omega = rng.uniform(0.2, 0.5) * rng.choice([-1.0, 1.0])
        for t in range(total_len):
            positions[t] = [x, y]
            new_theta = theta + omega * dt
            if abs(omega) > 1e-4:
                dx = (v / omega) * (np.sin(new_theta) - np.sin(theta))
                dy = (v / omega) * (-np.cos(new_theta) + np.cos(theta))
            else:
                dx = v * np.cos(theta) * dt
                dy = v * np.sin(theta) * dt
            x += dx
            y += dy
            theta = new_theta
    else:
        raise ValueError(f"unknown regime: {regime}")

    return positions


@dataclass
class Scene:
    history: np.ndarray  # [history_len, 2] noisy observed past positions
    future: np.ndarray   # [future_len, 2]  clean ground-truth future positions
    regime: str
    regime_idx: int


def generate_scene(rng: np.random.Generator, history_len: int, future_len: int, dt: float, noise_std: float) -> Scene:
    regime = REGIMES[rng.integers(0, len(REGIMES))]
    total = history_len + future_len
    positions = _generate_positions(regime, rng, total, dt)
    history_clean = positions[:history_len]
    future = positions[history_len:]
    noise = rng.normal(0.0, noise_std, size=history_clean.shape)
    history = history_clean + noise
    return Scene(history=history, future=future, regime=regime, regime_idx=REGIME_TO_IDX[regime])


class TrajectoryDataset(Dataset):
    """Pre-generates `num_scenes` synthetic scenes at construction time so
    that every epoch sees the same fixed split (reproducible with `seed`).
    """

    def __init__(self, num_scenes: int, history_len: int, future_len: int, dt: float, noise_std: float, seed: int):
        rng = np.random.default_rng(seed)
        self.scenes = [
            generate_scene(rng, history_len, future_len, dt, noise_std)
            for _ in range(num_scenes)
        ]
        self.history_len = history_len
        self.future_len = future_len
        self.dt = dt

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        s = self.scenes[idx]
        return {
            "history": torch.tensor(s.history, dtype=torch.float32),  # [T_hist, 2]
            "future": torch.tensor(s.future, dtype=torch.float32),    # [T_fut, 2]
            "regime_idx": s.regime_idx,
            "history_np": s.history,  # kept as numpy for the (non-torch) EKF bank
        }


def collate_scenes(batch):
    """Custom collate: keeps `history_np` as a python list of numpy arrays
    (needed by the EKF bank, which is pure NumPy) alongside stacked tensors.
    """
    history = torch.stack([b["history"] for b in batch], dim=0)       # [B, T_hist, 2]
    future = torch.stack([b["future"] for b in batch], dim=0)         # [B, T_fut, 2]
    regime_idx = torch.tensor([b["regime_idx"] for b in batch], dtype=torch.long)  # [B]
    history_np = [b["history_np"] for b in batch]
    return {"history": history, "future": future, "regime_idx": regime_idx, "history_np": history_np}
