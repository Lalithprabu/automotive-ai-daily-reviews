"""
src/dataset.py

Synthetic vehicle trajectory generator used to train ProbabilisticMotionPredictor.

SOURCING DISCLOSURE: the paper does not publish or reference a specific
dataset -- this is entirely this repo's own synthetic generator, a CTRV
("constant turn-rate and velocity")-style kinematic model with random-walk
process noise on acceleration and yaw rate, sampled at a fixed BSM cadence
(default 100ms, i.e. the standard SAE J2735 BSM broadcast rate). Each
generated trajectory is treated as one independent vehicle's short kinematic
history -> future window; "multi-agent" here just means many independently
sampled vehicles/trajectories, not modeled interaction between them (a real
V2X predictor would condition on neighboring vehicles too -- out of scope
for this reconstruction, which focuses on the single-vehicle BSM
recovery mechanism the paper describes).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.bsm import DEFAULT_FIELD_SPECS


def generate_trajectory(rng: np.random.Generator, length: int, dt: float = 0.1) -> np.ndarray:
    """
    Returns array [length, 5] of columns [x, y, speed, heading, accel], all within
    the ranges declared in src/bsm.DEFAULT_FIELD_SPECS (clipped there for safety).
    """
    x_low, x_high, _ = DEFAULT_FIELD_SPECS["x"]
    y_low, y_high, _ = DEFAULT_FIELD_SPECS["y"]
    v_low, v_high, _ = DEFAULT_FIELD_SPECS["speed"]
    a_low, a_high, _ = DEFAULT_FIELD_SPECS["accel"]

    x = rng.uniform(x_low * 0.2, x_high * 0.2)
    y = rng.uniform(y_low * 0.2, y_high * 0.2)
    v = rng.uniform(5.0, 25.0)
    theta = rng.uniform(-np.pi, np.pi)
    a = rng.normal(0.0, 0.5)
    omega = rng.normal(0.0, 0.15)  # rad/s yaw rate

    out = np.zeros((length, 5), dtype=np.float64)
    for t in range(length):
        out[t] = [x, y, v, theta, a]

        # random-walk process noise on the "hidden" control signals
        a = np.clip(a + rng.normal(0.0, 0.4), a_low, a_high)
        omega = np.clip(omega + rng.normal(0.0, 0.05), -1.2, 1.2)

        v = np.clip(v + a * dt + rng.normal(0.0, 0.05), v_low, v_high)
        theta = theta + omega * dt + rng.normal(0.0, 0.005)
        theta = (theta + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

        x = np.clip(x + v * np.cos(theta) * dt, x_low, x_high)
        y = np.clip(y + v * np.sin(theta) * dt, y_low, y_high)

    return out


class TrajectoryDataset(Dataset):
    """
    Pre-generates `num_trajectories` synthetic trajectories of length
    (history_len + future_len) and serves (history, future) pairs, both
    [T, 5] float32 tensors in physical units, field order = src.bsm.FIELD_ORDER.
    """

    def __init__(self, num_trajectories: int, history_len: int, future_len: int,
                 dt: float = 0.1, seed: int = 42):
        rng = np.random.default_rng(seed)
        total_len = history_len + future_len
        self.history_len = history_len
        self.future_len = future_len
        trajs = np.stack([generate_trajectory(rng, total_len, dt) for _ in range(num_trajectories)])
        self.history = torch.from_numpy(trajs[:, :history_len, :]).float()
        self.future = torch.from_numpy(trajs[:, history_len:, :]).float()

    def __len__(self):
        return self.history.shape[0]

    def __getitem__(self, idx):
        return self.history[idx], self.future[idx]
