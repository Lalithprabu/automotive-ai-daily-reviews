"""Dataset generation / loading for the synthetic LOWP toy (SYNTHETIC — not nuScenes/NAVSIM)."""
from __future__ import annotations

import numpy as np
import torch

from .world import T_FUT, DT, SIM_DT, make_rig, render, sample_at, simulate_episode

MAX_OBS = 3


def rig_tensors(rig):
    K = torch.tensor(np.stack([c.K for c in rig]), dtype=torch.float32)
    R = torch.tensor(np.stack([c.R for c in rig]), dtype=torch.float32)
    t = torch.tensor(np.stack([c.t for c in rig]), dtype=torch.float32)
    return K, R, t


def generate(n: int, seed: int, H: int = 40, W: int = 80, stride: int = 7, steps: int = 400) -> dict:
    """Sample n frames from random episodes (every `stride` sim steps)."""
    rng = np.random.default_rng(seed)
    rig = make_rig(H, W)
    horizon = int(T_FUT * DT / SIM_DT)
    imgs, fut, cmd, status, obs = [], [], [], [], []
    while len(imgs) < n:
        sim = simulate_episode(rng, steps)
        for k in range(int(rng.integers(0, stride)), steps - horizon - 1, stride):
            s = sample_at(sim, k)
            imgs.append(render(sim, k, rig, H, W, rng))
            fut.append(s["fut"]); cmd.append(s["cmd"]); status.append(s["status"])
            o = np.full((T_FUT, MAX_OBS, 3), np.nan, np.float32)
            o[:, : s["obs_fut"].shape[1]] = s["obs_fut"][:, :MAX_OBS]
            obs.append(o)
            if len(imgs) >= n:
                break
    return dict(imgs=np.stack(imgs), fut=np.stack(fut), cmd=np.array(cmd, np.int64),
                status=np.stack(status), obs_fut=np.stack(obs))


class Frames(torch.utils.data.Dataset):
    def __init__(self, d: dict):
        self.d = d

    def __len__(self):
        return len(self.d["fut"])

    def __getitem__(self, i):
        img = torch.from_numpy(self.d["imgs"][i]).permute(0, 3, 1, 2).float() / 255.0      # (N, 3, H, W)
        return img, torch.from_numpy(self.d["status"][i]), int(self.d["cmd"][i]), torch.from_numpy(self.d["fut"][i])
