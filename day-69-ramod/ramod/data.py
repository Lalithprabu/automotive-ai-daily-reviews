"""Dataset generation (synthetic episodes -> cached cubes + labels) and the torch Dataset / target builder."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import world as W

CUBE_MEAN, CUBE_STD = 0.65, 0.55          # measured on seed-0 frames; used to normalise log-power


def geom() -> dict:
    return dict(s_centers=W.S_CENTERS, v_centers=W.V_CENTERS, vmax=W.V_MAX)


def make_split(n_episodes: int, seed: int, steps: int = 40) -> dict:
    """Render every step of every episode once; samples reuse earlier steps as temporal context."""
    rng = np.random.default_rng(seed)
    cubes = np.zeros((n_episodes, steps, W.R_BINS, W.A_BINS, W.D_BINS), np.float16)
    labs, ego = [], np.zeros((n_episodes, steps, 2), np.float32)
    for e in range(n_episodes):
        sim = W.simulate_episode(rng, steps)
        le = []
        for k in range(steps):
            st = W.frame_state(sim, k)
            cubes[e, k] = (W.render_cube(st, rng) - CUBE_MEAN) / CUBE_STD
            le.append(W.labels(st))
            ego[e, k] = (st["v_ego"], st["w_ego"])
        labs.append(le)
    return dict(cubes=cubes, labels=labs, ego=ego)


def load_or_make(path: str, n_episodes: int, seed: int) -> dict:
    p = Path(path)
    if p.exists():
        return pickle.load(open(p, "rb"))
    d = make_split(n_episodes, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(d, open(p, "wb"), protocol=4)
    return d


def gaussian_radius(h: float, w: float) -> float:
    return max(0.8, 0.25 * min(h, w) + 0.5)


class RadarSeq(Dataset):
    """Item = T consecutive cubes ending at step k + dense CenterNet targets for step k."""

    def __init__(self, split: dict, T: int = 3, n_cls: int = 3, first: int = 0):
        """first: earliest step used as a target frame (evaluate.py sets 2 so every variant sees the same frames)."""
        self.d, self.T, self.n_cls = split, T, n_cls
        E, S = split["cubes"].shape[:2]
        self.idx = [(e, k) for e in range(E) for k in range(max(T - 1, first), S)]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        e, k = self.idx[i]
        cubes = torch.from_numpy(self.d["cubes"][e, k - self.T + 1:k + 1].astype(np.float32))   # (T, R, A, D)
        R, A = W.R_BINS, W.A_BINS
        heat = np.zeros((self.n_cls, R, A), np.float32)
        reg = np.zeros((8, R, A), np.float32)       # log_h, log_w, off_r, off_a, v_meas, k_idx, v_fwd, v_rad
        mask = np.zeros((R, A), np.float32)
        rr, aa = np.meshgrid(np.arange(R), np.arange(A), indexing="ij")
        for o in self.d["labels"][e][k]:
            b = o["box"]
            h, w = b[2] - b[0], b[3] - b[1]
            cr, ca = int(np.clip(o["rb"], 0, R - 1)), int(np.clip(o["ab"], 0, A - 1))
            sig = gaussian_radius(h, w) / 1.5
            g = np.exp(-((rr - cr) ** 2 + (aa - ca) ** 2) / (2 * sig ** 2))
            heat[o["cls"]] = np.maximum(heat[o["cls"]], g)
            reg[:, cr, ca] = (np.log(h), np.log(w), o["rb"] - cr, o["ab"] - ca, o["v_meas"], o["k"] + 2,
                              o["v_fwd"], o["v_rad"])
            mask[cr, ca] = 1
        ego = torch.from_numpy(self.d["ego"][e, k - self.T + 1:k + 1, 0])                     # (T,) v_e
        return cubes, torch.from_numpy(heat), torch.from_numpy(reg), torch.from_numpy(mask), ego, i

    def objects(self, i):
        e, k = self.idx[i]
        return self.d["labels"][e][k]
