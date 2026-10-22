"""Synthetic non-IID pedestrian trajectories for 20 emulated connected vehicles.

The paper emulates vehicles by splitting public datasets (nuScenes / UCY / ETH).
Here, four "regional" behaviour styles stand in for those splits.  A generic
base model is pre-trained on STRAIGHT walking only, so three of the four styles
are genuinely out-of-distribution until fine-tuned.

dt = 0.5 s, 8 observed + 12 future positions (4 s -> 6 s), as in the paper.
"""
from __future__ import annotations

import numpy as np
import torch

STYLES = ["straight", "curving", "stop_go", "crossing"]
OBS, HOR, DT, SCALE = 8, 12, 0.5, 5.0


def _integrate(x0, heading, speed):
    step = np.stack([np.cos(heading), np.sin(heading)], -1) * (speed * DT)[:, None]
    return x0 + np.concatenate([np.zeros((1, 2)), np.cumsum(step[:-1], 0)], 0)


def gen_trajectory(rng: np.random.Generator, style: str) -> np.ndarray:
    """One (20, 2) world-frame trajectory."""
    T = OBS + HOR
    t = np.arange(T)
    x0 = rng.uniform(-20, 20, 2)
    th0 = rng.uniform(-np.pi, np.pi)
    v = rng.normal(1.35, 0.2)
    heading = th0 + np.cumsum(rng.normal(0, 0.02, T))
    speed = v + np.cumsum(rng.normal(0, 0.02, T))
    if style == "curving":      # constant turn rate, visible in the history
        heading = th0 + rng.choice([-1, 1]) * rng.uniform(0.08, 0.2) * t
    elif style == "stop_go":    # keeps walking while observed, then stops in the future
        speed = v * np.clip(1 - 0.25 * np.maximum(t - (OBS), 0), 0, 1)
    elif style == "crossing":   # sharp turn after the window; sign hinted by a tiny drift
        sgn = rng.choice([-1, 1])
        heading = th0 + sgn * 0.03 * np.minimum(t, OBS) + \
            sgn * np.deg2rad(80) * np.clip((t - OBS + 1) / 3, 0, 1)
    speed = np.clip(speed, 0.0, None)
    return _integrate(x0, heading, speed) + rng.normal(0, 0.02, (T, 2))


def canonicalize(obs_xy: np.ndarray):
    """Translate to last observed point, rotate so last motion heads +x.

    obs_xy: (N, OBS, 2) world.  Returns feats (N, OBS, 4), R (N, 2, 2), origin (N, 2).
    feats = [rel_pos / SCALE, vel / SCALE]  (canonical frame)
    """
    origin = obs_xy[:, -1]
    d = obs_xy[:, -1] - obs_xy[:, -3]
    ang = np.arctan2(d[:, 1], d[:, 0])
    c, s = np.cos(ang), np.sin(ang)
    R = np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], 1)  # world -> canonical
    rel = np.einsum("nij,ntj->nti", R, obs_xy - origin[:, None])
    vel = np.concatenate([np.zeros_like(rel[:, :1]), np.diff(rel, axis=1)], 1)
    return np.concatenate([rel, vel], -1).astype(np.float32) / SCALE, R, origin


def to_canonical_future(fut_xy, R, origin):
    return (np.einsum("nij,ntj->nti", R, fut_xy - origin[:, None]) / SCALE).astype(np.float32)


def to_world(pred_canon, R, origin):
    """(N, HOR, 2) canonical (in SCALE units) -> world metres."""
    return np.einsum("nji,ntj->nti", R, pred_canon * SCALE) + origin[:, None]


def make_split(n: int, styles_probs: dict, seed: int):
    rng = np.random.default_rng(seed)
    names = list(styles_probs)
    probs = np.array([styles_probs[k] for k in names], float)
    probs /= probs.sum()
    trajs, sid = [], []
    for _ in range(n):
        k = rng.choice(len(names), p=probs)
        trajs.append(gen_trajectory(rng, names[k]))
        sid.append(STYLES.index(names[k]))
    traj = np.stack(trajs)
    feats, R, origin = canonicalize(traj[:, :OBS])
    return {
        "x": torch.from_numpy(feats),
        "y": torch.from_numpy(to_canonical_future(traj[:, OBS:], R, origin)),
        "R": R, "origin": origin, "world": traj, "style": np.array(sid),
    }


def make_pretrain_set(n: int, seed: int):
    return make_split(n, {"straight": 1.0}, seed)


def make_clients(num_clients: int, n_train: int, n_val: int, own_frac: float, seed: int):
    """Client i's dominant style is STYLES[i % 4] (own_frac of its data); rest uniform."""
    clients, val_parts = [], []
    for i in range(num_clients):
        dom = STYLES[i % len(STYLES)]
        p = {s: (1 - own_frac) / (len(STYLES) - 1) for s in STYLES}
        p[dom] = own_frac
        clients.append(make_split(n_train, p, seed + 100 + i))
        val_parts.append(make_split(n_val, p, seed + 5000 + i))
    return clients, concat(val_parts)


def concat(parts):
    return {
        "x": torch.cat([p["x"] for p in parts]), "y": torch.cat([p["y"] for p in parts]),
        "R": np.concatenate([p["R"] for p in parts]),
        "origin": np.concatenate([p["origin"] for p in parts]),
        "world": np.concatenate([p["world"] for p in parts]),
        "style": np.concatenate([p["style"] for p in parts]),
    }
