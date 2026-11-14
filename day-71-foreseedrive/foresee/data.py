"""Dataset generation + caching.  Every scene is stored as fixed-size arrays (padded to N_ACT actors)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import tokens as TK
from . import world as W

KEYS = ("v_ego", "hist", "fut", "typ", "valid", "radius", "event", "expert", "rank", "tE", "tC", "tR")


def build(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {k: [] for k in KEYS}
    for _ in range(n):
        sc = W.sample_scene(rng)
        rank = TK.rank_actors(sc["hist"], sc["valid"])
        E, C, R = TK.scene_tokens(sc, rank)
        for k in KEYS[:8]:
            out[k].append(sc[k])
        out["rank"].append(rank); out["tE"].append(E); out["tC"].append(C); out["tR"].append(R)
    return {k: np.asarray(v) for k, v in out.items()}


def load(split: str, n: int, seed: int, cache="data") -> dict:
    p = Path(cache) / f"{split}_{n}_{seed}.npz"
    if p.exists():
        d = np.load(p)
        return {k: d[k] for k in d.files}
    d = build(n, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **d)
    return d


def to_torch(d: dict) -> dict:
    t = {}
    for k, v in d.items():
        if v.dtype == bool:
            t[k] = torch.from_numpy(v)
        elif np.issubdtype(v.dtype, np.integer):
            t[k] = torch.from_numpy(v.astype(np.int64))
        else:
            t[k] = torch.from_numpy(v.astype(np.float32))
    return t


def batch(t: dict, idx) -> dict:
    return {k: v[idx] for k, v in t.items()}


def scene_dict(d: dict, i: int) -> dict:
    """Back to the per-scene dict used by world.pdm_score."""
    return {k: (d[k][i].numpy() if torch.is_tensor(d[k]) else d[k][i]) for k in KEYS[:8]} | {
        "v_ego": float(d["v_ego"][i])}
