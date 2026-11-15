"""Dataset generation, caching, batching, and the interaction (ego-merge) scenario attached to each sample."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import world as W


def ego_scenario(rng, sc):
    """An ego vehicle in a lane adjacent to the target wants to MERGE into the target's lane within 3 s."""
    sides = [s for s, ok in ((1.0, sc["left"]), (-1.0, sc["right"])) if ok] or [1.0]
    side = float(rng.choice(sides))
    return np.array([side, rng.uniform(-14.0, 12.0), max(3.0, sc["v0"] + rng.uniform(-2.5, 2.5))], np.float32)


def generate(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("hist", "ctx", "fut", "head", "intent", "v0", "lane_y", "psi0", "ego")}
    for _ in range(n):
        sc = W.sample_scene(rng)
        sc["ego"] = ego_scenario(rng, sc)
        for k in out:
            out[k].append(sc[k])
    return {k: torch.as_tensor(np.stack(v)) for k, v in out.items()}


def get(n: int, seed: int, cache: Path) -> dict:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return torch.load(cache)
    d = generate(n, seed)
    torch.save(d, cache)
    return d


def batch(d: dict, idx) -> dict:
    return {k: v[idx] for k, v in d.items()}


def physics_reach_box(d: dict, a_min=-8.0, a_max=3.0, a_lat=3.0, margin=0.3):
    """Conservative forward reachable set (axis-aligned box per step), the classic over-approximation.

    x in [x(a_min), x(a_max)] from the current speed, |y| <= 1/2 a_lat t^2 + current lateral velocity * t,
    inflated by the vehicle footprint.  Returns centres (B, 1, H, 2) and half extents (B, 1, H, 2).
    """
    v = d["hist"][:, -1, 2].clamp(min=0.0)                                   # (B,)
    vy = d["hist"][:, -1, 3]
    t = torch.tensor(W.T_FUT, dtype=torch.float32)                          # (H,)
    tstop = (v / -a_min)[:, None]
    tt = torch.minimum(t[None], tstop)
    x_lo = v[:, None] * tt + 0.5 * a_min * tt ** 2
    x_hi = v[:, None] * t + 0.5 * a_max * t ** 2
    y_c = vy[:, None] * t
    y_r = 0.5 * a_lat * t ** 2
    half_diag = 0.5 * float(np.hypot(W.VEH_L, W.VEH_W))
    c = torch.stack([(x_lo + x_hi) / 2, y_c.expand_as(x_lo)], -1)
    ex = (x_hi - x_lo) / 2 + half_diag + margin
    ey = (y_r + half_diag + margin).expand_as(ex)
    return c[:, None], torch.stack([ex, ey], -1)[:, None]
