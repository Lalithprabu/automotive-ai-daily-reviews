"""Synthetic driving scenes: junction (straight/left/right) + curved-road scenes.

Everything is in the agent frame at t=0: agent at origin, heading +x.
History = 20 steps (2 s @10 Hz), future = 30 steps (3 s). Map = candidate centerlines the agent may follow.
The agent's speed profile depends on the chosen maneuver (it slows before/in a turn), so the future
is multi-modal at junctions and fully map-determined on plain roads.
"""
import math
import numpy as np
import torch
from torch.utils.data import Dataset

DT, H_STEPS, F_STEPS = 0.1, 20, 30
N_LANES, N_LANE_PTS = 3, 28
KINDS = ["junction_straight", "junction_left", "junction_right", "road"]


def _path(theta_fn, s_grid):
    """Integrate heading -> xy along arclength, anchored so that s=0 is the origin."""
    th = theta_fn(s_grid)
    ds = np.diff(s_grid, prepend=s_grid[0])
    x = np.cumsum(np.cos(th) * ds); y = np.cumsum(np.sin(th) * ds)
    i0 = int(np.argmin(np.abs(s_grid)))
    return x - x[i0], y - y[i0], th


def _theta_junction(d, R, sign):
    arc = R * math.pi / 2
    def f(s):
        u = np.clip((s - d) / arc, 0.0, 1.0)  # 0..1 progress through the 90-degree arc
        return sign * (math.pi / 2) * u
    return f


def make_scene(rng: np.random.Generator, kind=None):
    s_grid = np.arange(-40.0, 120.0, 0.1)
    if kind is None:
        kind = rng.choice(KINDS, p=[0.28, 0.2, 0.2, 0.32])
    v0 = rng.uniform(4.0, 12.0)
    d = rng.uniform(12.0, 28.0)          # distance to the junction
    R = rng.uniform(6.0, 12.0)
    lanes_xy = []
    if kind == "road":
        k0, k1 = rng.uniform(-0.04, 0.04), rng.uniform(-0.002, 0.002)
        f = lambda s: k0 * s + 0.5 * k1 * s ** 2
        paths = [_path(f, s_grid)]
        chosen, turn_start, turn_end = 0, 1e9, 1e9
    else:
        # candidate centerlines at a junction: straight, left, right (all anchored at the origin)
        fs = [lambda s: np.zeros_like(s), _theta_junction(d, R, +1.0), _theta_junction(d, R, -1.0)]
        paths = [_path(f, s_grid) for f in fs]
        chosen = {"junction_straight": 0, "junction_left": 1, "junction_right": 2}[kind]
        turn_start, turn_end = d - 10.0, d + R * math.pi / 2
    for (x, y, _) in paths:
        s_pts = np.linspace(-8.0, 46.0, N_LANE_PTS)
        lanes_xy.append(np.stack([np.interp(s_pts, s_grid, x), np.interp(s_pts, s_grid, y)], -1))
    px, py, pth = paths[chosen]
    # speed profile as a function of arclength: slow down for turns
    slow = 0.0 if chosen == 0 else rng.uniform(0.35, 0.5)
    def v_of_s(s):
        if s < turn_start: return v0
        if s < turn_end:   return v0 * (1.0 - slow)
        return v0 * (1.0 - slow * max(0.0, 1.0 - (s - turn_end) / 8.0))
    s = [0.0]
    for _ in range(F_STEPS):
        s.append(s[-1] + max(0.5, v_of_s(s[-1]) + rng.normal(0, 0.15)) * DT)
    sh = [0.0]
    for _ in range(H_STEPS):
        sh.append(sh[-1] - max(0.5, v0 + rng.normal(0, 0.15)) * DT)
    s_all = np.array(sh[::-1][:-1] + s)  # 20 history pts + 31 future pts, index 20 is t=0
    pts = np.stack([np.interp(s_all, s_grid, px), np.interp(s_all, s_grid, py)], -1)
    hd = np.interp(s_all, s_grid, pth)
    lane_mask = np.zeros(N_LANES, np.float32); lane_pad = np.zeros((N_LANES, N_LANE_PTS, 2), np.float32)
    for i, l in enumerate(lanes_xy):
        lane_pad[i] = l; lane_mask[i] = 1.0
    return dict(hist_pts=pts[:H_STEPS + 1], hist_hd=hd[:H_STEPS + 1],
                fut_pts=pts[H_STEPS:], fut_hd=hd[H_STEPS:],
                lanes=lane_pad, lane_mask=lane_mask, kind=KINDS.index(kind), chosen_lane=chosen), kind


class SyntheticDrivingDataset(Dataset):
    def __init__(self, n, seed=0):
        rng = np.random.default_rng(seed)
        items = [make_scene(rng)[0] for _ in range(n)]
        t = lambda k, dt=torch.float32: torch.tensor(np.stack([it[k] for it in items]), dtype=dt)
        self.d = dict(hist_pts=t("hist_pts"), hist_hd=t("hist_hd"), fut_pts=t("fut_pts"), fut_hd=t("fut_hd"),
                      lanes=t("lanes"), lane_mask=t("lane_mask"),
                      kind=torch.tensor([it["kind"] for it in items]),
                      chosen_lane=torch.tensor([it["chosen_lane"] for it in items]))

    def __len__(self): return self.d["kind"].shape[0]
    def __getitem__(self, i): return {k: v[i] for k, v in self.d.items()}
