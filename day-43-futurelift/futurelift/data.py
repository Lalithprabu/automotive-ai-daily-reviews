"""Synthetic junction-forecasting world used to test the FutureLift mechanism.

Everything is in the target agent's frame at t=0: origin = agent position, +x = heading.
The world is built so a *map-blind* frozen backbone has structured residual errors that a
map-aware, future-graph refiner can in principle fix:

  * a junction at distance D_j ahead with three branches (straight / left arc / right arc)
  * the agent brakes *before* the junction for a turn (speed ~ sqrt(a_lat * R)) -- so an early
    segment's correct position depends on which branch the LATE segment ends up on
    (this is the situation bidirectional temporal edges are meant to exploit)
  * an optional lead vehicle caps the speed on the approach
"""
import math
import numpy as np
import torch
from torch.utils.data import Dataset

DT = 0.5          # seconds per step
T_H = 10          # history steps
T_F = 12          # future steps (6 s)
N_AGENTS = 4      # target + 3 neighbours
N_APP = 16        # approach-lane map points
N_BR = 10         # points per branch
N_MAP = N_APP + 3 * N_BR
SCALE = 30.0      # metres -> network units


def _arc(kappa, length, n, heading=0.0, start=(0.0, 0.0)):
    s = np.linspace(0.0, length, n)
    if abs(kappa) < 1e-9:
        x = start[0] + s * math.cos(heading); y = start[1] + s * math.sin(heading)
        h = np.full(n, heading)
    else:
        h = heading + kappa * s
        x = start[0] + (np.sin(h) - math.sin(heading)) / kappa
        y = start[1] - (np.cos(h) - math.cos(heading)) / kappa
    return np.stack([x, y, h], 1)


def make_scene(rng: np.random.Generator):
    Dj = rng.uniform(25, 45); R = rng.uniform(12, 20)
    branch = int(rng.choice(3, p=[0.5, 0.25, 0.25]))      # 0 straight, 1 left, 2 right
    v0 = rng.uniform(5, 11); a_h = rng.uniform(-1.0, 1.0)
    kappas = [0.0, 1.0 / R, -1.0 / R]
    lead_present = rng.random() < 0.6
    lead_x0 = rng.uniform(12, 30); v_lead = rng.uniform(3, 9)

    # ---- map (agent frame) -------------------------------------------------
    app = np.stack([np.linspace(-20, Dj, N_APP), np.zeros(N_APP), np.zeros(N_APP)], 1)
    brs = [_arc(k, 30.0, N_BR, 0.0, (Dj, 0.0)) for k in kappas]
    mp = np.concatenate([app] + brs, 0)
    map_feat = np.stack([mp[:, 0], mp[:, 1], np.cos(mp[:, 2]), np.sin(mp[:, 2])], 1)

    # ---- ground-truth path & speed profile ---------------------------------
    path_app = np.stack([np.linspace(-20, Dj, 200), np.zeros(200)], 1)
    br_dense = _arc(kappas[branch], 60.0, 400, 0.0, (Dj, 0.0))[:, :2]
    path = np.concatenate([path_app, br_dense[1:]], 0)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s_j = Dj + 20.0                      # arclength of junction
    s = 20.0                             # agent at x=0
    v = v0
    v_turn = math.sqrt(2.5 * R) if branch else 11.0
    gt = []
    for t in range(T_F):
        d = s_j - s
        if branch:
            vt = min(11.0, math.sqrt(v_turn ** 2 + 2 * 2.0 * max(d, 0.0))) if d > 0 else v_turn
        else:
            vt = 11.0
        if lead_present and (d > 0 or branch == 0):
            vt = min(vt, v_lead + 1.0)
        v += float(np.clip(vt - v, -3.0 * DT, 2.0 * DT))
        s += v * DT
        gt.append([np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])])
    gt = np.array(gt) + rng.normal(0, 0.05, (T_F, 2))

    # ---- histories (agent frame) -------------------------------------------
    tau = np.arange(T_H - 1, -1, -1) * DT                 # oldest .. newest(=0)
    hx = -(v0 * tau - 0.5 * a_h * tau ** 2)
    target = np.stack([hx, np.zeros(T_H)], 1)
    hist = np.zeros((N_AGENTS, T_H, 2)); mask = np.zeros(N_AGENTS, bool)
    hist[0] = target; mask[0] = True
    if lead_present:
        hist[1] = np.stack([lead_x0 - v_lead * tau, np.zeros(T_H)], 1); mask[1] = True
    if rng.random() < 0.7:     # oncoming distractor
        vo = rng.uniform(5, 10); xo = rng.uniform(15, 50)
        hist[2] = np.stack([xo + vo * tau, np.full(T_H, 3.5)], 1); mask[2] = True
    if rng.random() < 0.7:     # follower distractor
        vf = rng.uniform(5, 10); xf = rng.uniform(-25, -10)
        hist[3] = np.stack([xf - vf * tau, np.full(T_H, -3.5)], 1); mask[3] = True
    hist = hist + rng.normal(0, 0.03, hist.shape) * mask[:, None, None]
    vel = np.gradient(hist, DT, axis=1)
    hist_feat = np.concatenate([hist, vel], -1) * mask[:, None, None]    # (4,T_H,4)
    meta = np.array([Dj, R, branch, v0, float(lead_present), lead_x0, v_lead])
    return dict(hist=hist_feat.astype(np.float32), mask=mask, map=map_feat.astype(np.float32),
                gt=gt.astype(np.float32), meta=meta.astype(np.float32))


class JunctionDataset(Dataset):
    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        sc = [make_scene(rng) for _ in range(n)]
        self.hist = torch.tensor(np.stack([s["hist"] for s in sc]))
        self.mask = torch.tensor(np.stack([s["mask"] for s in sc]))
        self.map = torch.tensor(np.stack([s["map"] for s in sc]))
        self.gt = torch.tensor(np.stack([s["gt"] for s in sc]))
        self.meta = torch.tensor(np.stack([s["meta"] for s in sc]))

    def __len__(self): return len(self.gt)

    def batch(self, idx):
        return dict(hist=self.hist[idx], mask=self.mask[idx], map=self.map[idx],
                    gt=self.gt[idx], meta=self.meta[idx])

    def loader(self, bs, shuffle, seed=0):
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(self), generator=g) if shuffle else torch.arange(len(self))
        for i in range(0, len(self), bs):
            yield self.batch(order[i:i + bs])
