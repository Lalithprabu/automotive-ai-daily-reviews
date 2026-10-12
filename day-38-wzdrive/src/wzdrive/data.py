"""Synthetic work-zone driving data (human-in-the-loop style: ego kinematics + driver
controls + driver gaze + scene objects). This is NOT WZDrive-Data; it is a small
generator built so that the paper's *claim* (driver gaze/control + scene help predict
work-zone ego trajectories) is falsifiable here.

Ego frame at t=0: x forward, y left, origin = ego position at t=0.
Classes: 0 KEEP (closure in adjacent lane), 1 MERGE_EARLY, 2 MERGE_LATE, 3 YIELD (worker steps in).
MERGE_EARLY vs MERGE_LATE share nearly identical scenes + kinematics; only the driver's
gaze (looks at the open lane ~1 s before steering) and steering/throttle tell them apart.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_NAMES = ["KEEP", "MERGE_EARLY", "MERGE_LATE", "YIELD"]
N_TOK, TOK_DIM = 12, 8      # scene tokens: [is_cone,is_worker,x,y,vx,vy,valid,pad]
LANE_W = 3.5


def _smooth(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def _gen_one(rng, hist=8, fut=12, dt=0.25):
    cls = rng.choice(4, p=[0.2, 0.3, 0.3, 0.2])
    side = rng.choice([-1.0, 1.0])               # which adjacent lane is open (+1 = left)
    v0 = rng.uniform(7.0, 13.0)
    Dc = rng.uniform(26.0, 44.0)                 # distance to closure start
    t = (np.arange(-hist, fut + 1) * dt)         # includes t=-2.0 ... 3.0 -> 21 pts (first used for velocity)
    T = len(t)
    x = v0 * t
    y = np.zeros(T)
    gaze_target = np.zeros(T)                    # target yaw in rad (0 = ahead)
    on_target = np.zeros(T)
    w_info = None
    # ------ scene -----
    toks = []
    def cone(px, py): toks.append([1, 0, px, py, 0, 0, 1, 0])
    if cls == 0:                                  # closure in the *other* lane, ego lane is clear
        for k in range(4): cone(Dc + 4 * k, -side * LANE_W)
        for k in range(3): cone(Dc - 6 + 3 * k, -side * (LANE_W - 1.0 * (2 - k)))
    else:
        for k in range(4): cone(Dc + 4 * k, 0.0)
        for k in range(3): cone(Dc - 9 + 3 * k, 0.0 - side * 0.0 + (-side) * 0.4 * (k - 1))
        for k in range(2): cone(Dc + 2 * k, -side * LANE_W)     # blocked far lane markers (shows which lane is open)
    # worker (always present); only enters the ego lane in YIELD
    xw = Dc * rng.uniform(0.45, 0.7)
    if cls == 3:
        yw = rng.choice([-1, 1]) * rng.uniform(3.0, 4.0); vyw = -np.sign(yw) * rng.uniform(1.0, 1.6)
    else:
        yw = rng.choice([-1, 1]) * rng.uniform(4.0, 5.5); vyw = -np.sign(yw) * rng.uniform(0.0, 0.5) * rng.choice([0, 1])
    toks.append([0, 1, xw, yw, 0.0, vyw, 1, 0])
    for _ in range(2):                            # roadside distractors
        toks.append([0, 1, rng.uniform(10, 50), rng.choice([-1, 1]) * rng.uniform(6, 8), 0, 0, 1, 0])
    # ------ behaviour -----
    gaze_lead = rng.uniform(0.7, 1.2) if rng.random() > 0.15 else rng.uniform(0.0, 0.4)  # 15% "late glancers"
    if cls == 1 or cls == 2:
        ts = rng.uniform(0.1, 0.6) if cls == 1 else rng.uniform(1.5, 2.1)
        Tm = rng.uniform(2.0, 2.6)
        y = side * LANE_W * _smooth((t - ts) / Tm)
        if cls == 2:                              # late mergers shed a little speed first
            a = rng.uniform(-1.2, -0.3); x = v0 * t + 0.5 * a * np.maximum(t, 0) ** 2
        tg = ts - gaze_lead
        look = side * np.arctan2(LANE_W, 14.0)
        gaze_target = look * _smooth((t - tg) / 0.3)
        on_target = (np.abs(gaze_target - look) < 0.06).astype(float)
    elif cls == 0:
        if rng.random() < 0.25:                   # distractor glance toward the closed lane
            tg = rng.uniform(-1.2, 1.0)
            gaze_target = (-side) * 0.2 * (_smooth((t - tg) / 0.3) - _smooth((t - tg - 0.6) / 0.3))
    else:                                         # YIELD: decelerate to stop before the worker's crossing line
        tb = rng.uniform(0.05, 0.5)
        dstop = max(xw - 6.0, 6.0)
        v = np.where(t < tb, v0, 0.0)
        a_d = v0 ** 2 / (2.0 * max(dstop - v0 * tb, 4.0))
        s = np.where(t < tb, v0 * t, v0 * tb + v0 * (t - tb) - 0.5 * a_d * (t - tb) ** 2)
        tstop = tb + v0 / a_d
        s = np.where(t > tstop, v0 * tb + v0 * (tstop - tb) - 0.5 * a_d * (tstop - tb) ** 2, s)
        x = s
        tg = tb - rng.uniform(0.6, 1.2) if rng.random() > 0.2 else tb - rng.uniform(0.0, 0.3)
        wyaw = np.arctan2(yw, xw)
        gaze_target = wyaw * _smooth((t - tg) / 0.3)
        on_target = (np.abs(gaze_target - wyaw) < 0.06).astype(float)
    # ------ sensors -----
    gaze_yaw = gaze_target + rng.normal(0, 0.025, T)
    gaze_pitch = rng.normal(-0.04, 0.02, T)
    ax = np.gradient(np.gradient(x, dt), dt)
    ay = np.gradient(np.gradient(y, dt), dt)
    steer = np.clip(ay / 1.2, -1, 1) + rng.normal(0, 0.03, T)
    throttle = np.clip(ax / 3.0, 0, 1) + np.abs(rng.normal(0, 0.02, T))
    brake = np.clip(-ax / 4.0, 0, 1) + np.abs(rng.normal(0, 0.02, T))
    # ------ ego frame ------
    i0 = hist                                      # index of t=0
    p = np.stack([x, y], 1) + rng.normal(0, 0.03, (T, 2))
    origin = p[i0].copy()
    pos = p - origin
    vel = np.gradient(pos, dt, axis=0)
    hs = slice(1, hist + 1)                        # t=-1.75 .. 0.0  (8 steps)
    hist_kin = np.concatenate([pos[hs], vel[hs]], 1)            # (8,4)
    ctrl = np.stack([steer, throttle, brake], 1)[hs]            # (8,3)
    gaze = np.stack([gaze_yaw, gaze_pitch, on_target], 1)[hs]   # (8,3)
    fut_xy = pos[i0 + 1:]                                       # (12,2)
    sc = np.zeros((N_TOK, TOK_DIM), np.float32)
    for i, tk in enumerate(toks[:N_TOK]):
        tk = list(tk); tk[2] -= origin[0]; tk[3] -= origin[1]; sc[i] = tk
    return hist_kin, ctrl, gaze, sc, fut_xy, cls


def generate_dataset(n, seed, hist=8, fut=12, dt=0.25):
    rng = np.random.default_rng(seed)
    out = [_gen_one(rng, hist, fut, dt) for _ in range(n)]
    f = lambda i: torch.tensor(np.stack([o[i] for o in out]), dtype=torch.float32)
    return dict(kin=f(0), ctrl=f(1), gaze=f(2), scene=f(3), fut=f(4),
                cls=torch.tensor([o[5] for o in out]))


class WorkZoneDataset(Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d["cls"])
    def __getitem__(self, i): return {k: v[i] for k, v in self.d.items()}
