"""Synthetic multi-lane highway with hidden, multimodal driver intentions (agent-centric frame).

Every sample is ONE surrounding ("target") vehicle seen from its own frame at the present time:
origin = its centre, +x = its heading.  The future is drawn from a hidden intention:

    0 keep lane     1 brake (hidden onset, no brake light)    2 accelerate
    3 lane change LEFT (+y)   4 lane change RIGHT (-y)

Only weak cues leak into the 1 s history (a small lateral drift for some lane changes, a closing lead
gap that makes braking / lane changes more likely), so the realised future is genuinely uncertain:
exactly the setting where a single predicted footprint under-approximates the true occupancy and a
physics reachable set over-approximates it.

Everything is numpy; tensors are built in data.py.
"""
from __future__ import annotations

import numpy as np

DT_H, HIST = 0.1, 10                       # 1.0 s of history at 10 Hz
DT_F, H = 0.25, 12                         # 3.0 s future at 4 Hz
T_FUT = DT_F * np.arange(1, H + 1)         # (H,)
T_HIST = -DT_H * np.arange(HIST - 1, -1, -1)
LANE_W = 3.5
VEH_L, VEH_W = 4.6, 1.9                    # target footprint
EGO_L, EGO_W = 4.6, 1.9
N_INTENT = 5
INTENT_NAMES = ["keep", "brake", "accel", "lane-change L", "lane-change R"]
CTX_DIM = 6                                # [left lane?, right lane?, lead gap/50, lead dv/10, v0/30, drift cue]


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def rollout(rng, v0, intent, t):
    """Longitudinal + lateral position of the target at times t (any shape) for one intention."""
    a_noise = rng.normal(0, 0.25)
    if intent == 1:                                   # hidden hard brake
        tb, a = rng.uniform(0.0, 1.2), rng.uniform(-6.5, -3.0)
    elif intent == 2:
        tb, a = rng.uniform(0.0, 0.8), rng.uniform(1.0, 2.5)
    else:
        tb, a = 0.0, a_noise
    tt = np.maximum(t - tb, 0.0)
    tstop = -v0 / a if a < 0 else np.inf
    tt_c = np.minimum(tt, tstop)
    x = v0 * np.minimum(t, tb) + v0 * tt_c + 0.5 * a * tt_c ** 2
    y = np.zeros_like(t)
    if intent in (3, 4):
        side = 1.0 if intent == 3 else -1.0
        t0, dur = rng.uniform(-0.6, 0.9), rng.uniform(2.4, 4.2)
        y = side * LANE_W * smoothstep((t - t0) / dur)
    return x, y, dict(t_onset=tb if intent in (1, 2) else 0.0)


def sample_scene(rng):
    v0 = rng.uniform(8.0, 26.0)
    left, right = rng.random() < 0.75, rng.random() < 0.75
    gap = rng.uniform(8.0, 80.0)
    dv = rng.uniform(-6.0, 2.0)                       # lead speed - own speed
    closing = (gap < 30) and (dv < -1.5)
    # intention prior depends (weakly) on the visible context
    p = np.array([0.40, 0.14, 0.14, 0.16, 0.16])
    if closing:
        p = np.array([0.15, 0.35, 0.02, 0.24, 0.24])
    if not left:
        p[3] = 0.0
    if not right:
        p[4] = 0.0
    p /= p.sum()
    intent = int(rng.choice(N_INTENT, p=p))
    # history: lane changes that started before t=0 leave a lateral drift (a weak, noisy cue)
    t_all = np.concatenate([T_HIST, T_FUT])
    x, y, info = rollout(rng, v0, intent, t_all)
    # re-centre so that the present position is the origin, add noise to the history
    x0, y0 = np.interp(0.0, t_all, x), np.interp(0.0, t_all, y)
    x, y = x - x0, y - y0
    hx, hy = x[:HIST] + rng.normal(0, 0.05, HIST), y[:HIST] + rng.normal(0, 0.04, HIST)
    fx, fy = x[HIST:], y[HIST:]
    # rotate into the present-heading frame (heading at t=0 from the noiseless motion)
    eps = 0.05
    vy0 = (np.interp(eps, t_all, y) - np.interp(-eps, t_all, y)) / (2 * eps)
    vx0 = (np.interp(eps, t_all, x) - np.interp(-eps, t_all, x)) / (2 * eps)
    psi0 = np.arctan2(vy0, max(vx0, 1e-3))
    R = np.array([[np.cos(-psi0), -np.sin(-psi0)], [np.sin(-psi0), np.cos(-psi0)]])
    hist = (R @ np.stack([hx, hy])).T                                   # (HIST, 2)
    fut = (R @ np.stack([fx, fy])).T                                    # (H, 2)
    # heading of the future footprint from finite differences of the (rotated) path
    path = np.concatenate([np.zeros((1, 2)), fut], 0)
    d = np.diff(path, axis=0)
    spd = np.linalg.norm(d, axis=-1)
    head = np.arctan2(d[:, 1], d[:, 0])
    # when (almost) stopped keep the last valid heading
    for k in range(H):
        if spd[k] < 0.05:
            head[k] = head[k - 1] if k > 0 else 0.0
    hv = np.diff(hist, axis=0, prepend=hist[:1]) / DT_H
    hv[0] = hv[1]
    lane_y = -y0                                                        # lateral offset of the own-lane centre
    drift = float(np.clip(hy[-1] - hy[0], -1, 1))
    ctx = np.array([left, right, gap / 50.0, dv / 10.0, v0 / 30.0, drift], np.float32)
    return dict(hist=np.concatenate([hist, hv], -1).astype(np.float32),   # (HIST, 4) x, y, vx, vy
                ctx=ctx, fut=fut.astype(np.float32), head=head.astype(np.float32),
                intent=intent, v0=np.float32(v0), lane_y=np.float32(lane_y),
                left=bool(left), right=bool(right), psi0=np.float32(psi0))


def footprint_corners(c, psi, L=VEH_L, Wd=VEH_W):
    """Oriented rectangle corners. c (..., 2), psi (...) -> (..., 4, 2)."""
    ca, sa = np.cos(psi), np.sin(psi)
    loc = np.array([[L / 2, Wd / 2], [L / 2, -Wd / 2], [-L / 2, -Wd / 2], [-L / 2, Wd / 2]])
    rx = ca[..., None] * loc[:, 0] - sa[..., None] * loc[:, 1]
    ry = sa[..., None] * loc[:, 0] + ca[..., None] * loc[:, 1]
    return np.stack([rx, ry], -1) + c[..., None, :]
