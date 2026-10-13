"""Synthetic 3-lane driving scenes with latent driver style, hint-flag cut-ins and hard-braking leads.

Why this design: the paper's claim is that (1) one intention-conditioned generator covers diverse maneuvers
in one step and (2) scoring/refining with *future* evidence beats scoring with current features only.
So the generator makes (1) multi-modal (several maneuvers are feasible, a latent driver style picks one) and
(2) future-dependent (cut-ins / hard braking that only partially show in current state via a noisy hint flag).
"""
import numpy as np
import torch
from .scoring import DT, TF, MANEUVERS, factors, pdms

N_AG = 6
HIST = 8
LANES = (-3.5, 0.0, 3.5)
TAUS = np.arange(1, TF + 1) * DT


def _smooth(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def _ego_maneuver(m, v0, lead_v, rng):
    """Nominal ego future (TF,2) for maneuver m."""
    t = TAUS
    y = np.zeros(TF)
    if m == 0:                                    # KEEP
        x = v0 * t
    elif m in (1, 2):                             # lane change
        d0, dur = rng.uniform(0.0, 1.0), rng.uniform(2.5, 3.5)
        y = (3.5 if m == 1 else -3.5) * _smooth((t - d0) / dur)
        x = v0 * t
    elif m == 3:                                  # BRAKE to (near) stop
        a = rng.uniform(2.0, 3.6)
        ts = np.minimum(t, v0 / a)
        x = v0 * ts - 0.5 * a * ts ** 2
        x = x + 0.0 * t
    else:                                         # FOLLOW: relax to a slower speed
        vt = np.clip(lead_v if lead_v is not None else 0.6 * v0, 0.5 * v0 * 0.5, v0)
        tau = rng.uniform(1.0, 2.0)
        x = vt * t + (v0 - vt) * tau * (1 - np.exp(-t / tau))
    return np.stack([x, y], -1)


def _agent(kind, x0, lane_y, v, behaviour, rng):
    """Agent future (TF,2), velocity at t=0 (vx,vy), hint flag."""
    t = TAUS
    x, y = x0 + v * t, np.full(TF, lane_y)
    vx, vy, true_event = v, 0.0, 0
    if behaviour == "cutin":                       # side vehicle moves to ego lane (y=0)
        t0, dur = rng.uniform(0.5, 2.5), rng.uniform(2.0, 3.0)
        y = lane_y * (1 - _smooth((t - t0) / dur))
        true_event = 1
        vy = -np.sign(lane_y) * 0.15 * rng.uniform(0, 1)     # tiny lateral drift visible at t=0
    elif behaviour == "hardbrake":                 # lead decelerates
        t0, a = rng.uniform(0.0, 2.0), rng.uniform(2.5, 4.0)
        te = np.maximum(t - t0, 0.0)
        ts = np.minimum(te, v / a)
        # position: travels v*t until t0, then decelerates at a until stopped
        xs = v * np.minimum(t, t0) + np.where(t > t0, v * ts - 0.5 * a * ts ** 2, 0.0)
        x = x0 + xs
        true_event = 1
    return np.stack([x, y], -1), vx, vy, true_event


def make_scene(rng):
    v0 = rng.uniform(6.0, 12.0)
    hist = np.stack([-v0 * DT * np.arange(HIST - 1, -1, -1), np.zeros(HIST), np.full(HIST, v0), np.zeros(HIST)], -1)
    feats = np.zeros((N_AG, 6), np.float32)
    fut = np.zeros((N_AG, TF, 2), np.float32)
    lead_v = None
    slots = []
    # agent 0: lead in ego lane
    if rng.random() < 0.85:
        v = rng.uniform(1.0, 4.0) if rng.random() < 0.35 else rng.uniform(5.0, v0 + 2.0)
        beh = "hardbrake" if rng.random() < 0.25 else "cv"
        slots.append((0, rng.uniform(10, 45), 0.0, v, beh))
        lead_v = v
    for slot, lane in ((1, 3.5), (2, -3.5)):
        if rng.random() < 0.85:
            beh = "cutin" if rng.random() < 0.3 else "cv"
            slots.append((slot, rng.uniform(-12, 35), lane, rng.uniform(4.0, 13.0), beh))
    for slot in (3, 4, 5):
        if rng.random() < 0.5:
            slots.append((slot, rng.uniform(30, 70), float(rng.choice(LANES)), rng.uniform(4.0, 13.0), "cv"))
    for slot, x0, ly, v, beh in slots:
        traj, vx, vy, ev = _agent(slot, x0, ly, v, beh, rng)
        # hint flag (turn signal / brake light): 75% recall on true events, 12% false-positive rate
        hint = float(rng.random() < (0.75 if ev else 0.12))
        feats[slot] = [x0, ly, vx, vy, hint, 1.0]
        fut[slot] = traj
    valid = feats[:, 5].copy()
    # feasible maneuvers & latent-style choice
    ft = torch.from_numpy(fut)[None]
    vt = torch.from_numpy(valid)[None]
    cands = [_ego_maneuver(m, v0, lead_v, rng) for m in range(len(MANEUVERS))]
    sc = []
    for c in cands:
        f = factors(torch.from_numpy(c)[None].float(), ft, vt)
        sc.append(float(pdms(f)))
    sc = np.array(sc)
    style = rng.normal(0, 1.2, size=len(MANEUVERS))
    style[0] += 0.8                                          # mild preference to stay in lane
    logits = 5.0 * sc + style
    logits[sc <= 0] = -1e9
    if (sc <= 0).all():
        m = 3
    else:
        p = np.exp(logits - logits.max()); p /= p.sum()
        m = int(rng.choice(len(MANEUVERS), p=p))
    gt = cands[m] + rng.normal(0, 0.03, cands[m].shape)
    return dict(hist=hist.astype(np.float32), agents=feats, fut_agents=fut, gt=gt.astype(np.float32),
                maneuver=m, feasible=(sc > 0).astype(np.float32))


def make_dataset(n, seed):
    rng = np.random.default_rng(seed)
    sc = [make_scene(rng) for _ in range(n)]
    out = {k: torch.from_numpy(np.stack([s[k] for s in sc])) for k in ("hist", "agents", "fut_agents", "gt", "feasible")}
    out["maneuver"] = torch.tensor([s["maneuver"] for s in sc])
    return out
