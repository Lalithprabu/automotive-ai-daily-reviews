"""Synthetic ego-motion world with *persistent momentum* and *abrupt scene changes*.

Why this world exists (see SOURCING.md): the MomWorld paper's motivating claim is that
(1) motion trends in the observed history should persist into the future plan, and
(2) that persistence must be *suppressed* when the scene changes abruptly.
This generator makes both properties exactly known:

* Every episode has a hidden acceleration trend `a0` and yaw rate `w0` that continue
  into the future -> a single-latent-state rollout that forgets the trend is penalised.
* In ~50% of episodes ("event" episodes) a slower lead vehicle sits ahead on the
  reference arc.  When the gap falls under a speed-dependent trigger distance the expert brakes hard.
  The history acceleration trend is then OUTDATED -> naive momentum extrapolation drives
  the ego into the lead vehicle; the model must read the lead-vehicle cue and reset.

Frames: all inputs/targets are expressed in the ego frame at the last observed pose.
Time step dt = 0.5 s; T_HIST = 8 steps (4 s), H = 12 steps (6 s) horizon.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

DT = 0.5
T_HIST = 8
H = 12
POS_SCALE = 20.0          # metres -> network units
F_IN = 8                  # per-step history features
CAR_RADIUS = 3.0          # collision if two centres are closer than 2*CAR_RADIUS (6 m)
G_MIN = 10.0              # expert keeps at least ~this gap (m); brakes when gap < G_MIN + 1.5*dv^2/(2|BRAKE|)
BRAKE = -4.0              # expert braking decel (m/s^2)


@dataclass
class SceneBatch:
    hist: torch.Tensor        # (B, T_HIST, F_IN)
    future: torch.Tensor      # (B, H, 2)   expert waypoints, ego frame, metres
    lead_future: torch.Tensor # (B, H, 2)   lead-vehicle centre, ego frame, metres
    event: torch.Tensor       # (B,) bool   True when the expert brakes inside the horizon
    event_step: torch.Tensor  # (B,) long   first braking step (H if none)
    raw_hist_xy: torch.Tensor # (B, T_HIST, 2) history positions, ego frame, metres (for plots)


def _arc(kappa: np.ndarray, length: np.ndarray):
    """Point at arc length `length` on a constant-curvature arc through the origin heading +x."""
    small = np.abs(kappa) < 1e-6
    k = np.where(small, 1.0, kappa)
    x = np.where(small, length, np.sin(k * length) / k)
    y = np.where(small, 0.0, (1 - np.cos(k * length)) / k)
    return x, y


def _generate(n: int, rng: np.random.Generator):
    total = T_HIST + H
    v0 = rng.uniform(5.0, 12.0, n)               # speed at last observed step
    a0 = rng.uniform(-1.2, 1.2, n)               # hidden persistent acceleration trend
    w0 = rng.uniform(-0.06, 0.06, n)             # hidden yaw rate (rad/s)
    vl = np.clip(v0 - rng.uniform(1.0, 5.0, n), 1.0, None)   # lead speed (slower than ego)
    event = rng.random(n) < 0.5
    # gap at last observed step.  Event episodes: lead is close enough that the trigger
    # fires inside the horizon.  Free episodes: lead is far -> never triggers.
    gap0 = np.where(event, rng.uniform(17.0, 30.0, n), rng.uniform(90.0, 130.0, n))

    kappa = w0 / v0                              # road/reference curvature (1/m)
    # ---- expert future: speed profile with trend, switching to braking at trigger ----
    v = np.zeros((n, H + 1)); s = np.zeros((n, H + 1)); v[:, 0] = v0
    lead_s = np.zeros((n, H + 1)); lead_s[:, 0] = gap0
    braking = np.zeros(n, bool); ev_step = np.full(n, H)
    for k in range(1, H + 1):
        lead_s[:, k] = lead_s[:, k - 1] + vl * DT
        gap = lead_s[:, k - 1] - s[:, k - 1]
        dv = np.clip(v[:, k - 1] - vl, 0.0, None)
        trig = (gap < G_MIN + 1.5 * dv ** 2 / (2 * -BRAKE)) & ~braking
        ev_step = np.where(trig & (ev_step == H), k - 1, ev_step)
        braking |= trig
        a = np.where(braking, np.where(v[:, k - 1] > vl, BRAKE, 0.0), a0 * np.exp(-0.04 * (k - 1)))
        v[:, k] = np.clip(v[:, k - 1] + a * DT, 0.0, None)
        s[:, k] = s[:, k - 1] + 0.5 * (v[:, k - 1] + v[:, k]) * DT
    triggered = ev_step < H
    # ---- history (no braking ever inside history; same trend) ----
    vh = np.zeros((n, T_HIST)); vh[:, -1] = v0
    for i in range(T_HIST - 2, -1, -1):          # walk backwards: v_{i} = v_{i+1} - a*dt
        vh[:, i] = np.clip(vh[:, i + 1] - a0 * DT, 0.5, None)
    sh = np.zeros((n, T_HIST))
    for i in range(T_HIST - 2, -1, -1):
        sh[:, i] = sh[:, i + 1] - 0.5 * (vh[:, i] + vh[:, i + 1]) * DT
    hist_s = sh                                   # arc positions (last = 0)
    hist_lead_s = gap0[:, None] - (T_HIST - 1 - np.arange(T_HIST))[None] * vl[:, None] * DT
    # ---- to ego-frame coordinates along the reference arc ----
    hx, hy = _arc(kappa[:, None], hist_s)
    fx, fy = _arc(kappa[:, None], s[:, 1:])
    lhx, lhy = _arc(kappa[:, None], hist_lead_s)
    lfx, lfy = _arc(kappa[:, None], lead_s[:, 1:])
    th_h = kappa[:, None] * hist_s               # heading relative to last pose
    # observation noise on what the network sees
    nz = lambda a, sd: a + rng.normal(0, sd, a.shape)
    feats = np.stack([
        nz(hx, 0.05) / POS_SCALE, nz(hy, 0.05) / POS_SCALE,
        np.cos(th_h), np.sin(th_h),
        nz(vh, 0.05) / 10.0,
        nz(lhx, 0.2) / 100.0, nz(lhy, 0.2) / 100.0,
        np.broadcast_to(nz(vl[:, None], 0.1), hx.shape) / 10.0], axis=-1)
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    return SceneBatch(
        hist=f(feats),
        future=f(np.stack([fx, fy], -1)),
        lead_future=f(np.stack([lfx, lfy], -1)),
        event=torch.tensor(triggered),
        event_step=torch.tensor(ev_step, dtype=torch.long),
        raw_hist_xy=f(np.stack([hx, hy], -1)),
    )


def make_dataset(n: int, seed: int) -> SceneBatch:
    return _generate(n, np.random.default_rng(seed))


def index_batch(b: SceneBatch, idx) -> SceneBatch:
    return SceneBatch(*[t[idx] for t in (b.hist, b.future, b.lead_future, b.event, b.event_step, b.raw_hist_xy)])
