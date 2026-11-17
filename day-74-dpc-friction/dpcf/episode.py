"""Data collection (excited feedback driving) and closed-loop episodes."""
from __future__ import annotations

import numpy as np
import torch

from . import vehicle as V


def kappa_ahead(tr: V.Track, i0: int, v_ref: np.ndarray, N: int) -> np.ndarray:
    """Reference curvature at the positions the car will reach over the next N steps (at v_ref)."""
    out, s = [], tr.s[i0]
    for k in range(N):
        s = s + v_ref[tr.idx(s)] * V.DT
        out.append(tr.kappa[tr.idx(s)])
    return np.array(out)


def collect(tr: V.Track, T: int, seed: int, amp=0.06, v_scale=1.0, gains=(1.2, 1.5)):
    """Drive T control steps with an excited feedback law; return u (T, 2) = (delta, kappa) and
    y (T, 2) = (e_y, e_psi) measured AFTER applying u_t (shifted pairing: u_t -> y_{t+1})."""
    rng = np.random.default_rng(seed)
    car = V.Car(tr, v0=1.5, seed=seed)
    v_ref = tr.speed_profile() * v_scale
    y_meas, _ = car.measure()
    U, Y = [], []
    exc, hold = 0.0, 0
    L = V.P["lf"] + V.P["lr"]
    for t in range(T):
        if hold <= 0:
            exc, hold = rng.uniform(-amp, amp), int(rng.integers(2, 7))
        hold -= 1
        kap = tr.kappa[car.i]
        delta = L * kap - gains[0] * y_meas[0] - gains[1] * y_meas[1] + exc + rng.normal(0, 0.01)
        delta = float(np.clip(delta, -V.P["delta_max"], V.P["delta_max"]))
        car.step(delta, v_ref[car.i])
        y_meas, _ = car.measure()
        U.append([delta, kap]); Y.append(y_meas.copy())
    return torch.tensor(np.array(U), dtype=torch.float64), torch.tensor(np.array(Y), dtype=torch.float64)


def run(tr: V.Track, ctrl, kind: str, seed: int, laps: float = 2.0, log: bool = False, Ti: int = 5, N: int = 15,
        warm_gains=(1.2, 1.5)):
    """Closed loop for `laps` laps. kind = 'dpc' or 'model'. Returns metrics (+ per-step log)."""
    rng = np.random.default_rng(10_000 + seed)
    car = V.Car(tr, v0=1.5, ey0=rng.uniform(-0.05, 0.05), seed=seed)
    v_ref = tr.speed_profile()
    y_meas, true = car.measure()
    u_hist, y_hist = [], []
    delta = 0.0
    L = V.P["lf"] + V.P["lr"]
    errs, frames = [], []
    off = False
    t = 0
    while car.s_tot < laps * tr.L and t < 4000:
        kap = tr.kappa[car.i]
        kf = kappa_ahead(tr, car.i, v_ref, N)
        mu = car.mu_here()
        if len(u_hist) < Ti:                                        # warm-up: feedback to fill u_ini / y_ini
            d = L * kap - warm_gains[0] * y_meas[0] - warm_gains[1] * y_meas[1]
            info = {}
        elif kind == "dpc":
            d = ctrl.act(u_hist, y_hist, np.concatenate([[kap], kf[:-1]]), delta, mu)
            info = ctrl.last
        else:
            d = ctrl.act(car.state, true[0], true[1], kap, kf, delta, mu)
            info = ctrl.last
        delta = float(np.clip(d, -V.P["delta_max"], V.P["delta_max"]))
        if log:
            frames.append(dict(t=t * V.DT, state=car.state.copy(), i=car.i, s=car.s_tot, ey=true[0], epsi=true[1],
                               delta=delta, mu=mu, y_pred=info.get("y_pred"), key=info.get("key"),
                               kf=kf.copy(), vref=v_ref[car.i], plan=info.get("plan"),
                               u_ini=np.array(u_hist[-Ti:]), y_ini=np.array(y_hist[-Ti:]),
                               k_in=np.concatenate([[kap], kf[:-1]])))
        car.step(delta, v_ref[car.i])
        y_meas, true = car.measure()
        u_hist.append([delta, kap]); y_hist.append(y_meas.copy())
        u_hist, y_hist = u_hist[-Ti:], y_hist[-Ti:]
        errs.append(true[0])
        if abs(true[0]) > V.HALF_W:
            off = True
            break
        t += 1
    e = np.abs(np.array(errs))
    skip = int(1.0 / V.DT)                                           # ignore the first second (start transient)
    e = e[skip:] if len(e) > skip else e
    out = dict(rms_ey=float(np.sqrt((e ** 2).mean())), max_ey=float(e.max()), mean_ey=float(e.mean()),
               off_track=off, steps=t)
    if log:
        out["frames"] = frames
    return out
