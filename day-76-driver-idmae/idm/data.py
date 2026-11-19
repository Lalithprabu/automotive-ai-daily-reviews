"""Synthetic driving-simulator generator: a stand-in for the paper's DIVERT dataset (which is not released).

The paper (arXiv 2611.23211, abstract) records simulator drives with gaze, vehicle state and scene video under ten
impairment inductions (cognitive, visual, emotional).  This generator produces the same KIND of signals, at 10 Hz,
for 4 conditions (my reduction of the ten inductions):

    0 none        baseline driving
    1 cognitive   e.g. n-back task: gaze concentrates on the road centre, fewer mirror checks, slower and more
                  intermittent steering corrections (larger reaction delay, more micro-reversals), slightly slower
    2 visual      e.g. phone task: periodic 1-2 s glances down to a phone; steering is frozen during the glance, so the
                  car drifts and is corrected abruptly afterwards; small speed drop
    3 emotional   e.g. anger induction: higher preferred speed, harsher throttle, higher-gain / noisier steering

Every driver (participant) has their own style and their own SUSCEPTIBILITY (0.1-1.0) that scales every impairment
effect, so impairment is subtle and person-dependent.  Train / test are split BY DRIVER.

Location-matched pairs: for each (driver, location) the same road segment and the same initial state are driven once
per condition, which is what the paper's "paired AUC (location-matched pairs)" needs.

Per-step channels (all float32, normalised to O(1)):
    gaze    (T, 2)   yaw / 30 deg, pitch / 30 deg
    veh     (T, 4)   lateral offset [m], (speed - 25) / 5 [m/s], steering * 50 [rad], longitudinal accel / 2 [m/s^2]
    road    (T, 1)   road curvature * 150 [1/m]  (the "scene" stream; never masked)
"""
from __future__ import annotations

import numpy as np

DT = 0.1
CONDS = ["none", "cognitive", "visual", "emotional"]
WB = 2.8                                                       # wheelbase [m]

# gaze areas of interest (yaw, pitch) in degrees
AOI = dict(left_mirror=(-40.0, -6.0), right_mirror=(45.0, -6.0), rear_mirror=(18.0, 12.0), speedo=(0.0, -18.0),
           phone=(12.0, -32.0))


# --------------------------------------------------------------------------------------------- road and driver
def road_curvature(loc_seed: int, n: int, ds: float = 1.0) -> np.ndarray:
    """Curvature kappa(s) [1/m] sampled every ds metres along the road (smooth sum of sinusoids, radius >= ~150 m)."""
    r = np.random.default_rng(10_000 + loc_seed)
    s = np.arange(n) * ds
    k = np.zeros(n)
    for _ in range(3):
        L = r.uniform(150, 600)
        k += r.uniform(-1, 1) / 3.0 / 160.0 * np.sin(2 * np.pi * s / L + r.uniform(0, 2 * np.pi))
    return k


def make_driver(seed: int) -> dict:
    r = np.random.default_rng(seed)
    return dict(v_pref=r.uniform(21, 29), k_v=r.uniform(0.25, 0.5), a_noise=r.uniform(0.1, 0.3),
                k_y=r.uniform(0.25, 0.45), k_psi=r.uniform(1.0, 1.6), tau=r.uniform(0.15, 0.3),
                st_noise=r.uniform(0.0006, 0.0014), preview=r.uniform(0.8, 1.3),
                gaze_disp=r.uniform(4.0, 7.0), yaw_bias=r.uniform(-3, 3), p_mirror=r.uniform(0.06, 0.14),
                p_speedo=r.uniform(0.04, 0.09), fix_mu=r.uniform(-1.1, -0.7), suscept=r.uniform(0.1, 1.0))


def cond_params(drv: dict, cond: int) -> dict:
    """Blend the driver's baseline parameters towards a fully impaired version, scaled by susceptibility."""
    p = dict(drv)
    a = drv["suscept"]
    lerp = lambda key, mult: p.__setitem__(key, p[key] * (1 + a * (mult - 1)))
    p["phone_rate"] = 0.0
    p["hold_thresh"] = 0.0
    if cond == 1:                                              # cognitive
        lerp("gaze_disp", 0.45); lerp("p_mirror", 0.25); lerp("p_speedo", 0.5); lerp("tau", 1.6)
        lerp("st_noise", 1.8); lerp("v_pref", 0.95); lerp("fix_mu", 1.0)
        p["hold_thresh"] = 0.25 * a                            # intermittent control: ignore small lateral errors
    elif cond == 2:                                            # visual (phone)
        p["phone_rate"] = 0.04 + 0.10 * a                      # glance onsets per second
        lerp("v_pref", 0.96); lerp("p_mirror", 0.6)
    elif cond == 3:                                            # emotional (anger)
        lerp("v_pref", 1.10); lerp("k_v", 2.0); lerp("a_noise", 2.5); lerp("k_y", 1.3); lerp("st_noise", 1.6)
        lerp("k_psi", 1.1); lerp("gaze_disp", 1.2)
    return p


# --------------------------------------------------------------------------------------------- one drive
def simulate_drive(drv: dict, cond_seq: np.ndarray, loc_seed: int, run_seed: int, init_seed: int | None = None):
    """Simulate a drive whose condition may change per step.  cond_seq (T,) ints.

    Returns dict of per-step arrays: gaze (T,2) deg, veh raw (s, y, psi, v, delta, acc) and kappa (T,), plus
    'phone' (T,) bool (eyes on phone).  init_seed fixes the initial state (location matching), run_seed the noise.
    """
    T = len(cond_seq)
    rng = np.random.default_rng(run_seed)
    ir = np.random.default_rng(init_seed if init_seed is not None else run_seed + 7)
    kap = road_curvature(loc_seed, 4000)
    kfun = lambda s: np.interp(s, np.arange(len(kap)), kap)
    s, y, psi = ir.uniform(0, 200), ir.normal(0, 0.15), ir.normal(0, 0.005)
    v = drv["v_pref"] + ir.normal(0, 1.0)
    delta = WB * kfun(s)
    hist_y, hist_psi = [y] * 10, [psi] * 10
    out = {k: np.zeros(T) for k in ("s", "y", "psi", "v", "delta", "acc", "kappa", "gyaw", "gpitch")}
    phone = np.zeros(T, bool)
    fix_left, fix_target, glance_left = 0, ("road", 0.0, 0.0), 0
    gy, gp = 0.0, -2.0
    for t in range(T):
        p = cond_params(drv, int(cond_seq[t]))
        kap_now = kfun(s)
        # ---------------- gaze (fixation sequence)
        if glance_left > 0:
            glance_left -= 1
            tgt = AOI["phone"]; gy += 0.6 * (tgt[0] - gy); gp += 0.6 * (tgt[1] - gp); phone[t] = True
        else:
            if p["phone_rate"] > 0 and rng.random() < p["phone_rate"] * DT:
                glance_left = int(rng.uniform(0.8, 1.8) / DT)
            if fix_left <= 0:
                u = rng.random()
                if u < p["p_mirror"]:
                    name = rng.choice(["left_mirror", "right_mirror", "rear_mirror"])
                    fix_target = ("aoi",) + AOI[name]
                    fix_left = int(np.clip(rng.lognormal(-0.9, 0.3), 0.2, 1.0) / DT)
                elif u < p["p_mirror"] + p["p_speedo"]:
                    fix_target = ("aoi",) + AOI["speedo"]
                    fix_left = int(np.clip(rng.lognormal(-1.1, 0.3), 0.2, 0.8) / DT)
                else:                                          # road: look towards the curve, with dispersion
                    fix_target = ("road", rng.normal(0, p["gaze_disp"]), rng.normal(0, p["gaze_disp"] * 0.35))
                    fix_left = int(np.clip(rng.lognormal(p["fix_mu"], 0.4), 0.15, 2.0) / DT)
            fix_left -= 1
            if fix_target[0] == "road":
                ty = np.degrees(np.arctan(kap_now * v * 1.8 * p["preview"])) * 3.0 + p["yaw_bias"] + fix_target[1]
                tp = -2.0 + fix_target[2]
            else:
                ty, tp = fix_target[1], fix_target[2]
            gy += 0.6 * (ty - gy); gp += 0.6 * (tp - gp)
        # ---------------- lateral control (delayed perception, frozen while eyes on phone)
        d_steps = int(round(p["tau"] / DT))
        yo, po = hist_y[-1 - min(d_steps, 9)], hist_psi[-1 - min(d_steps, 9)]
        k_prev = kfun(s + v * p["preview"])
        if not phone[t]:
            err = p["k_y"] * yo + p["k_psi"] * po * 10
            if abs(yo) < p["hold_thresh"] and rng.random() < 0.7:
                err *= 0.2                                     # intermittent: small errors mostly ignored
            d_cmd = WB * k_prev - 0.02 * err + rng.normal(0, p["st_noise"])
            delta += 0.5 * (d_cmd - delta)
        # ---------------- longitudinal control
        v_des = min(p["v_pref"], np.sqrt(3.0 / max(abs(k_prev), 1e-4)))
        if phone[t]:
            v_des -= 1.0
        acc = np.clip(p["k_v"] * (v_des - v) + rng.normal(0, p["a_noise"]), -4, 3)
        # ---------------- kinematics in the road (Frenet) frame
        out["s"][t], out["y"][t], out["psi"][t], out["v"][t] = s, y, psi, v
        out["delta"][t], out["acc"][t], out["kappa"][t] = delta, acc, kap_now
        out["gyaw"][t] = gy + rng.normal(0, 0.7); out["gpitch"][t] = gp + rng.normal(0, 0.7)
        psi += (v * np.tan(delta) / WB - v * kap_now / max(1 - kap_now * y, 0.5)) * DT
        y += v * np.sin(psi) * DT
        s += v * np.cos(psi) / max(1 - kap_now * y, 0.5) * DT
        v = max(1.0, v + acc * DT)
        hist_y = hist_y[1:] + [y]; hist_psi = hist_psi[1:] + [psi]
    out["phone"] = phone
    return out


def to_features(d: dict):
    """Raw drive dict -> (gaze (T,2), veh (T,4), road (T,1)) float32 normalised channels."""
    gaze = np.stack([d["gyaw"] / 30.0, d["gpitch"] / 30.0], -1)
    veh = np.stack([d["y"], (d["v"] - 25.0) / 5.0, d["delta"] * 50.0, d["acc"] / 2.0], -1)
    road = (d["kappa"] * 150.0)[:, None]
    return gaze.astype(np.float32), veh.astype(np.float32), road.astype(np.float32)


# --------------------------------------------------------------------------------------------- dataset
def make_dataset(driver_ids, n_loc: int, win_s: float = 10.0, warm_s: float = 3.0, seed: int = 0):
    """Location-matched windows: for every driver x location, one window per condition (same road, same start).

    Returns dict: gaze (N,T,2), veh (N,T,4), road (N,T,1), y (N,) condition, driver (N,), pair (N,) pair-group id,
    suscept (N,).  T = win_s / DT.  A warm-up of warm_s seconds is simulated and dropped.
    """
    T, W = int(win_s / DT), int(warm_s / DT)
    G, V, R, Y, D, P, S = [], [], [], [], [], [], []
    for di in driver_ids:
        drv = make_driver(int(di))
        for li in range(n_loc):
            loc = int(di) * 1000 + li
            for c in range(len(CONDS)):
                d = simulate_drive(drv, np.full(T + W, c), loc_seed=loc, run_seed=seed * 7919 + loc * 13 + c,
                                   init_seed=loc)
                g, v, r = to_features(d)
                G.append(g[W:]); V.append(v[W:]); R.append(r[W:])
                Y.append(c); D.append(int(di)); P.append(loc); S.append(drv["suscept"])
    a = lambda x, t=np.float32: np.asarray(x, t)
    return dict(gaze=a(G), veh=a(V), road=a(R), y=a(Y, np.int64), driver=a(D, np.int64), pair=a(P, np.int64),
                suscept=a(S))
