"""Synthetic car-following world used to test a NegWAM-style model (NOT NAVSIM data).

One scene = ego car on a gently curved lane + a lead car that may be braking.
The expert is an IDM-style driver. "Negatives" are trajectories that stay *close to
the expert* but are unsafe (too fast -> gap/collision, or lateral drift -> off-lane),
mirroring the paper's idea of "near-expert negatives".

Conventions (ego frame at t=0): x forward [m], y left [m]. T future steps, DT seconds apart.
Scene feature vector (7): [v0, kappa, y0, d_lead, v_lead, a_lead, v_desired]
"""
from __future__ import annotations
import numpy as np

T = 8                 # future waypoints
DT = 0.5              # seconds between waypoints (4 s horizon)
CAR_LEN = 5.0         # centre-to-centre distance below which cars overlap [m]
LANE_HALF = 1.6       # |y - centreline| above this = off-lane [m]
HEADWAY_MIN = 0.5     # min time headway for the toy "TTC" gate [s]
COORD_SCALE = np.array([30.0, 2.0], dtype=np.float32)   # normalisation of (x, y)
SCENE_MEAN = np.array([9.0, 0.0, 0.0, 30.0, 6.0, -1.0, 12.0], dtype=np.float32)
SCENE_STD = np.array([4.0, 0.01, 0.3, 15.0, 4.0, 1.5, 2.0], dtype=np.float32)


def lead_positions(sc: np.ndarray) -> np.ndarray:
    """Lead-car longitudinal position at the T waypoint times. sc: (N,7) -> (N,T). Lead never reverses."""
    t = (np.arange(1, T + 1) * DT)[None, :]
    d, vl, al = sc[:, 3:4], sc[:, 4:5], sc[:, 5:6]
    t_stop = np.where(al < -1e-6, vl / np.maximum(-al, 1e-6), 1e9)
    tt = np.minimum(t, t_stop)
    return d + vl * tt + 0.5 * al * tt ** 2


def centreline(x: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    return 0.5 * kappa * x ** 2


def expert_trajectory(sc: np.ndarray) -> np.ndarray:
    """IDM follower integrated at 0.1 s then sampled every DT. Returns (N,T,2)."""
    N = sc.shape[0]
    v0, kappa, y0, d, vl, al, vd = [sc[:, i] for i in range(7)]
    sub = int(round(DT / 0.1))
    x = np.zeros(N); v = v0.copy()
    xl, vlead = d.copy(), vl.copy()
    out = np.zeros((N, T, 2))
    for k in range(T * sub):
        s = np.maximum(xl - x - CAR_LEN, 0.1)
        dv = v - vlead
        s_star = 2.0 + v * 1.4 + v * dv / (2 * np.sqrt(1.5 * 2.0))
        acc = 1.5 * (1 - (v / vd) ** 4 - (np.maximum(s_star, 0) / s) ** 2)
        acc = np.clip(acc, -7.0, 2.0)
        v = np.maximum(v + acc * 0.1, 0.0)
        x = x + v * 0.1
        vlead = np.maximum(vlead + al * 0.1, 0.0)
        xl = xl + vlead * 0.1
        if (k + 1) % sub == 0:
            j = (k + 1) // sub - 1
            out[:, j, 0] = x
    out[:, :, 1] = centreline(out[:, :, 0], kappa[:, None]) + y0[:, None] * np.exp(-out[:, :, 0] / 15.0)
    return out


def safety(sc: np.ndarray, traj: np.ndarray) -> dict:
    """Vectorised toy safety metrics. traj (N,T,2) in metres. Returns dict of (N,) arrays."""
    x, y = traj[..., 0], traj[..., 1]
    kappa = sc[:, 1:2]
    xl = lead_positions(sc)
    yl = centreline(xl, kappa)
    gap = xl - x
    collide = ((gap < CAR_LEN) & (np.abs(y - yl) < 1.8)).any(1)
    offlane = (np.abs(y - centreline(x, kappa)) > LANE_HALF).any(1)
    vx = np.concatenate([(sc[:, 0:1] * 0 + x[:, :1]) / DT, np.diff(x, axis=1) / DT], axis=1)  # speed along x
    headway = (gap - CAR_LEN) / np.maximum(vx, 1.0)
    ttc_ok = ~(headway < HEADWAY_MIN).any(1)
    return dict(collision=collide, offlane=offlane, ttc_ok=ttc_ok, min_gap=gap.min(1), x_end=x[:, -1])


def toy_score(sc, traj, ref_traj) -> np.ndarray:
    """Multiplicative-gate score in the spirit of (E)PDMS: nc * dac * (0.6*ttc + 0.4*progress). Toy, not the real metric."""
    s = safety(sc, traj)
    prog = np.clip(traj[:, -1, 0] / np.maximum(ref_traj[:, -1, 0], 1.0), 0, 1)
    return (~s["collision"]) * (~s["offlane"]) * (0.6 * s["ttc_ok"] + 0.4 * prog)


def sample_scenes(n: int, rng: np.random.Generator, hard: bool = False) -> np.ndarray:
    """Random scenes. hard=True keeps only scenes where the expert's own margin is thin (close lead, braking)."""
    out = []
    while sum(len(o) for o in out) < n:
        m = n * 3
        v0 = rng.uniform(3, 15, m); kappa = rng.uniform(-0.02, 0.02, m); y0 = rng.uniform(-0.6, 0.6, m)
        d = rng.uniform(10, 55, m); vl = rng.uniform(0, 13, m)
        al = np.where(rng.random(m) < 0.5, rng.uniform(-4.0, 0, m), rng.uniform(-0.3, 0.3, m))
        vd = rng.uniform(10, 14, m)
        sc = np.stack([v0, kappa, y0, d, vl, al, vd], 1).astype(np.float64)
        ex = expert_trajectory(sc)
        s = safety(sc, ex)
        ok = ~s["collision"] & ~s["offlane"] & s["ttc_ok"]
        if hard:
            ok &= (s["min_gap"] - CAR_LEN) < 8.0 + 0.0 * s["min_gap"]    # thin margin to the lead
        out.append(sc[ok])
    return np.concatenate(out)[:n].astype(np.float32)


def make_negatives(sc: np.ndarray, ex: np.ndarray, rng: np.random.Generator):
    """Near-expert unsafe trajectories.
    neg_long: stretch the expert longitudinally (drive faster than the expert) until unsafe (collision / headway).
    neg_lat : add a growing lateral drift until off-lane.
    The smallest perturbation level that is unsafe is kept -> negatives hug the expert (hard negatives)."""
    N = len(sc)
    ramp = np.linspace(1 / T, 1, T)[None, :]
    long_levels = np.linspace(0.04, 0.6, 15)
    neg_long = ex.copy()
    found = np.zeros(N, bool)
    for lv in long_levels:
        cand = ex.copy()
        cand[..., 0] = ex[..., 0] * (1 + lv * ramp)
        cand[..., 1] = centreline(cand[..., 0], sc[:, 1:2]) + (ex[..., 1] - centreline(ex[..., 0], sc[:, 1:2]))
        s = safety(sc, cand)
        bad = (s["collision"] | ~s["ttc_ok"]) & ~found
        neg_long[bad] = cand[bad]; found |= bad
    sign = rng.choice([-1.0, 1.0], N)[:, None]
    neg_lat = ex.copy(); found_l = np.zeros(N, bool)
    for lv in np.linspace(0.3, 2.5, 12):
        cand = ex.copy(); cand[..., 1] = ex[..., 1] + sign * lv * ramp
        bad = safety(sc, cand)["offlane"] & ~found_l
        neg_lat[bad] = cand[bad]; found_l |= bad
    # where no unsafe level was found fall back to a strongly unsafe one (keeps array shapes fixed)
    fb = ex[~found_l].copy(); fb[..., 1] += sign[~found_l] * 2.5 * ramp
    neg_lat[~found_l] = fb
    # free-road scenes where no speed-up is unsafe: reuse the lateral negative (every negative is genuinely unsafe)
    neg_long[~found] = neg_lat[~found]
    return neg_long.astype(np.float32), neg_lat.astype(np.float32), found, found_l


def build_dataset(n: int, seed: int, hard: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    sc = sample_scenes(n, rng, hard)
    ex = expert_trajectory(sc.astype(np.float64))
    nl, nt, f1, f2 = make_negatives(sc.astype(np.float64), ex, rng)
    return dict(scene=sc, expert=ex.astype(np.float32), neg_long=nl, neg_lat=nt,
                lead=lead_positions(sc.astype(np.float64)).astype(np.float32))


def norm_scene(sc):
    return (sc - SCENE_MEAN) / SCENE_STD


def norm_traj(tr):
    return tr / COORD_SCALE


def denorm_traj(tr):
    return tr * COORD_SCALE
