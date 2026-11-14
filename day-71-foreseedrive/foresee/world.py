"""Synthetic 2-lane urban driving world with HIDDEN future intent (toy stand-in for NAVSIM / nuScenes).

Ego at the origin at t = 0, heading +x, in the right lane (y = 0).  Left lane centre y = 3.5.
Drivable band y in [-1.75, 5.25].  Sidewalks at y = -3.5 (right) and y = 7.0 (left).

Each scene mixes some of:
  * a LEAD vehicle in the ego lane that may brake hard at a hidden time      (future-only event)
  * a PEDESTRIAN on a sidewalk who may step out and cross                      (weak cue in history)
  * a CUT-IN vehicle in the left lane that may merge into the ego lane         (weak cue in history)
  * left-lane TRAFFIC at constant speed (blocks a lane change)

The "expert" plans with the TRUE future (like a logged human driver who saw what happened) by
scoring a lattice of {keep lane, change left} x {2-stage accel profile} plans.  A planner that only
sees the present must infer that future, which is exactly what ForeseeDrive's future tokens target.

All units SI.  DT = 0.5 s, H = 8 future steps (4 s), HIST = 4 past steps incl. the present.
"""
from __future__ import annotations

import numpy as np

DT, H, HIST = 0.5, 8, 4
N_ACT = 6
LANE_W, LEFT_Y = 3.5, 3.5
Y_MIN, Y_MAX = -1.75, 5.25
R_EGO, R_VEH, R_PED = 1.3, 1.3, 0.5
V_MAX_EGO = 14.0
TYPE_VEH, TYPE_PED = 0, 1
T_FUT = DT * np.arange(1, H + 1)                     # (H,)  0.5 .. 4.0 s
T_HIS = DT * np.arange(-(HIST - 1), 1)               # (HIST,) -1.5 .. 0 s


def wrap_none(x):
    return x


# ----------------------------------------------------------------------------- ego plan helpers
def rollout_profile(v0: float, a1, a2, lane_change, t_lc: float = 3.0, sub: int = 2):
    """Vectorised ego plans.

    a1, a2, lane_change: broadcastable arrays (P,).  Accel a1 for t < 2 s, a2 after.
    Returns xy (P, H*sub, 2) on a fine grid (dt/sub) and the coarse (P, H, 2) samples.
    """
    a1, a2, lc = np.broadcast_arrays(np.asarray(a1, float), np.asarray(a2, float), np.asarray(lane_change, float))
    n = H * sub
    dt = DT / sub
    t = dt * np.arange(1, n + 1)
    acc = np.where(t[None] <= 2.0, a1[:, None], a2[:, None])                  # (P, n)
    v = np.empty_like(acc); x = np.empty_like(acc)
    vp, xp = np.full(len(a1), v0), np.zeros(len(a1))
    for i in range(n):
        vn = np.clip(vp + acc[:, i] * dt, 0.0, V_MAX_EGO)
        xp = xp + 0.5 * (vp + vn) * dt
        vp = vn
        v[:, i], x[:, i] = vn, xp
    s = np.clip(t / t_lc, 0, 1)
    y = lc[:, None] * LEFT_Y * (3 * s ** 2 - 2 * s ** 3)[None]                 # smoothstep lane change
    fine = np.stack([x, y], -1)
    return fine, fine[:, sub - 1::sub]


def clearance(ego_xy: np.ndarray, act_xy: np.ndarray, act_r: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Min over time & actors of (centre distance - radii).

    ego_xy (P, T, 2), act_xy (N, T, 2), act_r (N,), valid (N,)  ->  (P,)   (inf if no actor)
    """
    if valid.sum() == 0:
        return np.full(ego_xy.shape[0], np.inf)
    d = np.linalg.norm(ego_xy[:, None] - act_xy[None, valid], axis=-1)         # (P, Nv, T)
    return (d - (R_EGO + act_r[valid])[None, :, None]).min(axis=(1, 2))


def upsample(fut: np.ndarray, cur: np.ndarray, sub: int = 2) -> np.ndarray:
    """Linear interpolation of (N, H, 2) futures to the fine grid, starting from the present (N, 2)."""
    full = np.concatenate([cur[:, None], fut], 1)                               # (N, H+1, 2)
    out = []
    for i in range(H):
        for j in range(1, sub + 1):
            w = j / sub
            out.append((1 - w) * full[:, i] + w * full[:, i + 1])
    return np.stack(out, 1)                                                     # (N, H*sub, 2)


# ----------------------------------------------------------------------------- scene sampler
def _const_vel(p0, v, t):
    return p0[None] + v[None] * t[:, None]


def sample_scene(rng: np.random.Generator) -> dict:
    """One scene: ego speed, actor history (N, HIST, 4), actor future (N, H, 2), type, valid, events."""
    v_ego = rng.uniform(6.0, 12.0)
    hist = np.zeros((N_ACT, HIST, 4)); fut = np.zeros((N_ACT, H, 2))
    typ = np.zeros(N_ACT, int); valid = np.zeros(N_ACT, bool); event = np.zeros(N_ACT, int)
    k = 0

    def add(h, f, ty, ev=0):
        nonlocal k
        if k < N_ACT:
            hist[k], fut[k], typ[k], valid[k], event[k] = h, f, ty, True, ev
            k += 1

    # --- lead vehicle (may brake hard at a hidden time; brake lights are NOT observed)
    if rng.random() < 0.7:
        x0, v = rng.uniform(12, 40), rng.uniform(3.0, max(3.5, v_ego))
        brake = rng.random() < 0.4
        tb, dec = rng.uniform(0.0, 2.0), rng.uniform(4.0, 6.0)
        th = T_HIS
        h = np.stack([x0 + v * th, np.zeros(HIST), np.full(HIST, v), np.zeros(HIST)], -1)
        xs, vv, xp = [], v, x0
        tt = np.concatenate([[0.0], T_FUT])
        fine = np.linspace(0, H * DT, H * 10 + 1)
        xf = [x0]
        for i in range(1, len(fine)):
            a = -dec if (brake and fine[i] > tb) else 0.0
            vn = max(0.0, vv + a * (fine[i] - fine[i - 1]))
            xp = xp + 0.5 * (vv + vn) * (fine[i] - fine[i - 1]); vv = vn
            xf.append(xp)
        xf = np.interp(T_FUT, fine, np.array(xf))
        add(h, np.stack([xf, np.zeros(H)], -1), TYPE_VEH, 1 if brake else 0)

    # --- pedestrian at the kerb (crosses or not; weak lateral-velocity cue)
    if rng.random() < 0.6:
        side = rng.choice([-1, 1])
        y0 = -3.5 if side < 0 else 7.0
        to_road = 1.0 if side < 0 else -1.0
        x0 = rng.uniform(8, 35)
        vx = rng.uniform(-1.2, 1.2)
        cross = rng.random() < 0.5
        cue = rng.normal(0.35 if cross else 0.0, 0.2) * to_road          # weak, noisy lateral velocity
        h = np.stack([x0 + vx * T_HIS, y0 + cue * T_HIS, np.full(HIST, vx), np.full(HIST, cue)], -1)
        if cross:
            tc, vy = rng.uniform(0.0, 1.5), rng.uniform(1.2, 1.9)
            lat = np.where(T_FUT > tc, (T_FUT - tc) * vy, 0.0) * to_road + cue * np.minimum(T_FUT, tc)
            f = np.stack([x0 + vx * 0.3 * T_FUT, y0 + lat], -1)
        else:
            f = np.stack([x0 + vx * T_FUT, y0 + cue * T_FUT * 0.3], -1)
        add(h, f, TYPE_PED, 2 if cross else 0)

    # --- left-lane vehicle that may cut in (weak lateral drift cue)
    if rng.random() < 0.4:
        x0, v = rng.uniform(4, 25), np.clip(v_ego + rng.uniform(-3, 1), 2, 14)
        cut = rng.random() < 0.5
        cue = rng.normal(-0.25 if cut else 0.0, 0.15)
        h = np.stack([x0 + v * T_HIS, LEFT_Y + cue * T_HIS, np.full(HIST, v), np.full(HIST, cue)], -1)
        if cut:
            tc = rng.uniform(0.0, 1.5)
            s = np.clip((T_FUT - tc) / 2.0, 0, 1)
            y = LEFT_Y - LEFT_Y * (3 * s ** 2 - 2 * s ** 3)
            f = np.stack([x0 + v * T_FUT * 0.85, y], -1)
        else:
            f = np.stack([x0 + v * T_FUT, LEFT_Y + 0 * T_FUT], -1)
        add(h, f, TYPE_VEH, 3 if cut else 0)

    # --- left-lane traffic (constant speed)
    for _ in range(rng.integers(0, 3)):
        x0, v = rng.uniform(-20, 50), rng.uniform(4, 13)
        h = np.stack([x0 + v * T_HIS, np.full(HIST, LEFT_Y), np.full(HIST, v), np.zeros(HIST)], -1)
        add(h, np.stack([x0 + v * T_FUT, np.full(H, LEFT_Y)], -1), TYPE_VEH, 0)

    radius = np.where(typ == TYPE_PED, R_PED, R_VEH)
    sc = dict(v_ego=v_ego, hist=hist, fut=fut, typ=typ, valid=valid, event=event, radius=radius)
    sc["expert"] = expert_plan(sc)
    return sc


# ----------------------------------------------------------------------------- expert (sees the future)
A_GRID = np.array([-6.0, -4.5, -3.0, -1.5, 0.0, 1.0, 2.0])


def plan_lattice(v0: float):
    a1, a2, lc = np.meshgrid(A_GRID, A_GRID, [0.0, 1.0], indexing="ij")
    a1, a2, lc = a1.ravel(), a2.ravel(), lc.ravel()
    fine, coarse = rollout_profile(v0, a1, a2, lc)
    return a1, a2, lc, fine, coarse


def expert_plan(sc: dict) -> np.ndarray:
    a1, a2, lc, fine, coarse = plan_lattice(sc["v_ego"])
    cur = sc["hist"][:, -1, :2]
    act = upsample(sc["fut"], cur)
    clr = clearance(fine, act, sc["radius"], sc["valid"])
    progress = coarse[:, -1, 0]
    jerk = np.abs(a2 - a1)
    score = (progress
             - 0.8 * np.abs(a1) - 0.5 * np.abs(a2) - 0.6 * jerk
             - 8.0 * lc                                           # lane change only when it pays
             - 40.0 * (clr < 0.5) - 8.0 * np.clip(1.5 - clr, 0, 1.5))
    return coarse[int(np.argmax(score))]


# ----------------------------------------------------------------------------- metrics (PDMS-like)
def pdm_score(traj: np.ndarray, sc: dict, expert_progress: float | None = None) -> dict:
    """NAVSIM-flavoured score of one ego trajectory (H, 2) against the TRUE future."""
    cur = sc["hist"][:, -1, :2]
    fine_e = upsample(traj[None], np.zeros((1, 2)))
    act = upsample(sc["fut"], cur)
    clr = float(clearance(fine_e, act, sc["radius"], sc["valid"])[0])
    nc = float(clr > 0)
    dac = float((traj[:, 1] > Y_MIN - 0.25).all() and (traj[:, 1] < Y_MAX + 0.25).all())
    ttc = float(clr > 0.5)
    p = np.concatenate([[[0, 0]], traj], 0)
    v = np.linalg.norm(np.diff(p, axis=0), axis=-1) / DT
    v = np.concatenate([[sc["v_ego"]], v])
    acc = np.diff(v) / DT
    comf = float(np.abs(acc).max() <= 6.5 and np.abs(np.diff(acc) / DT).max() <= 12.0)
    ep_ref = expert_progress if expert_progress is not None else sc["expert"][-1, 0]
    ep = float(np.clip(traj[-1, 0] / max(ep_ref, 1.0), 0, 1))
    pdms = nc * dac * (5 * ttc + 2 * comf + 5 * ep) / 12
    return dict(pdms=pdms, nc=nc, dac=dac, ttc=ttc, comf=comf, ep=ep, clear=clr)
