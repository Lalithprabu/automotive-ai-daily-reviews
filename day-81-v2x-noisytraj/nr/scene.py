"""Synthetic 3-lane highway traffic with interactions (IDM car-following, gap-checked lane changes, sudden braking).

Stand-in for the curated driving datasets used by arXiv 2611.28128 (Salehi, Lusvarghi, Sepulcre, Gozalvez, "Robust
Trajectory Prediction Under Sensing and V2X Noise: Evaluation and Mitigation Strategies").  Nothing here comes from the
paper beyond the idea: interacting road users whose CLEAN states are known, so noise can be added afterwards.

    frame       x along the road (+ = driving direction), y lateral (+ = left), metres.  Origin: the ego's clean x at
                t0 snapped to a 20 m grid (so the origin carries no noise information), y = 0 at the centre lane
    lanes       3 lanes of 3.5 m at y = -3.5, 0, +3.5
    agents      N = 8 vehicles (4.5 m long); agent 0 is the "ego" only for the frame origin
    timing      10 Hz; 20 history steps (2 s, last one is t0) + 30 future steps (3 s); 25 warm-up steps are discarded
    dynamics    longitudinal IDM toward a desired speed; lateral first-order move to a target lane; lane change when the
                leader is slow and the target lane has safe gaps; in 35% of scenes one agent brakes suddenly
    state       per step [x, y, heading, speed]
"""
from __future__ import annotations

import numpy as np

N_AG, T_H, T_F, WARM, DT = 8, 20, 30, 25, 0.1
LANE_W, N_LANE, CAR_L = 3.5, 3, 4.5
LANE_Y = (np.arange(N_LANE) - 1) * LANE_W            # [-3.5, 0, 3.5]


def _idm(v, v_des, gap, dv, a_max=1.5, b=2.5, s0=2.5, T=1.3):
    s_star = s0 + np.maximum(0.0, v * T + v * dv / (2 * np.sqrt(a_max * b)))
    return a_max * (1 - (v / np.maximum(v_des, 0.5)) ** 4 - (s_star / np.maximum(gap, 0.5)) ** 2)


def _leader(x, lane_of, lane_q, v):
    """For each agent i and query lane lane_q[i]: gap to and speed of the nearest agent ahead in that lane."""
    S, N = x.shape
    dx = x[:, None, :] - x[:, :, None]                                     # (S, i, j): x_j - x_i
    same = lane_of[:, None, :] == lane_q[:, :, None]
    ahead = same & (dx > 0) & ~np.eye(N, dtype=bool)[None]
    d = np.where(ahead, dx, np.inf)
    j = d.argmin(-1)
    gap = np.take_along_axis(d, j[..., None], -1)[..., 0] - CAR_L
    vl = np.take_along_axis(v, j, -1)
    has = np.isfinite(gap)
    return np.where(has, gap, 200.0), np.where(has, vl, 40.0)


def _follower_gap(x, lane_of, lane_q):
    S, N = x.shape
    dx = x[:, :, None] - x[:, None, :]                                     # x_i - x_j  (j behind i if > 0)
    same = lane_of[:, None, :] == lane_q[:, :, None]
    behind = same & (dx > 0) & ~np.eye(N, dtype=bool)[None]
    d = np.where(behind, dx, np.inf).min(-1) - CAR_L
    return np.where(np.isfinite(d), d, 200.0)


def simulate(S: int, rng: np.random.Generator) -> dict:
    """-> dict(states (S, N, T_H + T_F, 4) [x, y, heading, v], brake (S,) bool, brake_agent (S,))."""
    N, T = N_AG, WARM + T_H + T_F
    # ---- initial placement: per lane, sorted x with >= 14 m spacing
    lane = rng.integers(0, N_LANE, (S, N))
    x = np.zeros((S, N))
    for s in range(S):
        for l in range(N_LANE):
            idx = np.where(lane[s] == l)[0]
            if len(idx) == 0:
                continue
            gaps = rng.uniform(14, 40, len(idx))
            xs = np.cumsum(gaps) - gaps.sum() / 2 + rng.uniform(-15, 15)
            x[s, idx] = rng.permutation(xs)
    y = LANE_Y[lane] + rng.normal(0, 0.15, (S, N))
    v_des = np.where(rng.random((S, N)) < 0.25, rng.uniform(12, 18, (S, N)), rng.uniform(20, 31, (S, N)))
    v = np.clip(v_des + rng.normal(0, 2, (S, N)), 5, 33)
    vy = np.zeros((S, N))
    target = lane.copy()
    brake = rng.random(S) < 0.35
    brake_agent = rng.integers(0, N, S)
    brake_t = rng.integers(WARM + 8, WARM + T_H + 12, S)
    brake_v = rng.uniform(0, 8, S)
    out = np.zeros((S, N, T, 4), np.float32)
    for t in range(T):
        # sudden braking event (desired speed collapses)
        hit = brake & (t == brake_t)
        if hit.any():
            v_des[hit, brake_agent[hit]] = brake_v[hit]
        cur = np.abs(y[..., None] - LANE_Y).argmin(-1)                     # lane the agent occupies now
        occ = np.where(target != cur, target, cur)                         # merging agents block their target too
        gap, vl = _leader(x, occ, target, v)
        gap2, vl2 = _leader(x, cur, cur, v)                                # also respect the leader in the current lane
        use2 = gap2 < gap
        gap, vl = np.where(use2, gap2, gap), np.where(use2, vl2, vl)
        a = np.clip(_idm(v, v_des, gap, v - vl), -8.0, 2.0)
        # lane-change decision
        settled = (target == cur) & (np.abs(y - LANE_Y[target]) < 0.3)
        want = settled & (gap < 35) & (vl < v_des - 2) & (rng.random((S, N)) < 0.08)
        for side in (+1, -1):
            cand = np.clip(target + side, 0, N_LANE - 1)
            ok = want & (cand != target)
            if not ok.any():
                continue
            g_a, v_a = _leader(x, occ, cand, v)
            g_b = _follower_gap(x, occ, cand)
            safe = ok & (g_a > 12) & (g_b > 10) & (v_a > vl + 1)
            target = np.where(safe, cand, target)
            want = want & ~safe
        vy_t = np.clip(0.8 * (LANE_Y[target] - y), -1.3, 1.3)
        vy = vy + np.clip(vy_t - vy, -0.25, 0.25)                          # lateral rate limit
        v = np.clip(v + a * DT, 0.0, 35.0)
        x = x + v * DT
        y = y + vy * DT
        h = np.arctan2(vy, np.maximum(v, 0.1))
        out[:, :, t] = np.stack([x, y, h, v], -1)
    out = out[:, :, WARM:]
    return dict(states=out, brake=brake & (brake_t >= WARM), brake_agent=brake_agent)


def make_dataset(S: int, seed: int, chunk: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    parts = [simulate(min(chunk, S - i), rng) for i in range(0, S, chunk)]
    st = np.concatenate([p["states"] for p in parts])
    origin = np.floor(st[:, 0, T_H - 1, 0] / 20.0) * 20.0                 # ego clean x at t0, snapped to 20 m
    st[..., 0] -= origin[:, None, None]
    v2x = rng.random((S, N_AG)) < 0.5                                      # which agents are received over V2X
    v2x[:, 0] = False                                                      # the ego's own state is from its sensors
    return dict(states=st.astype(np.float32), v2x=v2x,
                brake=np.concatenate([p["brake"] for p in parts]),
                brake_agent=np.concatenate([p["brake_agent"] for p in parts]))


if __name__ == "__main__":
    import time
    t0 = time.time()
    d = make_dataset(500, 0)
    st = d["states"]
    print(f"500 scenes in {time.time() - t0:.1f}s", st.shape)
    lat = np.abs(st[:, :, -1, 1] - st[:, :, T_H - 1, 1])
    print("lane changes in future (|dy|>1.5):", (lat > 1.5).mean().round(3))
    dv = st[:, :, -1, 3] - st[:, :, T_H - 1, 3]
    print("speed change future: mean |dv|", np.abs(dv).mean().round(2), " share |dv|>3:", (np.abs(dv) > 3).mean().round(3))
    # collisions sanity: min longitudinal gap among same-lane pairs
    x, y = st[..., 0], st[..., 1]
    dx = np.abs(x[:, :, None] - x[:, None]); dy = np.abs(y[:, :, None] - y[:, None])
    over = (dx < CAR_L) & (dy < 1.8)
    over[:, np.arange(N_AG), np.arange(N_AG)] = False
    print("scene share with overlap:", over.any((1, 2, 3)).mean().round(3))
    # CV baseline error
    p0 = st[:, :, T_H - 1, :2]; vel = (st[:, :, T_H - 1, :2] - st[:, :, T_H - 2, :2]) / DT
    cv = p0[:, :, None] + vel[:, :, None] * (np.arange(1, T_F + 1) * DT)[:, None]
    print("CV ADE", np.linalg.norm(cv - st[:, :, T_H:, :2], axis=-1).mean().round(3))
