"""Synthetic 4-way-intersection scenes: a stand-in for ETH/UCY, NBA and SDD used by arXiv 2611.26390 (SpecFlow).

SpecFlow forecasts the futures of ALL agents in a scene (best-of-20).  The paper's benchmarks are not bundled here, so
this module simulates an urban intersection with vehicles and pedestrians (my construction, not the paper's):

    geometry    4 arms, right-hand traffic, 2 inbound lanes per arm (left lane y=-1.75, right lane y=-5.25 in the
                canonical eastbound frame), stop line at x=-9, sidewalks at |y| = 9
    lights      one phase per scene: east-west green / north-south red or the reverse; vehicles on red arms stop
    vehicles    path = straight -> (left arc R=8.75 | right arc R=6.75 | none) -> straight, followed by arc length s
                speed: cruise 8-13 m/s, slow to sqrt(a_lat R) before turns, brake to the stop line on red,
                IDM-lite car following behind a leader in the same lane (this is the only interaction)
    pedestrians walk the sidewalk towards a corner, then cross, turn the corner or wait at the curb
    sampling    simulated at 0.1 s, sub-sampled to dt = 0.4 s (ETH/UCY rate): 8 observed + 12 future frames
                (3.2 s history, 4.8 s horizon); 0.03 m Gaussian position noise on the "measured" tracks

The future is MULTIMODAL by construction: a vehicle 20 m before the intersection may go straight, turn or stop, and the
history only partly gives it away (lane choice, early deceleration).

Model inputs are agent-centric (origin = last observed position, x-axis = heading), built by `agent_frame()`.
"""
from __future__ import annotations

import numpy as np

DT, FINE, SUB = 0.4, 0.1, 4
T_OBS, T_PRED = 8, 12
T_ALL = T_OBS + T_PRED
A_MAX = 6
NOISE = 0.03
STOP_X = -9.0
VEH, PED = 0, 1
MANEUVERS = ["straight", "left", "right", "stop", "ped-cross", "ped-turn", "ped-wait"]


# --------------------------------------------------------------------------------------------- paths
def make_path(x0: float, y0: float, segs: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Dense polyline (0.1 m) starting at (x0, y0) heading +x.  segs: ('S', length) or ('A', radius, signed_angle).
    Returns pts (N, 2) and cumulative arc length s (N,)."""
    chunks = [np.array([[x0, y0]])]
    h, cur = 0.0, np.array([x0, y0], float)
    for sg in segs:
        if sg[0] == "S":
            n = max(2, int(sg[1] / 0.1))
            d = np.arange(1, n + 1)[:, None] * 0.1 * np.array([np.cos(h), np.sin(h)])
            seg = cur + d
        else:
            R, ang = sg[1], sg[2]
            n = max(2, int(abs(ang) * R / 0.1))
            sgn = np.sign(ang)
            ctr = cur + R * np.array([np.cos(h + sgn * np.pi / 2), np.sin(h + sgn * np.pi / 2)])
            th = h - sgn * np.pi / 2 + np.arange(1, n + 1) * ang / n
            seg = ctr + R * np.stack([np.cos(th), np.sin(th)], -1)
            h += ang
        chunks.append(seg)
        cur = seg[-1]
    pts = np.concatenate(chunks)
    P = np.asarray(pts)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return P, s


def rot(k: int) -> np.ndarray:
    a = k * np.pi / 2
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


# --------------------------------------------------------------------------------------------- one scene
def make_scene(rng: np.random.Generator) -> dict:
    n_veh = int(rng.integers(2, 6))
    n_ped = int(rng.integers(0, min(2, A_MAX - n_veh) + 1))
    ew_green = rng.random() < 0.5
    green = [0, 2] if ew_green else [1, 3]
    agents = []
    lane_tail: dict = {}                                   # (arm, lane) -> x of the last vehicle spawned there
    for _ in range(n_veh):
        arm = int(rng.choice(green)) if rng.random() < 0.7 else int(rng.integers(0, 4))
        lane = int(rng.integers(0, 2))                     # 0 = left lane, 1 = right lane
        y = -1.75 if lane == 0 else -5.25
        red = arm not in green
        u = rng.random()
        if red:
            man = 3
        elif lane == 0:
            man = 0 if u < 0.55 else 1
        else:
            man = 0 if u < 0.55 else 2
        v_cr = rng.uniform(8, 13)
        key = (arm, lane)
        if key in lane_tail:
            x_start = lane_tail[key] - rng.uniform(14, 30)
        else:
            x_start = STOP_X - v_cr * (3.2 + rng.uniform(-0.8, 3.5))
        lane_tail[key] = x_start
        a_lat = rng.uniform(2.0, 3.0)
        if man == 1:
            pre = -7.0 - x_start
            P, s = make_path(x_start, y, [("S", pre), ("A", 8.75, np.pi / 2), ("S", 80)])
            turn = (pre, pre + 8.75 * np.pi / 2, np.sqrt(a_lat * 8.75))
        elif man == 2:
            pre = -12.0 - x_start
            P, s = make_path(x_start, y, [("S", pre), ("A", 6.75, -np.pi / 2), ("S", 80)])
            turn = (pre, pre + 6.75 * np.pi / 2, np.sqrt(a_lat * 6.75))
        else:
            P, s = make_path(x_start, y, [("S", 200)])
            turn = None
        agents.append(dict(kind=VEH, arm=arm, lane=lane, man=man, P=P, s=s, turn=turn, v_cr=v_cr,
                           v=v_cr * rng.uniform(0.85, 1.0), pos_s=0.0, x_start=x_start,
                           b=rng.uniform(1.5, 2.5), stop_s=STOP_X - x_start - 0.5 if man == 3 else None,
                           div_x=(-7.0 if man == 1 else -12.0 if man == 2 else 1e9)))
    for _ in range(n_ped):
        arm = int(rng.integers(0, 4))
        v_w = rng.uniform(1.0, 1.7)
        x_start = -9.5 - v_w * (3.2 + rng.uniform(-1.0, 3.0))
        u = rng.random()
        man = 4 if u < 0.45 else 5 if u < 0.8 else 6
        pre = -9.5 - x_start
        if man == 4:
            P, s = make_path(x_start, -9.0, [("S", pre), ("A", 1.0, np.pi / 2), ("S", 40)])
        elif man == 5:
            P, s = make_path(x_start, -9.0, [("S", pre), ("A", 1.0, -np.pi / 2), ("S", 40)])
        else:
            P, s = make_path(x_start, -9.0, [("S", 60)])
        agents.append(dict(kind=PED, arm=arm, lane=-1, man=man, P=P, s=s, turn=None, v_cr=v_w,
                           v=v_w * rng.uniform(0.9, 1.05), pos_s=0.0, x_start=x_start, b=0.8,
                           stop_s=pre - 0.3 if man == 6 else None, div_x=1e9))
    # ---- simulate 8 s at 0.1 s
    n_fine = T_ALL * SUB
    traj = np.zeros((len(agents), n_fine, 2))
    for t in range(n_fine):
        xs = [ag["x_start"] + ag["pos_s"] for ag in agents]
        for i, ag in enumerate(agents):
            P, s = ag["P"], ag["s"]
            traj[i, t] = [np.interp(ag["pos_s"], s, P[:, 0]), np.interp(ag["pos_s"], s, P[:, 1])]
            vt = ag["v_cr"]
            if ag["turn"] is not None:
                s0, s1, vturn = ag["turn"]
                if ag["pos_s"] < s0:
                    vt = min(vt, np.sqrt(vturn ** 2 + 2 * ag["b"] * (s0 - ag["pos_s"])))
                elif ag["pos_s"] < s1:
                    vt = min(vt, vturn)
            if ag["stop_s"] is not None:
                vt = min(vt, np.sqrt(max(0.0, 2 * ag["b"] * (ag["stop_s"] - ag["pos_s"]))))
            if ag["kind"] == VEH:                          # leader in the same lane, before either diverges
                for j, o in enumerate(agents):
                    if j == i or o["kind"] != VEH or o["arm"] != ag["arm"] or o["lane"] != ag["lane"]:
                        continue
                    if xs[j] > xs[i] and xs[j] < min(o["div_x"], ag["div_x"]) + 2.0:
                        gap = xs[j] - xs[i] - 5.0
                        g_star = 2.0 + 1.2 * ag["v"]
                        if gap < g_star:
                            vt = min(vt, max(0.0, o["v"] * gap / g_star - 0.5 * (g_star - gap) / g_star))
            a_max, a_min = (1.8, -4.0) if ag["kind"] == VEH else (0.8, -1.5)
            acc = np.clip((vt - ag["v"]) / 0.8, a_min, a_max)
            ag["v"] = max(0.0, ag["v"] + acc * FINE)
            ag["pos_s"] += ag["v"] * FINE
    clean = traj[:, ::SUB]                                  # (n, 20, 2)
    for i, ag in enumerate(agents):
        clean[i] = clean[i] @ rot(ag["arm"]).T
    return dict(clean=clean, kind=np.array([a["kind"] for a in agents]), man=np.array([a["man"] for a in agents]),
                arm=np.array([a["arm"] for a in agents]), green=green)


def make_dataset(n_scenes: int, seed: int) -> dict:
    """Returns pos (S, A, 20, 2) noisy, clean (S, A, 20, 2), mask (S, A), kind (S, A), man (S, A)."""
    rng = np.random.default_rng(seed)
    pos = np.zeros((n_scenes, A_MAX, T_ALL, 2), np.float32)
    clean = np.zeros_like(pos)
    mask = np.zeros((n_scenes, A_MAX), bool)
    kind = np.zeros((n_scenes, A_MAX), np.int64)
    man = np.full((n_scenes, A_MAX), -1, np.int64)
    for k in range(n_scenes):
        sc = make_scene(rng)
        n = len(sc["kind"])
        clean[k, :n] = sc["clean"]
        pos[k, :n] = sc["clean"] + rng.normal(0, NOISE, sc["clean"].shape)
        mask[k, :n] = True
        kind[k, :n] = sc["kind"]
        man[k, :n] = sc["man"]
    return dict(pos=pos, clean=clean, mask=mask, kind=kind, man=man)


# --------------------------------------------------------------------------------------------- agent frame
def agent_frame(pos: np.ndarray, clean: np.ndarray, mask: np.ndarray, kind: np.ndarray) -> dict:
    """World tracks -> agent-centric tensors.

    origin o_i = last observed position, heading psi_i from the last observed displacement.
        hist  (S, A, 8, 4)  observed [x, y, vx, vy] in the agent's own frame
        fut   (S, A, 12, 2) measured future positions (training target), fut_clean (S, A, 12, 2) noise-free truth
        v0    (S, A, 2)     last observed velocity (integration start)
        rel   (S, A, A, 7)  agent j seen from agent i: [dx, dy, cos dpsi, sin dpsi, vx_j, vy_j, kind_j]
        origin (S, A, 2), psi (S, A)  to map predictions back to the world
    """
    o = pos[:, :, T_OBS - 1]
    d1 = pos[:, :, T_OBS - 1] - pos[:, :, T_OBS - 3]
    d2 = pos[:, :, T_OBS - 1] - pos[:, :, 0]
    d = np.where((np.linalg.norm(d1, axis=-1) > 0.4)[..., None], d1, d2)
    psi = np.where(np.linalg.norm(d, axis=-1) > 0.4, np.arctan2(d[..., 1], d[..., 0]), 0.0)
    c, s = np.cos(psi), np.sin(psi)
    Rt = np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)          # (S, A, 2, 2) world -> local

    def to_local(x):                                                           # x (S, A, T, 2)
        return np.einsum("saij,satj->sati", Rt, x - o[:, :, None])

    loc = to_local(pos)
    loc_c = to_local(clean)
    hist_p = loc[:, :, :T_OBS]
    hist_v = np.diff(hist_p, axis=2, prepend=hist_p[:, :, :1]) / DT
    hist_v[:, :, 0] = hist_v[:, :, 1]
    v_world = (pos[:, :, T_OBS - 1] - pos[:, :, T_OBS - 2]) / DT
    v0 = np.einsum("saij,saj->sai", Rt, v_world)
    rel_p = np.einsum("saij,sabj->sabi", Rt, o[:, None, :, :] - o[:, :, None, :])
    rel_v = np.einsum("saij,sabj->sabi", Rt, np.broadcast_to(v_world[:, None], rel_p.shape))
    dpsi = psi[:, None, :] - psi[:, :, None]
    rel = np.concatenate([rel_p / 20.0, np.cos(dpsi)[..., None], np.sin(dpsi)[..., None], rel_v / 10.0,
                          np.broadcast_to(kind[:, None, :, None], rel_p.shape[:3] + (1,)).astype(np.float32)], -1)
    hist = np.concatenate([hist_p, hist_v], -1)
    hist[~mask] = 0
    return dict(hist=hist.astype(np.float32), fut=loc[:, :, T_OBS:].astype(np.float32),
                fut_clean=loc_c[:, :, T_OBS:].astype(np.float32), v0=v0.astype(np.float32),
                rel=rel.astype(np.float32), mask=mask, kind=kind, origin=o.astype(np.float32),
                psi=psi.astype(np.float32))


def local_to_world(p: np.ndarray, origin: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """p (..., T, 2) in an agent frame -> world, origin (2,), psi scalar."""
    c, s = np.cos(psi), np.sin(psi)
    R = np.array([[c, -s], [s, c]])
    return p @ R.T + origin


if __name__ == "__main__":
    import time
    t0 = time.time()
    d = make_dataset(500, 0)
    print("500 scenes in %.1fs" % (time.time() - t0))
    m = d["mask"]
    print("agents/scene %.2f, ped share %.2f" % (m.sum(1).mean(), (d["kind"][m] == PED).mean()))
    print("maneuver counts", np.bincount(d["man"][m], minlength=7))
    f = agent_frame(d["pos"], d["clean"], d["mask"], d["kind"])
    veh = m & (d["kind"] == VEH)
    print("vehicle future disp p50/p95 %.1f / %.1f m" % tuple(np.percentile(np.linalg.norm(f["fut_clean"][veh][:, -1], axis=-1), [50, 95])))
    acc = np.diff(d["clean"], 2, axis=2) / DT ** 2
    print("clean |a| p99 veh %.2f" % np.percentile(np.linalg.norm(acc[veh], axis=-1), 99))
