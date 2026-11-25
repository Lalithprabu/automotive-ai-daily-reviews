"""Synthetic forward-camera driving scenes + an analytic, PDMS-like trajectory scorer.

Stand-in for NAVSIM / WOD-E2E / Bench2Drive, the benchmarks used by arXiv 2611.27330 (Sun, Li, Shi, Zheng, Hoe, Bai, Mi,
"GGDriver: Glance at What Is Shared, Gaze at What Differs for Perception-Free Planning").  Nothing here comes from the
paper beyond the idea: one camera with known calibration, a set of candidate ego trajectories, and a rule-based score
of each trajectory obtained by simulating it against the (static) scene.

    camera      pinhole, 96 x 160 px, f = 110 px, principal point (80, 34), 1.6 m above a flat road, looking along +Z
    ego frame   X lateral (+ = right), Z forward, metres; ego starts at X = 0 (centre of its lane), Z = 0
    road        2-4 lanes of 3.5 m, dashed lane lines, solid edge lines, grass outside
    objects     static: stalled car (1.8 x 4.5 x 1.5 m), cone, road debris (0.8 x 0.6 x 0.35 m), barrel, pedestrian;
                hazards on the road at Z = 22-62 m, distractors off the road
    trajectory  (d, v): lateral target offset d [m] reached smoothly over 25 m of travel, speed moves from the ego speed v0
                to target v with |accel| <= 3 m/s^2 (brake <= 6 m/s^2); 16 waypoints over 4 s at 0.25 s
    score       NC * DAC * (0.75 * EP + 0.25 * C)      (a toy cousin of NAVSIM's PDMS)
                NC  no collision with any object box over 4 s           {0, 1}
                DAC ego box stays inside the road edges                 {0, 1}
                EP  progress / best progress among safe vocabulary anchors, clipped to [0, 1]
                C   comfort: 1 - 0.5 min(|d| / 7, 1) - 0.5 min(|v - v0| / 12, 1)

Scene types: 0 "small" = a cone / debris / barrel blocks the ego lane (no car in it); 1 "big" = a stalled car blocks the
ego lane; 2 "clear" = the ego lane is free (other lanes and verges may hold objects).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

H, W = 96, 160
F_PX, CX, CY, CAM_H = 110.0, 80.0, 34.0, 1.6
PATCH = 8
GH, GW = H // PATCH, W // PATCH                      # 12 x 20 patch grid
LANE = 3.5
DT, N_T = 0.25, 16                                   # 16 waypoints, 4 s
L_LAT = 25.0                                         # metres of travel to complete a lateral offset
EGO_W, EGO_REAR, EGO_FRONT = 1.9, 1.0, 3.6           # ego box around the waypoint (rear axle)
MAX_OBJ = 8
SCENE_TYPES = ["small", "big", "clear"]
# type: (width, length, height, BGR colour)
OBJ = {"car": (1.8, 4.5, 1.5, None), "cone": (0.45, 0.45, 0.75, (0, 120, 255)),
       "debris": (0.8, 0.6, 0.35, (40, 70, 110)), "barrel": (0.6, 0.6, 1.0, (30, 60, 220)),
       "ped": (0.5, 0.5, 1.75, None)}
OBJ_NAMES = list(OBJ)

D_VALUES = np.arange(-7.0, 7.01, 1.75)               # 9 lateral targets
V_VALUES = np.array([0.0, 4.0, 8.0, 12.0, 16.0])     # 5 target speeds
ANCHORS = np.array([(d, v) for d in D_VALUES for v in V_VALUES], np.float32)     # (45, 2)
REF_D, REF_V = 0.875, 2.0                            # refinement range: d +- 0.875 m, v +- 2 m/s


# ------------------------------------------------------------------------------------------------ trajectories (torch)
def traj_points(dv: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """dv (..., 2) [d, v_target], v0 broadcastable to dv[..., 0] -> waypoints (..., 16, 2) [X, Z] at t = 0.25 .. 4 s."""
    d, vt = dv[..., 0], dv[..., 1]
    v = v0.expand_as(vt) if torch.is_tensor(v0) else torch.full_like(vt, float(v0))
    zs, z = [], torch.zeros_like(vt)
    for _ in range(N_T):
        dvel = (vt - v).clamp(-6.0 * DT, 3.0 * DT)                  # brake 6 m/s^2, accelerate 3 m/s^2
        v_new = v + dvel
        z = z + 0.5 * (v + v_new) * DT
        v = v_new
        zs.append(z)
    Z = torch.stack(zs, -1)                                          # (..., 16)
    tau = (Z / L_LAT).clamp(0, 1)
    X = d[..., None] * (3 * tau ** 2 - 2 * tau ** 3)                 # smoothstep lateral motion
    return torch.stack([X, Z], -1)


def score_traj(pts: torch.Tensor, dv: torch.Tensor, v0: torch.Tensor, obj: torch.Tensor, valid: torch.Tensor,
               edges: torch.Tensor, zmax: torch.Tensor | None = None) -> dict:
    """Rule-based score.  pts (B, K, 16, 2), dv (B, K, 2), v0 (B,), obj (B, M, 4) [X, Z, w, l], valid (B, M) bool,
    edges (B, 2) [left X, right X], zmax (B,) best safe progress (None -> computed from this K).  Returns dict of (B, K)."""
    X, Z = pts[..., 0], pts[..., 1]                                  # (B, K, T)
    ox, oz, ow, ol = [obj[:, None, None, :, i] for i in range(4)]    # (B, 1, 1, M)
    hit = ((X[..., None] - ox).abs() < (EGO_W + ow) / 2) & (oz - ol / 2 < Z[..., None] + EGO_FRONT) \
        & (oz + ol / 2 > Z[..., None] - EGO_REAR) & valid[:, None, None, :]
    nc = (~hit.any(-1).any(-1)).float()                              # (B, K)
    dac = ((X - EGO_W / 2 > edges[:, None, None, 0]) & (X + EGO_W / 2 < edges[:, None, None, 1])).all(-1).float()
    prog = Z[..., -1]
    if zmax is None:
        zmax = (prog * nc * dac).amax(-1)
    ep = (prog / zmax.clamp(min=1e-3)[:, None]).clamp(0, 1)
    c = 1 - 0.5 * (dv[..., 0].abs() / 7).clamp(max=1) - 0.5 * ((dv[..., 1] - v0[:, None]).abs() / 12).clamp(max=1)
    return dict(score=nc * dac * (0.75 * ep + 0.25 * c), nc=nc, dac=dac, ep=ep, c=c)


def comfort(dv: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    return 1 - 0.5 * (dv[..., 0].abs() / 7).clamp(max=1) - 0.5 * ((dv[..., 1] - v0[:, None]).abs() / 12).clamp(max=1)


def scene_targets(v0, obj, valid, edges, chunk: int = 256) -> dict:
    """Per scene: zmax over the 45 anchors, the score of every anchor, and the best (d, v) in each anchor's 5 x 5
    refinement grid (the regression target for the refinement head)."""
    A = torch.as_tensor(ANCHORS)
    gd, gv = torch.meshgrid(torch.linspace(-REF_D, REF_D, 5), torch.linspace(-REF_V, REF_V, 5), indexing="ij")
    grid = torch.stack([gd.flatten(), gv.flatten()], -1)            # (25, 2)
    out = dict(zmax=[], anchor_score=[], best_delta=[], best_score=[])
    for s in range(0, len(v0), chunk):
        sl = slice(s, s + chunk)
        b = v0[sl].shape[0]
        dvA = A[None].expand(b, -1, -1)
        sa = score_traj(traj_points(dvA, v0[sl, None]), dvA, v0[sl], obj[sl], valid[sl], edges[sl])
        zmax = (traj_points(dvA, v0[sl, None])[..., -1, 1] * sa["nc"] * sa["dac"]).amax(-1)
        cand = (A[:, None] + grid[None]).reshape(-1, 2)              # (45 * 25, 2)
        cand[:, 1] = cand[:, 1].clamp(min=0)
        dvC = cand[None].expand(b, -1, -1)
        sc = score_traj(traj_points(dvC, v0[sl, None]), dvC, v0[sl], obj[sl], valid[sl], edges[sl], zmax)["score"]
        sc = sc.view(b, 45, 25) - 1e-4 * (grid.abs() / torch.tensor([REF_D, REF_V])).sum(-1)   # tie-break: small moves
        best = sc.argmax(-1)
        out["zmax"].append(zmax)
        out["anchor_score"].append(sa["score"])
        out["best_delta"].append(grid[best])
        out["best_score"].append(sc.gather(-1, best[..., None])[..., 0] + 1e-4 * (grid[best].abs()
                                 / torch.tensor([REF_D, REF_V])).sum(-1))
    return {k: torch.cat(v) for k, v in out.items()}


# ------------------------------------------------------------------------------------------------ scenes (numpy)
def make_scene(rng: np.random.Generator) -> dict:
    n = int(rng.integers(2, 5))
    e = int(rng.integers(0, n))
    edges = np.array([(-e - 0.5) * LANE, (n - 1 - e + 0.5) * LANE], np.float32)
    v0 = float(rng.uniform(4, 14))
    st = int(rng.choice(3, p=[0.4, 0.3, 0.3]))
    objs = []

    def lane_x(i):
        return (i - e) * LANE + rng.normal(0, 0.45)

    if st == 0:
        objs.append(("cone" if rng.random() < 0.35 else ("debris" if rng.random() < 0.6 else "barrel"),
                     lane_x(e), rng.uniform(22, 62)))
    elif st == 1:
        objs.append(("car", lane_x(e), rng.uniform(24, 62)))
    for _ in range(int(rng.integers(0, 4))):                      # hazards in other lanes
        others = [i for i in range(n) if i != e]
        if not others:
            break
        t = str(rng.choice(OBJ_NAMES, p=[0.35, 0.2, 0.25, 0.1, 0.1]))
        objs.append((t, lane_x(int(rng.choice(others))), rng.uniform(14, 66)))
    if st == 2 and rng.random() < 0.3:                            # far object in ego lane, beyond reach at low speed
        objs.append((str(rng.choice(["car", "debris", "cone"])), lane_x(e), rng.uniform(55, 75)))
    for _ in range(int(rng.integers(1, 4))):                      # off-road distractors
        side = rng.choice([0, 1])
        x = edges[0] - rng.uniform(1.2, 6) if side == 0 else edges[1] + rng.uniform(1.2, 6)
        objs.append((str(rng.choice(OBJ_NAMES, p=[0.35, 0.15, 0.15, 0.1, 0.25])), x, rng.uniform(8, 60)))
    objs = objs[:MAX_OBJ]
    arr = np.zeros((MAX_OBJ, 4), np.float32)
    typ = np.full(MAX_OBJ, -1, np.int64)
    for i, (t, x, z) in enumerate(objs):
        w, l, h, _ = OBJ[t]
        arr[i] = (x, z, w, l)
        typ[i] = OBJ_NAMES.index(t)
    cols = rng.integers(40, 230, (MAX_OBJ, 3)).astype(np.uint8)
    return dict(n=n, e=e, edges=edges, v0=np.float32(v0), stype=st, obj=arr, typ=typ, cols=cols,
                light=np.float32(rng.uniform(0.7, 1.15)), dash=np.float32(rng.uniform(0, 8)))


def render(sc: dict, ego_x: float = 0.0, ego_z: float = 0.0, rng: np.random.Generator | None = None,
           ss: int = 2) -> np.ndarray:
    """Scene -> BGR uint8 (96, 160, 3).  ego_x / ego_z move the camera (used by the simulation's drive-out)."""
    h, w, f, cx, cy = H * ss, W * ss, F_PX * ss, CX * ss, CY * ss
    v = np.arange(h)[:, None] + 0.5
    u = np.arange(w)[None, :] + 0.5
    ground = np.broadcast_to(v > cy + 0.5, (h, w))
    zg = np.broadcast_to(np.where(v > cy + 0.5, f * CAM_H / np.maximum(v - cy, 0.5), np.inf), (h, w))
    xg = np.where(ground, (u - cx) * np.where(ground, zg, 0) / f, 0) + ego_x
    zw = np.where(ground, zg, 0) + ego_z
    L, R = sc["edges"]
    img = np.zeros((h, w, 3), np.float32)
    img[:] = (235, 200, 160)
    grass = ground & ((xg < L - 0.4) | (xg > R + 0.4))
    road = ground & ~grass
    img[grass] = (60, 120, 70)
    img[road] = (95, 95, 98)
    lw = 0.12
    edge = road & ((np.abs(xg - L) < lw) | (np.abs(xg - R) < lw))
    dash = np.mod(zw + sc["dash"], 8.0) < 4.0
    lanes = np.zeros_like(road)
    for i in range(1, sc["n"]):
        lanes |= np.abs(xg - (L + i * LANE)) < lw
    img[edge | (road & lanes & dash)] = (235, 235, 235)
    depth = np.array(zg, np.float64)
    order = np.argsort(-sc["obj"][:, 1])
    for i in order:
        if sc["typ"][i] < 0:
            continue
        x, z, ow, ol = sc["obj"][i]
        x, z = x - ego_x, z - ego_z - ol / 2                         # near face
        if z < 0.8:
            continue
        name = OBJ_NAMES[sc["typ"][i]]
        oh = OBJ[name][2]
        u0, u1 = cx + f * (x - ow / 2) / z, cx + f * (x + ow / 2) / z
        vb, vt = cy + f * CAM_H / z, cy + f * (CAM_H - oh) / z
        a0, a1, b0, b1 = int(max(0, np.floor(u0))), int(min(w, np.ceil(u1))), int(max(0, np.floor(vt))), \
            int(min(h, np.ceil(vb)))
        if a1 <= a0 or b1 <= b0:
            continue
        m = depth[b0:b1, a0:a1] > z
        col = OBJ[name][3]
        if name == "car":
            col = tuple(int(c) for c in sc["cols"][i])
        elif name == "ped":
            col = (int(sc["cols"][i][0]) // 2, int(sc["cols"][i][1]) // 2, 150)
        patch = img[b0:b1, a0:a1]
        patch[m] = col
        if name == "car":                                            # rear window + tail lights
            hh = b1 - b0
            ww = a1 - a0
            win = np.zeros_like(m)
            win[: max(1, hh // 3)] = True
            patch[m & win] = (60, 50, 40)
            tl = np.zeros_like(m)
            tl[hh // 2: hh // 2 + max(1, hh // 6), : max(1, ww // 6)] = True
            tl[hh // 2: hh // 2 + max(1, hh // 6), ww - max(1, ww // 6):] = True
            patch[m & tl] = (30, 30, 220)
        if name == "cone":                                           # white stripe
            hh = b1 - b0
            st = np.zeros_like(m)
            st[hh // 3: hh // 3 + max(1, hh // 5)] = True
            patch[m & st] = (240, 240, 240)
        depth[b0:b1, a0:a1][m] = z
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) if ss > 1 else img
    img = img * sc["light"]
    if rng is not None:
        img = img + rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_dataset(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    scs = [make_scene(rng) for _ in range(n)]
    imgs = np.stack([render(s, rng=rng) for s in scs])
    d = dict(img=imgs, v0=np.array([s["v0"] for s in scs], np.float32),
             obj=np.stack([s["obj"] for s in scs]), typ=np.stack([s["typ"] for s in scs]),
             edges=np.stack([s["edges"] for s in scs]), stype=np.array([s["stype"] for s in scs]),
             n=np.array([s["n"] for s in scs]), e=np.array([s["e"] for s in scs]),
             cols=np.stack([s["cols"] for s in scs]), light=np.array([s["light"] for s in scs], np.float32),
             dash=np.array([s["dash"] for s in scs], np.float32))
    t = scene_targets(torch.from_numpy(d["v0"]), torch.from_numpy(d["obj"]), torch.from_numpy(d["typ"] >= 0),
                      torch.from_numpy(d["edges"]))
    d.update({k: v.numpy() for k, v in t.items()})
    return d


def scene_from(d: dict, i: int) -> dict:
    return dict(n=int(d["n"][i]), e=int(d["e"][i]), edges=d["edges"][i], v0=d["v0"][i], stype=int(d["stype"][i]),
                obj=d["obj"][i], typ=d["typ"][i], cols=d["cols"][i], light=d["light"][i], dash=d["dash"][i])


if __name__ == "__main__":
    import time
    t0 = time.time()
    d = make_dataset(300, 0)
    print(f"300 scenes in {time.time() - t0:.1f}s")
    best_anchor = d["anchor_score"].max(1)
    for st, nm in enumerate(SCENE_TYPES):
        m = d["stype"] == st
        print(f"{nm:6s} n={m.sum():4d}  oracle anchor {best_anchor[m].mean():.3f}  oracle refined "
              f"{d['best_score'][m].max(1).mean():.3f}")
    cv2.imwrite("scene_grid.png", cv2.resize(np.concatenate([np.concatenate(list(d["img"][r * 4:(r + 1) * 4]), 1)
                                                                      for r in range(4)], 0), None, fx=2, fy=2,
                                                       interpolation=cv2.INTER_NEAREST))
