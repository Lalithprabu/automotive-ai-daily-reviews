"""Synthetic street scenes seen from a moving ego car, plus Cartesian ground truth (my construction).

The abstract of arXiv 2611.30078 (RadarOSV) evaluates on K-Radar and BRIDGE-Liv. Neither is used here: this module
builds toy scenes so the paper's mechanism (two radars with complementary velocity ranges + polar->Cartesian voxels +
radar-estimated ego motion) can be tested end to end on a CPU.

Ego frame: the radar sits at the origin, the ego drives along +x at speed v_ego. Bird's-eye grid (G x G = 64 x 64,
0.4 m cells) covers x in [0, 25.6) m and y in [-12.8, 12.8) m. Index convention: grid[i, j] -> x = (i + .5) * 0.4,
y = -12.8 + (j + .5) * 0.4.

Classes: 0 free, 1 static, 2 car, 3 truck, 4 cyclist, 5 pedestrian (the four dynamic classes are 2..5).

Velocity ground truth per occupied cell:
    v_gr   ground-relative radial velocity  = v_obj . u          (u = unit vector from radar to the cell)
    v_rel  radar-relative radial velocity   = (v_obj - v_ego e_x) . u = v_gr - v_ego cos(theta)
A static wall straight ahead therefore shows v_rel = -v_ego, which aliases in a radar whose unambiguous range is small.
"""
from __future__ import annotations

import numpy as np

G, CELL, X0, Y0 = 64, 0.4, 0.0, -12.8
R_MAX, FOV = 25.6, np.deg2rad(60.0)
CLASSES = ["free", "static", "car", "truck", "cyclist", "pedestrian"]
DYN = (2, 3, 4, 5)
SIZE = {2: (4.5, 1.9), 3: (8.0, 2.5), 4: (1.8, 0.7), 5: (0.6, 0.6)}           # length, width (m)
SPEED = {2: (0.0, 12.0), 3: (0.0, 10.0), 4: (2.0, 7.0), 5: (0.3, 2.0)}        # ground speed range (m/s)
P_CLS = {2: 0.35, 3: 0.15, 4: 0.25, 5: 0.25}
V_EGO_MAX = 8.0

# cell centres
XC = X0 + (np.arange(G) + 0.5) * CELL
YC = Y0 + (np.arange(G) + 0.5) * CELL
GX, GY = np.meshgrid(XC, YC, indexing="ij")                                   # (G, G)
GR, GTH = np.hypot(GX, GY), np.arctan2(GY, GX)
FOV_MASK = (np.abs(GTH) <= FOV) & (GR < R_MAX) & (GR >= 1.0)                  # cells either radar can see


def _box(cls, cx, cy, yaw, vx=0.0, vy=0.0, L=None, W=None):
    l, w = SIZE.get(cls, (L, W)) if L is None else (L, W)
    return dict(cls=cls, cx=cx, cy=cy, yaw=yaw, L=l, W=w, vx=vx, vy=vy)


def make_scene(rng: np.random.Generator, v_ego: float | None = None) -> dict:
    """Random street: two walls with gaps, parked cars, poles, and 2-6 moving road users."""
    v_ego = rng.uniform(0, V_EGO_MAX) if v_ego is None else v_ego
    objs = []
    w_l, w_r = rng.uniform(6, 11.5), rng.uniform(6, 11.5)
    for side, w in ((1, w_l), (-1, w_r)):                                     # walls as segments with gaps
        x = -20.0
        while x < 70:
            seg = rng.uniform(4, 14)
            objs.append(_box(1, x + seg / 2, side * (w + 0.2), 0.0, L=seg, W=0.4))
            x += seg + rng.uniform(0.5, 4)
        for _ in range(rng.integers(0, 3)):                                   # parked cars along the wall
            objs.append(_box(1, rng.uniform(3, 30), side * (w - 1.3), 0.0, L=4.4, W=1.8))
    for _ in range(rng.integers(0, 5)):                                       # poles / bollards
        objs.append(_box(1, rng.uniform(2, 26), rng.choice([-1, 1]) * rng.uniform(4, 10), 0.0, L=0.35, W=0.35))
    n = rng.integers(2, 7)
    cls_p = np.array(list(P_CLS.values()))
    placed = 0
    for _ in range(60):
        if placed >= n:
            break
        c = int(rng.choice(list(P_CLS), p=cls_p))
        cx, cy = rng.uniform(2.5, 25.0), rng.uniform(-min(w_r, 11) + 1.5, min(w_l, 11) - 1.5)
        if any(o["cls"] in DYN and np.hypot(o["cx"] - cx, o["cy"] - cy) < 3.5 for o in objs):
            continue
        if c == 5:
            yaw = rng.uniform(-np.pi, np.pi)
        elif rng.random() < 0.2:                                              # crossing traffic
            yaw = rng.choice([-1, 1]) * np.pi / 2 + rng.normal(0, 0.1)
        else:
            yaw = rng.choice([0.0, np.pi]) + rng.normal(0, 0.08)
        sp = rng.uniform(*SPEED[c])
        objs.append(_box(c, cx, cy, yaw, sp * np.cos(yaw), sp * np.sin(yaw)))
        placed += 1
    return dict(v_ego=float(v_ego), objs=objs)


def step_scene(scene: dict, dt: float, rng: np.random.Generator) -> dict:
    """Advance by dt in the ego frame (ego moves +x, so the world shifts by -v_ego dt). Respawns road users that leave."""
    out = dict(v_ego=scene["v_ego"], objs=[])
    for o in scene["objs"]:
        o = dict(o)
        o["cx"] += (o["vx"] - scene["v_ego"]) * dt
        o["cy"] += o["vy"] * dt
        if o["cls"] == 1 and o["cx"] + o["L"] / 2 < -20:                      # recycle static clutter ahead
            o["cx"] += 90.0
        if o["cls"] in DYN and not (-6 < o["cx"] < 34 and abs(o["cy"]) < 14):
            fresh = make_scene(rng, scene["v_ego"])["objs"]
            dyn = [f for f in fresh if f["cls"] == o["cls"]] or [f for f in fresh if f["cls"] in DYN]
            o = dyn[0]
            o["cx"] = 26.0 if scene["v_ego"] > 2 else o["cx"]
        out["objs"].append(o)
    return out


def box_corners(o: dict) -> np.ndarray:
    c, s = np.cos(o["yaw"]), np.sin(o["yaw"])
    dx = np.array([1, 1, -1, -1]) * o["L"] / 2
    dy = np.array([1, -1, -1, 1]) * o["W"] / 2
    return np.stack([o["cx"] + c * dx - s * dy, o["cy"] + s * dx + c * dy], -1)  # (4, 2)


def rasterize(scene: dict) -> dict:
    """Cartesian ground truth with 2x2 supersampling. Dynamic objects overwrite static ones."""
    sem = np.zeros((G, G), np.uint8)
    vgr = np.zeros((G, G), np.float32)
    sub = np.array([-0.25, 0.25]) * CELL
    for o in sorted(scene["objs"], key=lambda o: o["cls"] in DYN):
        r = max(o["L"], o["W"]) / 2 + CELL
        i0, i1 = np.clip([(o["cx"] - r - X0) / CELL, (o["cx"] + r - X0) / CELL + 1], 0, G).astype(int)
        j0, j1 = np.clip([(o["cy"] - r - Y0) / CELL, (o["cy"] + r - Y0) / CELL + 1], 0, G).astype(int)
        if i0 >= i1 or j0 >= j1:
            continue
        px = GX[i0:i1, j0:j1, None, None] + sub[:, None]
        py = GY[i0:i1, j0:j1, None, None] + sub[None, :]
        c, s = np.cos(o["yaw"]), np.sin(o["yaw"])
        lx = (px - o["cx"]) * c + (py - o["cy"]) * s
        ly = -(px - o["cx"]) * s + (py - o["cy"]) * c
        hit = ((np.abs(lx) <= o["L"] / 2) & (np.abs(ly) <= o["W"] / 2)).any((-1, -2))
        sem[i0:i1, j0:j1][hit] = o["cls"]
        ux, uy = np.cos(GTH[i0:i1, j0:j1]), np.sin(GTH[i0:i1, j0:j1])
        vgr[i0:i1, j0:j1][hit] = (o["vx"] * ux + o["vy"] * uy)[hit]
    vrel = (vgr - scene["v_ego"] * np.cos(GTH)).astype(np.float32)
    vrel[sem == 0] = 0.0
    return dict(sem=sem, vgr=vgr, vrel=vrel)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    s = make_scene(rng)
    gt = rasterize(s)
    print("v_ego", round(s["v_ego"], 2), "objects", len(s["objs"]),
          "cells per class", np.bincount(gt["sem"][FOV_MASK], minlength=6))
