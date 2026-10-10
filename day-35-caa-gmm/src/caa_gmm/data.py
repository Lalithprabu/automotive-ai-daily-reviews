"""Synthetic junction-approach dataset for the CAA-GMM reconstruction (Day 35).

NOT the paper's data (nuScenes / Argoverse 2). It is built around the two things the CAA-GMM abstract
says its attention must handle:
  * map/context cues  -> a "route hint" raster channel marks the branch the agent will take 75% of the time
                         (so the junction outcome is genuinely multimodal: left / straight / right);
  * agent interaction -> a lead vehicle in the ego lane (IDM car-following) changes the ego speed profile,
                         while 0-3 side-lane distractor vehicles do not.
Everything is in the ego frame at the "present" step: ego at the origin, heading +y.
Raster: 64x64 px, 1 m/px, x in [-32, 32), y in [-16, 48); array index is [row=y, col=x].
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

DT = 0.25          # s per step
T_H = 8            # history steps (t = -7..0)
T_F = 16           # future steps (t = 1..16), 4 s
N_NB = 6           # neighbour slots
GRID = 64
X0, Y0 = -32.0, -16.0
R_TURN = 7.0       # turn radius (m)
LANE_W = 3.5
BRANCHES = ("left", "straight", "right")
J_MIN, J_MAX = 8, 18
HINT_RELIABILITY = 0.75
VEH_HALF_W, VEH_HALF_L = 1.0, 2.25


# ----------------------------------------------------------------------------- geometry
def path_xy(u: np.ndarray, J: float, branch: int) -> np.ndarray:
    """Ego-frame position at arc-length u along the lane for a given junction distance J and branch.
    u < J: straight lane x=0.  u >= J: left(0)/straight(1)/right(2) branch."""
    u = np.asarray(u, dtype=np.float64)
    x = np.zeros_like(u)
    y = u.copy()
    if branch == 1:
        return np.stack([x, y], -1)
    sgn = -1.0 if branch == 0 else 1.0
    t = np.maximum(u - J, 0.0)
    L = R_TURN * np.pi / 2
    phi = np.minimum(t, L) / R_TURN
    xa = R_TURN - R_TURN * np.cos(phi)
    ya = J + R_TURN * np.sin(phi)
    xb = R_TURN + np.maximum(t - L, 0.0)
    on_turn = u >= J
    xx = np.where(t <= L, xa, xb)
    yy = np.where(t <= L, ya, J + R_TURN)
    x = np.where(on_turn, sgn * xx, 0.0)
    y = np.where(on_turn, yy, u)
    return np.stack([x, y], -1)


def _pixel_centres():
    cols, rows = np.meshgrid(np.arange(GRID), np.arange(GRID))
    return X0 + cols + 0.5, Y0 + rows + 0.5     # (GRID,GRID) x, y


def _min_dist(px, py, pts):
    d = (px[..., None] - pts[:, 0]) ** 2 + (py[..., None] - pts[:, 1]) ** 2
    return np.sqrt(d.min(-1))


@lru_cache(maxsize=None)
def static_tables():
    """drivable[J_idx] (GRID,GRID) and hint[J_idx, branch] (GRID,GRID) for integer J in [10, 30]."""
    px, py = _pixel_centres()
    Js = list(range(4, 31))
    drv = np.zeros((len(Js), GRID, GRID), np.float32)
    hint = np.zeros((len(Js), 3, GRID, GRID), np.float32)
    for ji, J in enumerate(Js):
        mask = np.zeros((GRID, GRID), bool)
        u = np.arange(-16.0, J + 40.0, 0.5)
        for b in range(3):
            pts = path_xy(u, J, b)
            mask |= _min_dist(px, py, pts) < LANE_W / 2
            post = path_xy(np.arange(J + 1.0, J + 26.0, 0.5), J, b)
            hint[ji, b] = (_min_dist(px, py, post) < LANE_W / 2).astype(np.float32)
        for sx in (-LANE_W, LANE_W):                      # side lanes, stop before the junction
            mask |= (np.abs(px - sx) < LANE_W / 2) & (py < J - 6)
        drv[ji] = mask
    return torch.from_numpy(drv), torch.from_numpy(hint), Js


# ----------------------------------------------------------------------------- scene simulation
def _idm(v, gap, dv, v_des, a_max=2.0, b=2.5, T=1.2, s0=2.0):
    if gap is None:
        a = a_max * (1 - (v / v_des) ** 4)
    else:
        s_star = s0 + v * T + v * dv / (2 * np.sqrt(a_max * b))
        a = a_max * (1 - (v / v_des) ** 4 - (s_star / max(gap, 0.1)) ** 2)
    return float(np.clip(a, -6.0, a_max))


def rollout(rng: np.random.Generator, J: int, branch: int, hint: int, n_future: int = T_F,
            force_lead: bool | None = None) -> dict:
    """Simulate one episode. Time index i = 0..T_H-1+n_future with i = T_H-1 the 'present' of a dataset sample."""
    n_steps = T_H + n_future
    v = rng.uniform(5.0, 10.0)
    v_des = rng.uniform(7.0, 11.0)
    has_lead = (rng.random() < 0.7) if force_lead is None else force_lead
    g0 = rng.uniform(12.0, 40.0)
    v_l = rng.uniform(2.0, 9.0)
    n_dis = int(rng.integers(0, 4))
    dis = [dict(x=float(rng.choice([-LANE_W, LANE_W])), y0=float(rng.uniform(-14.0, J - 10.0)),
                v=float(rng.uniform(3.0, 10.0))) for _ in range(n_dis)]
    s = np.zeros(n_steps)
    vel = np.zeros(n_steps)
    lead_y = (g0 + v_l * DT * np.arange(n_steps)) if has_lead else None
    Jg = None                                   # junction position in the rollout's arc-length frame
    vel[0] = v
    turning = branch != 1
    for i in range(1, n_steps):
        if Jg is None and i - 1 == T_H - 1:     # present reached: fix junction J metres ahead of ego
            Jg = s[T_H - 1] + J
        pos, vv = s[i - 1], vel[i - 1]
        for k in range(5):                      # 5 sub-steps of 0.05 s
            gap = dv = None
            if has_lead and (not turning or Jg is None or pos < Jg + 2):
                gap = lead_y[i - 1] + v_l * 0.05 * k - pos - 4.5
                dv = vv - v_l
            vd = v_des
            if turning and Jg is not None and pos > Jg - 10:
                vd = min(v_des, 5.5)            # slow for the turn
            a = _idm(vv, gap, dv, vd)
            vv = max(0.0, vv + a * 0.05)
            pos += vv * 0.05
        s[i], vel[i] = pos, vv
    if Jg is None:
        Jg = s[T_H - 1] + J
    dis_y = [d["y0"] + s[T_H - 1] * 0 + d["v"] * DT * (np.arange(n_steps) - (T_H - 1)) for d in dis]
    return dict(s=s, vel=vel, J=J, Jg=Jg, branch=branch, hint=hint, lead_y=lead_y, v_l=v_l,
                has_lead=has_lead, dis=dis, dis_y=dis_y, lead_x=0.0)


def window(roll: dict, i_now: int) -> dict:
    """Extract a dataset sample whose 'present' is rollout index i_now (needs i_now>=T_H-1, i_now+T_F<len)."""
    s = roll["s"]
    hist_idx = np.arange(i_now - T_H + 1, i_now + 1)
    fut_idx = np.arange(i_now + 1, i_now + 1 + T_F)
    s_now = s[i_now]
    Jrel = roll["Jg"] - s_now
    # ego (straight section during history: x=0, only small lateral noise handled by caller)
    ego_h_pos = np.stack([np.zeros(T_H), s[hist_idx] - s_now], -1)
    # future positions along the branch (ego-frame, junction at Jrel)
    u_f = s[fut_idx] - s_now
    fut = path_xy(u_f, Jrel, roll["branch"])
    # neighbours
    nbs, is_lead = [], []
    if roll["has_lead"]:
        nbs.append(np.stack([np.full(T_H, roll["lead_x"]), roll["lead_y"][hist_idx] - s_now], -1))
        is_lead.append(True)
    for d, dy in zip(roll["dis"], roll["dis_y"]):
        # distractor y is stored relative to the *dataset* present (rollout index T_H-1)
        yy = dy[hist_idx] - (s_now - s[T_H - 1])
        nbs.append(np.stack([np.full(T_H, d["x"]), yy], -1))
        is_lead.append(False)
    return dict(ego_h=ego_h_pos, fut=fut, nbs=nbs, is_lead=is_lead, Jrel=float(Jrel))


def _vel_feats(pos: np.ndarray) -> np.ndarray:
    v = np.zeros_like(pos)
    v[1:] = (pos[1:] - pos[:-1]) / DT
    v[0] = v[1]
    return np.concatenate([pos, v], -1)         # (T_H, 4)


def sample_to_arrays(roll: dict, i_now: int, rng: np.random.Generator) -> dict:
    w = window(roll, i_now)
    ego_pos = w["ego_h"] + rng.normal(0, 0.03, w["ego_h"].shape) * np.array([1.0, 0.3])
    ego = _vel_feats(ego_pos)
    nb = np.zeros((N_NB, T_H, 4), np.float32)
    mask = np.zeros(N_NB, bool)
    perm = rng.permutation(N_NB)                 # random slot order: no positional shortcut to the lead
    lead_slot = -1
    for k, (p, il) in enumerate(zip(w["nbs"], w["is_lead"])):
        slot = int(perm[k])
        nb[slot] = _vel_feats(p + rng.normal(0, 0.03, p.shape))
        mask[slot] = True
        if il:
            lead_slot = slot
    return dict(ego=ego.astype(np.float32), nb=nb, mask=mask, fut=w["fut"].astype(np.float32),
                J=roll["J"] if i_now == T_H - 1 else int(round(np.clip(w["Jrel"], 4, 30))),
                hint=roll["hint"], branch=roll["branch"], lead_slot=lead_slot, has_lead=roll["has_lead"])


def make_dataset(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("ego", "nb", "mask", "fut", "J", "hint", "branch", "lead_slot", "has_lead")}
    for _ in range(n):
        J = int(rng.integers(J_MIN, J_MAX + 1))
        hint = int(rng.integers(0, 3))
        if rng.random() < HINT_RELIABILITY:
            branch = hint
        else:
            branch = int(rng.choice([b for b in range(3) if b != hint]))
        r = rollout(rng, J, branch, hint)
        a = sample_to_arrays(r, T_H - 1, rng)
        for k in out:
            out[k].append(a[k])
    t = {k: np.stack(v) for k, v in out.items()}
    return {
        "ego": torch.from_numpy(t["ego"]), "nb": torch.from_numpy(t["nb"]),
        "mask": torch.from_numpy(t["mask"]), "fut": torch.from_numpy(t["fut"]),
        "J": torch.from_numpy(t["J"].astype(np.int64)), "hint": torch.from_numpy(t["hint"].astype(np.int64)),
        "branch": torch.from_numpy(t["branch"].astype(np.int64)),
        "lead_slot": torch.from_numpy(t["lead_slot"].astype(np.int64)),
        "has_lead": torch.from_numpy(t["has_lead"].astype(bool)),
    }


# ----------------------------------------------------------------------------- batching / raster
def perturb_neighbours(batch: dict, drop_p: float = 0.0, pos_noise: float = 0.0,
                       gen: torch.Generator | None = None) -> dict:
    """Simulate imperfect V2X communication / perception: drop neighbours and add a per-vehicle position bias."""
    b = dict(batch)
    B, N = batch["mask"].shape
    keep = torch.rand(B, N, generator=gen) >= drop_p
    b["mask"] = batch["mask"] & keep
    if pos_noise > 0:
        bias = torch.randn(B, N, 1, 2, generator=gen) * pos_noise
        nb = batch["nb"].clone()
        nb[..., :2] = nb[..., :2] + bias
        b["nb"] = nb
    return b


def build_raster(batch: dict) -> torch.Tensor:
    """(B,4,64,64): drivable, route-hint, visible-neighbour occupancy (present step), ego footprint."""
    drv, hint, Js = static_tables()
    ji = (batch["J"].clamp(4, 30) - 4)
    r_drv = drv[ji]                                              # (B,G,G)
    r_hint = hint[ji, batch["hint"]]                             # (B,G,G)
    px, py = _pixel_centres()
    px = torch.from_numpy(px).float()
    py = torch.from_numpy(py).float()
    pos = batch["nb"][:, :, -1, :2]                              # (B,N,2) present position
    inside = ((px[None, None] - pos[..., 0, None, None]).abs() < VEH_HALF_W) & \
             ((py[None, None] - pos[..., 1, None, None]).abs() < VEH_HALF_L)
    occ = (inside & batch["mask"][..., None, None]).any(1).float()
    ego = ((px.abs() < VEH_HALF_W) & (py.abs() < VEH_HALF_L)).float().expand(r_drv.shape[0], -1, -1)
    return torch.stack([r_drv, r_hint, occ, ego], 1)


def take(data: dict, idx: torch.Tensor) -> dict:
    return {k: v[idx] for k, v in data.items()}
