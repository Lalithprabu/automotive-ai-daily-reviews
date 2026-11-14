"""Query-indexed FUTURE TOKENS (teacher side, built from the logged future).

Paper (arXiv 2611.14286): Q = Q_a U Q_r U Q_o U Q_m, each query i -> z_i = (e_i, c_i, r_i)
    e_i : continuous descriptor   c_i : validity / confidence label   r_i : boundary-risk label
Fixed query identities make teacher/student matching pointwise; missing entries are null-padded.

Toy instantiation (my sizes; the paper does not state N_a, N_r, N_o, N_m or dims in the parts I read):
  N_a = 4 actor queries   - actors ranked by PRESENT relevance to the ego corridor (deployable ranking)
        e = [long. deviation from constant-velocity at 4 s /10, lateral approach to ego path /3.5,
             approach direction {-1,0,1}, first ego-corridor entry time /4 (1 = never)]
        c = slot holds a real actor;  r = actor enters the ego corridor, or brakes >= 3 m off its CV path
  N_r = 8 route anchors   - points on the ego lane at s = 5, 10, ..., 40 m
        e = [fraction of horizon occupied, first-occupied time /4, cleared flag, contested-by-pedestrian]
        c = 1;  r = becomes occupied in the future while free now
  N_o = 8 occupancy cells - {ego lane, left lane} x {0-10, 10-20, 20-30, 30-45 m}
        e = [occ(0 s), occ(2 s), occ(4 s), occ(4 s) - occ(0 s)]
        c = 1;  r = becomes occupied while empty now
  N_m = K margin queries  - one per BASE candidate (needs the frozen generator, so built in torch, see losses.py)

Injected token vector (what the adapter sees):  v_i = c_i * [e_i, r_i]   (5 dims).
An empty/quiet query therefore injects exactly zero, which is what makes the zero-token path a no-op.
"""
from __future__ import annotations

import numpy as np

from . import world as W

N_A, N_R, N_O = 4, 8, 8
N_SCENE_Q = N_A + N_R + N_O
E_DIM = 4
ROUTE_S = np.arange(5.0, 41.0, 5.0)
OCC_X = np.array([0.0, 10.0, 20.0, 30.0, 45.0])
OCC_LANES = np.array([0.0, W.LEFT_Y])
CORR_HALF = W.LANE_W / 2


def rank_actors(hist: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Present-only relevance ranking -> (N_A,) actor indices, -1 = empty slot."""
    x, y = hist[:, -1, 0], hist[:, -1, 1]
    d = np.maximum(np.abs(y) - CORR_HALF, 0) + 0.15 * np.maximum(x, 0) + 10.0 * (x < -5)
    d = np.where(valid, d, np.inf)
    order = np.argsort(d)[:N_A]
    return np.where(np.isfinite(d[order]), order, -1)


def in_corridor(xy: np.ndarray, r: np.ndarray) -> np.ndarray:
    """xy (..., 2) -> bool, inside the ego lane corridor ahead of the ego."""
    return (np.abs(xy[..., 1]) < CORR_HALF + r) & (xy[..., 0] > 0) & (xy[..., 0] < 45)


def scene_tokens(sc: dict, rank: np.ndarray):
    """Teacher tokens for actor + route + occupancy queries.  Returns E (20, 4), C (20,), R (20,)."""
    E = np.zeros((N_SCENE_Q, E_DIM), np.float32)
    C = np.zeros(N_SCENE_Q, np.float32)
    R = np.zeros(N_SCENE_Q, np.float32)
    hist, fut, rad, typ, val = sc["hist"], sc["fut"], sc["radius"], sc["typ"], sc["valid"]
    cur = hist[:, -1, :2]
    full = np.concatenate([cur[:, None], fut], 1)                         # (N, H+1, 2) incl. present

    # ---- actor queries
    for i, j in enumerate(rank):
        if j < 0:
            continue
        cv4 = cur[j] + hist[j, -1, 2:] * W.T_FUT[-1]
        dev = (fut[j, -1, 0] - cv4[0]) / 10.0
        dl = (abs(cur[j, 1]) - abs(fut[j, -1, 1])) / W.LANE_W
        app = float(np.sign(dl)) if abs(dl) > 0.1 else 0.0
        inc = in_corridor(full[j], rad[j])
        t_cross = 0.0 if inc[0] else (W.DT * np.argmax(inc) / 4.0 if inc.any() else 1.0)
        enters = (not inc[0]) and inc[1:].any()
        brakes = inc[0] and (cv4[0] - fut[j, -1, 0] > 3.0)
        E[i] = [dev, np.clip(dl, -1.5, 1.5), app, t_cross]
        C[i] = 1.0
        R[i] = float(enters or brakes)

    # ---- route anchors on the ego lane
    for q, s in enumerate(ROUTE_S):
        occ = ((np.abs(full[..., 0] - s) < 2.5 + rad[:, None]) & (np.abs(full[..., 1]) < CORR_HALF + rad[:, None])
               & val[:, None])                                            # (N, H+1)
        o_t = occ.any(0)                                                  # (H+1,)
        ped = (occ & (typ == W.TYPE_PED)[:, None]).any()
        first = W.DT * np.argmax(o_t[1:]) / 4.0 + 0.125 if o_t[1:].any() else 1.0
        E[N_A + q] = [o_t[1:].mean(), first, float(o_t[0] and not o_t[-1]), float(ped)]
        C[N_A + q] = 1.0
        R[N_A + q] = float((not o_t[0]) and o_t[1:].any())

    # ---- occupancy cells
    q = N_A + N_R
    for lane in OCC_LANES:
        for b in range(len(OCC_X) - 1):
            inside = ((np.abs(full[..., 1] - lane) < CORR_HALF) & (full[..., 0] >= OCC_X[b])
                      & (full[..., 0] < OCC_X[b + 1]) & val[:, None])     # (N, H+1)
            o = np.clip(inside.sum(0), 0, 1).astype(float)
            E[q] = [o[0], o[4], o[8], o[8] - o[0]]
            C[q] = 1.0
            R[q] = float(o[0] == 0 and o[1:].max() > 0)
            q += 1
    return E, C, R
