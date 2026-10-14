"""Dynamic Object Primitives (DOPs) + latency-compensated Bayesian fusion.

Mirrors the paper's fusion pipeline (spatial alignment -> velocity latency
compensation -> Mahalanobis Hungarian association with chi2_{3,0.95}=7.815 gating
-> additive information-filter update). Layout of the packed DOP is 35 bytes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import struct
import numpy as np
from scipy.optimize import linear_sum_assignment

CHI2_3_95 = 7.815
DOP_BYTES = 35
_FMT = "<3f3e1e3e9B"   # mu 12B | v 6B | theta 2B | dims 6B | 9B quantised (R-diag etc.)
assert struct.calcsize(_FMT) == DOP_BYTES


@dataclass
class DOP:
    mu: np.ndarray            # [3] position (m)
    vel: np.ndarray           # [3] velocity (m/s)
    pos_var: np.ndarray       # [3] positional variance (diag)
    vel_var: np.ndarray       # [3] velocity variance (diag)
    p_dyn: float = 1.0
    agent: int = 0
    theta: float = 0.0
    dims: tuple = (1.8, 4.5, 1.6)

    def copy(self) -> "DOP":
        return DOP(self.mu.copy(), self.vel.copy(), self.pos_var.copy(), self.vel_var.copy(),
                   self.p_dyn, self.agent, self.theta, self.dims)


def _q(sig: float, lo=-4.0, hi=4.0) -> int:   # log-sigma -> uint8
    return int(np.clip((np.log(max(sig, 1e-4)) - lo) / (hi - lo) * 255, 0, 255))


def _dq(b: int, lo=-4.0, hi=4.0) -> float:
    return float(np.exp(lo + b / 255.0 * (hi - lo)))


def pack_dop(d: DOP) -> bytes:
    """9 quantised bytes = pos sigma x3, vel sigma x3, p_dyn, agent id, reserved (our layout)."""
    q = [_q(np.sqrt(v)) for v in d.pos_var] + [_q(np.sqrt(v)) for v in d.vel_var] + \
        [int(np.clip(d.p_dyn, 0, 1) * 255), d.agent & 0xFF, 0]
    return struct.pack(_FMT, *d.mu.astype(np.float32), *d.vel.astype(np.float16),
                       np.float16(d.theta), *np.float16(d.dims), *q)


def unpack_dop(b: bytes) -> DOP:
    t = struct.unpack(_FMT, b)
    q = t[10:]
    return DOP(np.array(t[0:3], dtype=np.float64), np.array(t[3:6], dtype=np.float64),
               np.array([_dq(q[0]) ** 2, _dq(q[1]) ** 2, _dq(q[2]) ** 2]),
               np.array([_dq(q[3]) ** 2, _dq(q[4]) ** 2, _dq(q[5]) ** 2]),
               q[6] / 255.0, q[7], float(t[6]), tuple(float(x) for x in t[7:10]))


def compensate_latency(d: DOP, delta: float, q_acc: float = 1.0) -> DOP:
    """Propagate a DOP forward by `delta` seconds (constant-velocity, paper Eq. 21 analogue).
    mu' = mu + v*delta ;  var' = var + var_v*delta^2 + q_acc*delta^2  (velocity + accel uncertainty)."""
    o = d.copy()
    if delta <= 0:
        return o
    v = d.vel * (1.0 if d.p_dyn >= 0.5 else 0.0)        # static objects don't move
    o.mu = d.mu + v * delta
    o.pos_var = d.pos_var + d.vel_var * delta ** 2 + q_acc * delta ** 2 * (d.p_dyn >= 0.5)
    return o


def mahalanobis2(a: DOP, b: DOP) -> float:
    r = a.mu - b.mu
    return float(np.sum(r * r / (a.pos_var + b.pos_var)))


def associate(tracks: list[DOP], dets: list[DOP], gate: float = CHI2_3_95):
    """Hungarian on Mahalanobis^2 cost; pairs above the chi2 gate are rejected.
    Returns (matches [(track_i, det_j)], unmatched_dets [j])."""
    if not tracks or not dets:
        return [], list(range(len(dets)))
    C = np.array([[mahalanobis2(t, d) for d in dets] for t in tracks])
    big = 1e6
    r, c = linear_sum_assignment(np.where(C <= gate, C, big))
    matches = [(i, j) for i, j in zip(r, c) if C[i, j] <= gate]
    used = {j for _, j in matches}
    return matches, [j for j in range(len(dets)) if j not in used]


def information_fuse(a: DOP, b: DOP) -> DOP:
    """Additive information-filter update (paper Eqs. 23-25): Lambda = sum Lambda_i ; eta = sum Lambda_i x_i."""
    out = a.copy()
    for attr_m, attr_v in (("mu", "pos_var"), ("vel", "vel_var")):
        la, lb = 1.0 / getattr(a, attr_v), 1.0 / getattr(b, attr_v)
        lam = la + lb
        setattr(out, attr_m, (la * getattr(a, attr_m) + lb * getattr(b, attr_m)) / lam)
        setattr(out, attr_v, 1.0 / lam)
    out.p_dyn = 0.5 * (a.p_dyn + b.p_dyn)
    return out


def fuse_agents(per_agent: list[list[DOP]], deltas: list[float], *, compensate=True,
                pose_sigma=0.0, gate=CHI2_3_95, bayesian=True) -> list[DOP]:
    """Fuse DOP lists (agent 0 = ego). deltas[k] = age (s) of agent k's message.
    compensate=False -> 'old way' (use stale positions as-is).
    bayesian=False   -> 'old way' (late fusion: plain average of matched detections)."""
    tracks: list[DOP] = []
    for k, dops in enumerate(per_agent):
        dets = []
        for d in dops:
            x = compensate_latency(d, deltas[k]) if compensate else d.copy()
            if k > 0 and pose_sigma > 0:
                x.pos_var = x.pos_var + pose_sigma ** 2           # relative-pose uncertainty
            dets.append(x)
        matches, new = associate(tracks, dets, gate)
        for ti, dj in matches:
            if bayesian:
                tracks[ti] = information_fuse(tracks[ti], dets[dj])
            else:
                t, d = tracks[ti], dets[dj]
                t.mu = 0.5 * (t.mu + d.mu); t.vel = 0.5 * (t.vel + d.vel)
        tracks.extend(dets[j] for j in new)
    return tracks
