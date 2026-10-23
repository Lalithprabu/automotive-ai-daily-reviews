"""Synthetic highway lane-change data + symbolic discretisation (knowledge-graph facts).

Maneuver hypotheses:  0 = LLC (left lane change), 1 = LK (lane keep), 2 = RLC (right lane change).
Sign convention: +y = left.

A lane change is modelled with a bell-shaped lateral-velocity profile
    vy(t) = d * VMAX * exp(-((t_c - t)/H0)^2)       (d = +1 left / -1 right, t_c = lane-line crossing time)
so `h = t_c - t` seconds before crossing the signal is *weak* (h=4 s -> 9% of peak) and gets
stronger as the crossing approaches. Lateral travel over the whole bell = VMAX*H0*sqrt(pi) = 3.5 m
(one lane width) -> VMAX = 0.76 m/s with H0 = 2.6 s. This lets the same profile drive both the
classifier dataset and the closed-loop cut-in simulation in `decision.py`.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

H0 = 2.6
LANE_W = 3.5
VMAX = LANE_W / (H0 * math.sqrt(math.pi))      # ~0.759 m/s
MANEUVERS = ["LLC", "LK", "RLC"]
LLC, LK, RLC = 0, 1, 2

# ---- relations (evidence types) ----------------------------------------------------------
RELATIONS = ["hasLatVel", "hasLatAcc", "hasTTC_front", "hasTTC_leftFront",
             "hasTTC_rightFront", "hasTTC_leftRear", "hasTTC_rightRear"]
N_REL = len(RELATIONS)
VY_LABELS = ["strongRight", "right", "straight", "left", "strongLeft"]
AY_LABELS = ["accRightHard", "accRight", "accStraight", "accLeft", "accLeftHard"]
RISK_LABELS = ["HIGH", "MEDIUM", "LOW"]           # TTC in [0,4]s / (4,10)s / else (precursor paper)
# entity vocabulary: 3 maneuvers, 5 vy, 5 ay, 3 risk
ENTITIES = MANEUVERS + VY_LABELS + AY_LABELS + RISK_LABELS
N_ENT = len(ENTITIES)
VY_OFF, AY_OFF, RISK_OFF = 3, 8, 13
# candidate value-entity ids per relation (used as the negative-sampling / likelihood support)
REL_VALUES = [list(range(VY_OFF, VY_OFF + 5)), list(range(AY_OFF, AY_OFF + 5))] + \
             [list(range(RISK_OFF, RISK_OFF + 3))] * 5
VY_EDGES = np.array([-0.45, -0.15, 0.15, 0.45])
AY_EDGES = np.array([-0.15, -0.06, 0.06, 0.15])


def ttc_risk_class(ttc: np.ndarray) -> np.ndarray:
    """0=HIGH [0,4], 1=MEDIUM (4,10), 2=LOW."""
    return np.where(ttc <= 4.0, 0, np.where(ttc < 10.0, 1, 2))


def discretise(feats: np.ndarray) -> np.ndarray:
    """(N,7) numeric [vy, ay, ttc_f, ttc_lf, ttc_rf, ttc_lr, ttc_rr] -> (N,7) entity ids."""
    f = np.atleast_2d(feats)
    out = np.empty(f.shape, dtype=np.int64)
    out[:, 0] = VY_OFF + np.digitize(f[:, 0], VY_EDGES)
    out[:, 1] = AY_OFF + np.digitize(f[:, 1], AY_EDGES)
    out[:, 2:] = RISK_OFF + ttc_risk_class(f[:, 2:])
    return out


def _sample_ttc(rng, n, p_high, p_med):
    u = rng.random(n)
    t = np.where(u < p_high, rng.uniform(0.8, 4.0, n),
                 np.where(u < p_high + p_med, rng.uniform(4.0, 10.0, n), rng.uniform(10.0, 30.0, n)))
    return t


def features_for(label: np.ndarray, horizon: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Numeric features observed `horizon` seconds before the lane-line crossing (LC) or at a random time (LK)."""
    n = len(label)
    d = np.where(label == LLC, 1.0, np.where(label == RLC, -1.0, 0.0))
    s = np.exp(-(horizon / H0) ** 2)
    vy = d * VMAX * s + rng.normal(0, 0.12, n)
    ay = d * VMAX * s * (2 * horizon / H0 ** 2) + rng.normal(0, 0.07, n)
    lc = d != 0
    ttc_f = np.where(lc, _sample_ttc(rng, n, 0.30, 0.40), _sample_ttc(rng, n, 0.05, 0.25))
    ttc = [ttc_f]
    # target-side neighbours are rarely high-risk right before a lane change; other side unconstrained
    for side, (is_front) in [("lf", True), ("rf", True), ("lr", False), ("rr", False)]:
        target_side = ((side[0] == "l") & (d > 0)) | ((side[0] == "r") & (d < 0))
        base = _sample_ttc(rng, n, 0.12, 0.28)
        safe = _sample_ttc(rng, n, 0.02, 0.10)
        ttc.append(np.where(target_side, safe, base))
    return np.stack([vy, ay] + ttc, axis=1).astype(np.float32)


@dataclass
class Split:
    feats: np.ndarray      # (N,7) float
    ents: np.ndarray       # (N,7) entity ids
    label: np.ndarray      # (N,)
    horizon: np.ndarray    # (N,)


def make_split(n: int, rng, horizons=(0.5, 4.5), fixed_h: float | None = None) -> Split:
    label = np.tile(np.array([LLC, LK, RLC]), n // 3 + 1)[:n]
    rng.shuffle(label)
    if fixed_h is None:
        h = rng.uniform(horizons[0], horizons[1], n)
    else:
        h = np.full(n, fixed_h)
    f = features_for(label, h, rng)
    return Split(f, discretise(f), label.astype(np.int64), h.astype(np.float32))


def cutter_features(h: float, direction: float, rng: np.random.Generator) -> np.ndarray:
    """Single observation vector for the simulated cut-in vehicle, `h` seconds before crossing
    (h may be negative after crossing: signal then follows the symmetric bell)."""
    label = np.array([LLC if direction > 0 else RLC])
    return features_for(label, np.array([abs(h)]), rng)[0]


def cutter_lateral_velocity(t: float, t_c: float) -> float:
    """Deterministic bell profile used by the simulator (magnitude, toward the ego lane)."""
    return VMAX * math.exp(-((t_c - t) / H0) ** 2)
