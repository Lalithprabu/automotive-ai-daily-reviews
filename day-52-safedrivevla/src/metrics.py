import numpy as np
from .synthetic import FOLLOW, ADAPT, REFUSE, executed_collision


def mode_accuracy(pred, true):
    return float((pred == true).mean())


def unsafe_recall(pred, true):
    """Of scenes where literal following is unsafe (ADAPT/REFUSE), fraction flagged as non-FOLLOW."""
    m = true != FOLLOW
    return float((pred[m] != FOLLOW).mean()) if m.any() else float("nan")


def false_refusal(pred, true):
    """Of safe scenes (FOLLOW), fraction wrongly refused/adapted (over-conservatism)."""
    m = true == FOLLOW
    return float((pred[m] != FOLLOW).mean()) if m.any() else float("nan")


def collision_rate(sc, profile, mask):
    """Collision rate of an executed lateral profile on the scenes selected by mask."""
    hit = executed_collision(sc, profile)
    return float(hit[mask].mean()) if mask.any() else float("nan")
