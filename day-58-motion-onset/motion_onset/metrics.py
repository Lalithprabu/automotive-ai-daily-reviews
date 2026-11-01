"""Brier skill score (vs climatology), average precision (PR-AUC), ECE with equal-mass bins."""
import numpy as np


def brier(p, y): return float(np.mean((p - y) ** 2))


def bss(p, y, ref): return 1.0 - brier(p, y) / brier(np.full_like(p, ref), y)        # >0 beats a constant climatological rate


def pr_auc(p, y):
    o = np.argsort(-p, kind="stable"); y = y[o]
    if y.sum() == 0: return float("nan")
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())                                         # average precision


def ece(p, y, bins=20):
    o = np.argsort(p, kind="stable"); p, y = p[o], y[o]; e = 0.0
    for idx in np.array_split(np.arange(len(p)), bins):
        if len(idx): e += len(idx) / len(p) * abs(p[idx].mean() - y[idx].mean())
    return float(e)
