"""Synthetic mode-conditioned pedestrian-forecasting data with KNOWN heteroscedastic noise.

A pedestrian walks along +x and passes a parked obstacle on the left (mode 0, +y) or right (mode 1, -y).
Future noise is a temporally-correlated Gaussian whose covariance depends on the scene:
  * along-track std grows with walking speed v
  * lateral std grows with crowd density c
so a single fixed covariance is provably miscalibrated, and the ground-truth covariance is known exactly.
Flattened layout everywhere: index 2*k + {0: x, 1: y} for future step k.
"""
import numpy as np
import torch

T_PAST, T_FUT, DT = 8, 12, 0.4
D = 2 * T_FUT
COND_DIM = 2 * T_PAST + 1
N_MODES = 2
LAT_AMP = 0.8
_T = (np.arange(T_FUT) + 1) * DT


def _time_kernel(ell=1.6):
    k = np.exp(-0.5 * ((_T[:, None] - _T[None, :]) / ell) ** 2)
    g = 0.35 + 0.65 * _T / _T[-1]          # uncertainty grows with horizon
    return k * g[:, None] * g[None, :]


_K = _time_kernel()


def true_cov(v, c):
    """(N,) speed, (N,) density -> (N,D,D) exact noise covariance."""
    s = np.stack([0.10 * v + 0.03, 0.06 + 0.22 * c], -1)
    cov = np.zeros((len(v), D, D))
    for a in range(2):
        idx = np.arange(a, D, 2)
        cov[:, idx[:, None], idx[None, :]] = _K[None] * (s[:, a] ** 2)[:, None, None]
    return cov


def mean_future(v, mode):
    x = v[:, None] * _T[None]
    y = (1 - 2 * mode)[:, None] * LAT_AMP * np.sin(np.pi * _T / _T[-1])[None]
    out = np.empty((len(v), D))
    out[:, 0::2], out[:, 1::2] = x, y
    return out


def cv_baseline(past):
    """Constant-velocity extrapolation from the (N,T_PAST,2) past (relative to current position)."""
    vel = (past[:, -1] - past[:, 0]) / ((T_PAST - 1) * DT)
    out = np.empty((len(past), D))
    out[:, 0::2] = vel[:, 0:1] * _T[None]
    out[:, 1::2] = vel[:, 1:2] * _T[None]
    return out


def make_split(n, rng):
    v = rng.uniform(0.8, 1.8, n)
    c = rng.uniform(0.0, 1.0, n)
    mode = rng.integers(0, 2, n)
    j = np.arange(T_PAST - 1, -1, -1)
    past = np.zeros((n, T_PAST, 2))
    past[:, :, 0] = -v[:, None] * DT * j[None]
    past += rng.normal(0, 0.02, past.shape)
    cov = true_cov(v, c)
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(D))
    fut = mean_future(v, mode) + np.einsum("nij,nj->ni", L, rng.normal(size=(n, D)))
    cv = cv_baseline(past)
    cond = np.concatenate([past.reshape(n, -1) / 4.0, c[:, None]], 1)
    f = lambda a: torch.tensor(a, dtype=torch.float32)
    return dict(v=v, c=c, mode=mode, past=past, fut=fut, cv=cv, cov=cov,
                cond=f(cond), mode_t=f(np.eye(N_MODES)[mode]), resid=f(fut - cv),
                mean_resid=f(mean_future(v, mode) - cv))


def make_data(cfg, seed):
    rng = np.random.default_rng(seed)
    return {k: make_split(cfg["data"][f"n_{k}"], rng) for k in ("train", "val", "test")}
