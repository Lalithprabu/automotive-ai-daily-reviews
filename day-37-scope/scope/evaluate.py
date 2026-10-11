"""Calibration + metrics shared by train.py / simulate.py / tests.

Primary metrics are PER-TIMESTEP 2-D tube metrics (what a planner consumes as an occupancy prediction).
The joint 24-D likelihood is deliberately not used: smooth temporal kernels make full covariances near-singular,
so a joint NLL is dominated by numerically meaningless tiny-eigenvalue directions (found + dropped during development).
"""
import math
import numpy as np
import torch
from scipy.stats import chi2
from .data import D, T_FUT

_I = np.arange(T_FUT)


def blocks(Sigma):
    """(B,D,D) -> (B,T,2,2) float64 per-timestep position covariance."""
    S = Sigma.double()
    B = S.shape[0]
    return S.reshape(B, T_FUT, 2, T_FUT, 2)[:, _I, :, _I, :].permute(1, 0, 2, 3)


def marg_m2_logdet(Sigma, e):
    Sm = blocks(Sigma) + 1e-6 * torch.eye(2, dtype=torch.float64)
    ee = e.double().reshape(e.shape[0], T_FUT, 2)
    m2 = torch.einsum("bti,btij,btj->bt", ee, torch.linalg.inv(Sm), ee)
    return m2, torch.log(torch.linalg.det(Sm))


def fit_alpha(Sigma, e):
    """Closed-form scalar minimising mean per-timestep 2-D NLL: alpha = mean(m2)/2."""
    m2, _ = marg_m2_logdet(Sigma, e)
    return float(m2.mean() / 2)


def metrics(Sigma, e, alpha=1.0):
    m2, ld = marg_m2_logdet(Sigma.double() * alpha, e)
    nll = 0.5 * (2 * math.log(2 * math.pi) + ld + m2)
    area = torch.sqrt(torch.exp(ld[:, -1])).mean()             # final-step ellipse sharpness (m^2, ~1-sigma area/pi)
    return dict(nll_per_step=float(nll.mean()), nll_last_step=float(nll[:, -1].mean()),
                cov95=float((m2 <= chi2.ppf(0.95, 2)).double().mean()),
                cov95_last_step=float((m2[:, -1] <= chi2.ppf(0.95, 2)).double().mean()),
                final_tube_area=float(area))
