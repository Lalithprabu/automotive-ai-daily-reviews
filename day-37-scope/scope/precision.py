"""SCOPE-style module: predicts a structured precision matrix around a nominal trajectory by distilling the
frozen diffusion model's score curvature, then turns it into per-timestep Gaussian 'tubes' without sampling.

Score at noise level sigma0:  s(x) = (D(x,sigma0) - x) / sigma0^2   ->   Hessian of -log p = (I - dD/dx) / sigma0^2.
Head output:  P = s0 * ( L L^T + U U^T + eps I ),  s0 = 1/sigma0^2,
  L : lower-triangular, element-banded (local temporal coupling), positive diagonal  -> block-banded part
  U : (D x rank) low-rank factor (long-range coupling)
PSD by construction. (Parameterisation is this repo's own choice; the abstract only says banded + low-rank.)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp
from .data import D, COND_DIM, N_MODES, T_FUT


class PrecisionHead(nn.Module):
    def __init__(self, sigma0=0.05, hidden=256, bandwidth=6, rank=4):
        super().__init__()
        self.sigma0, self.rank, self.eps = sigma0, rank, 1e-4
        i, j = torch.meshgrid(torch.arange(D), torch.arange(D), indexing="ij")
        self.register_buffer("mask", ((i >= j) & (i - j <= bandwidth)).float())
        self.register_buffer("log_scale", torch.zeros(1))      # calibration: log alpha (set by calibrate)
        self.net = nn.Sequential(nn.Linear(COND_DIM + N_MODES + D, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, D * D + D * rank))
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def normalized_precision(self, cond, mode, anchor):
        """P / s0 (B,D,D); shapes: cond (B,17) mode (B,2) anchor (B,24)."""
        B = cond.shape[0]
        out = self.net(torch.cat([cond, mode, anchor], 1))
        raw = out[:, : D * D].reshape(B, D, D)
        U = out[:, D * D:].reshape(B, D, self.rank)
        diag = F.softplus(torch.diagonal(raw, dim1=1, dim2=2) - 1.5) + 1e-3
        off = raw * self.mask * (1 - torch.eye(D))
        L = off * 0.5 + torch.diag_embed(diag)                       # (B,D,D) banded lower-tri
        return L @ L.transpose(1, 2) + U @ U.transpose(1, 2) + self.eps * torch.eye(D)

    def covariance(self, cond, mode, anchor, calibrated=True):
        """Dense solve for D=24 (a structured banded+Woodbury solver would be O(T); not implemented here)."""
        Pn = self.normalized_precision(cond, mode, anchor)
        Sig = torch.cholesky_inverse(torch.linalg.cholesky(Pn)) * self.sigma0 ** 2   # = (s0*Pn)^-1
        return Sig * torch.exp(self.log_scale) if calibrated else Sig


def marginals(Sigma):
    """(B,D,D) -> (B,T,2,2) per-timestep position covariance blocks."""
    B = Sigma.shape[0]
    S = Sigma.reshape(B, T_FUT, 2, T_FUT, 2)
    idx = torch.arange(T_FUT)
    return S[:, idx, :, idx, :].permute(1, 0, 2, 3)


def hvp_target(model, anchor, cond, mode, v, sigma0):
    """Normalised curvature-vector product (I - dD/dx) v  via forward-mode JVP (no dense Hessian)."""
    f = lambda x: model(x, sigma0, cond, mode)
    _, Jv = jvp(f, (anchor,), (v,))
    return v - Jv


def dense_target(model, anchor, cond, mode, sigma0):
    """Dense (I - J) for D=24 (only used for diagnostics + KL term)."""
    cols = []
    for k in range(D):
        e = torch.zeros_like(anchor); e[:, k] = 1.0
        cols.append(hvp_target(model, anchor, cond, mode, e, sigma0))
    Pt = torch.stack(cols, 2)
    return 0.5 * (Pt + Pt.transpose(1, 2))


def psd_inverse(Pt, lo=1e-3, hi=1.0):
    w, V = torch.linalg.eigh(Pt)
    w = w.clamp(lo, hi)
    return (V / w[:, None, :]) @ V.transpose(1, 2), torch.log(w).sum(1)


def distill_loss(head, model, anchor, cond, mode, probes=8, Sig_t=None, logdet_Sig_t=None, kl_weight=0.0):
    """Hutchinson-probe matching of curvature responses + optional affine-invariant (Gaussian KL) term."""
    Pn = head.normalized_precision(cond, mode, anchor)
    loss = 0.0
    for _ in range(probes):
        v = torch.randint(0, 2, anchor.shape).float() * 2 - 1
        with torch.no_grad():
            tgt = hvp_target(model, anchor, cond, mode, v, head.sigma0)
        loss = loss + ((torch.einsum("bij,bj->bi", Pn, v) - tgt) ** 2).mean() / probes
    kl = torch.zeros(())
    if Sig_t is not None and kl_weight > 0:
        ld = 2 * torch.log(torch.diagonal(torch.linalg.cholesky(Pn), dim1=1, dim2=2)).sum(1)
        tr = torch.einsum("bij,bji->b", Pn, Sig_t)
        kl = (0.5 * (tr - D - ld - logdet_Sig_t)).mean()
        loss = loss + kl_weight * kl
    return loss, kl
