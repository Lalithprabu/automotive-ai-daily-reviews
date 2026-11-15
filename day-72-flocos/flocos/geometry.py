"""Half-space convex polygons with temporal basis coefficients (the FloCOS set representation).

One component i of the forward occupancy set at future time t is the convex polygon

    P_i(t) = { p in R^2 :  n_h . (p - c_i(t)) <= s_{i,h}(t),  h = 1..N_H }          (N_H = 8 fixed normals)

with centre and support offsets that evolve smoothly in time through a Bernstein (Bezier) basis of degree D:

    c_i(t)     = sum_d B_d(tau) C_{i,d}                 tau = t / T in [0, 1],  C_{i,d} in R^2
    s_{i,h}(t) = softplus( sum_d B_d(tau) S_{i,d,h} )   >= 0

so a whole 3-second "tube" is a fixed-size coefficient tensor theta_i in R^{(D+1) x (2 + N_H)} (60 numbers).
The predicted occupancy set is the (non-convex) union  O(t) = U_i P_i(t)  of M components.

Every set operation used by training and evaluation is a vectorised point test against the half-spaces,
so it is differentiable (losses) and exact up to Monte-Carlo sampling (metrics).
My choices (paper details were not readable): N_H = 8 normals at 45 deg, Bernstein degree 5, softplus offsets.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import world as W

N_H = 8
D = 5
PHI = torch.arange(N_H) * (2 * math.pi / N_H)                  # (N_H,) normal angles in the agent frame
NORMALS = torch.stack([PHI.cos(), PHI.sin()], -1)               # (N_H, 2)
COEF_DIM = (D + 1) * (2 + N_H)                                  # 60
C_SCALE = torch.tensor([25.0, 2.5])                             # coefficient normalisation (centre)
S_MEAN, S_STD = 1.4, 0.6                                        # coefficient normalisation (raw offsets)


def bernstein(tau: torch.Tensor, deg: int = D) -> torch.Tensor:
    """tau (T,) -> basis matrix (T, deg+1)."""
    k = torch.arange(deg + 1, dtype=tau.dtype)
    binom = torch.tensor([math.comb(deg, int(i)) for i in k], dtype=tau.dtype)
    return binom * tau[:, None] ** k * (1 - tau[:, None]) ** (deg - k)


TAU_FIT = torch.cat([torch.zeros(1), torch.tensor(W.T_FUT, dtype=torch.float32)]) / W.T_FUT[-1]   # (H+1,)
B_FIT = bernstein(TAU_FIT)                                      # (H+1, D+1) incl. t = 0
B_FUT = B_FIT[1:]                                               # (H, D+1)  future steps only
B_PINV = torch.linalg.pinv(B_FIT)                               # (D+1, H+1)


def inv_softplus(y):
    return y + torch.log(-torch.expm1(-y))


def rect_support(psi: torch.Tensor, L: float = W.VEH_L, Wd: float = W.VEH_W, ex=None, ey=None) -> torch.Tensor:
    """Support values of an oriented L x W rectangle along the N_H normals.  psi (...) -> (..., N_H).

    h(n) = L/2 |n . e_long| + W/2 |n . e_lat|  is the exact support function of the rectangle, so the octagon
    with these offsets is the TIGHTEST 8-normal polygon enclosing it: identical to the rectangle when psi is a
    multiple of 45 deg, an over-approximation otherwise (+10 % area at 2 deg, +25 % at 6 deg, +59 % at 23 deg).
    """
    if ex is not None:                                           # axis-aligned box with half extents ex, ey
        return ex[..., None] * PHI.cos().abs() + ey[..., None] * PHI.sin().abs()
    a = PHI - psi[..., None]
    return L / 2 * a.cos().abs() + Wd / 2 * a.sin().abs()


# ----------------------------------------------------------------------------------------------- coefficients
def encode_tube(fut: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    """Least-squares Bernstein fit of a realised future footprint tube.

    fut (B, H, 2), head (B, H) -> normalised coefficients (B, (D+1)*(2+N_H)).  Used as the flow-matching
    target x_1 ("a single tight polygon tube around one plausible future").
    """
    B = fut.shape[0]
    path = torch.cat([fut.new_zeros(B, 1, 2), fut], 1)                       # (B, H+1, 2), starts at origin
    sup = rect_support(torch.cat([head.new_zeros(B, 1), head], 1))           # (B, H+1, N_H)
    raw = inv_softplus(sup.clamp(min=1e-3))
    Cc = B_PINV @ path                                                       # (B, D+1, 2)
    Sc = B_PINV @ raw                                                        # (B, D+1, N_H)
    return torch.cat([Cc / C_SCALE, (Sc - S_MEAN) / S_STD], -1).flatten(1)


def decode(theta: torch.Tensor):
    """theta (..., COEF_DIM) normalised -> centres (..., H, 2), supports (..., H, N_H) at the future steps."""
    th = theta.unflatten(-1, (D + 1, 2 + N_H))
    Cc = th[..., :2] * C_SCALE
    Sc = th[..., 2:] * S_STD + S_MEAN
    c = torch.einsum("td,...dk->...tk", B_FUT, Cc)
    s = F.softplus(torch.einsum("td,...dk->...tk", B_FUT, Sc))
    return c, s


# ----------------------------------------------------------------------------------------------- point tests
def violation(p: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Signed half-space violation  max_h ( n_h.(p - c) - s_h ).  <= 0  <=>  p inside the polygon.

    p (..., P, 2), c (..., 2), s (..., N_H) -> (..., P)
    """
    proj = (p - c[..., None, :]) @ NORMALS.T                                 # (..., P, N_H)
    return (proj - s[..., None, :]).amax(-1)


def union_violation(p, c, s):
    """p (B, T, P, 2); c (B, M, T, 2); s (B, M, T, N_H) -> min over components (B, T, P)."""
    v = violation(p[:, None], c, s)                                          # (B, M, T, P)
    return v.amin(1)


def footprint_points(c: torch.Tensor, psi: torch.Tensor, nx=9, ny=5, L=W.VEH_L, Wd=W.VEH_W, shrink=0.999):
    """Uniform grid of points on an oriented rectangle.  c (..., 2), psi (...) -> (..., nx*ny, 2)."""
    gx = torch.linspace(-L / 2, L / 2, nx) * shrink
    gy = torch.linspace(-Wd / 2, Wd / 2, ny) * shrink
    loc = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)   # (P, 2)
    ca, sa = psi.cos()[..., None], psi.sin()[..., None]
    x = ca * loc[:, 0] - sa * loc[:, 1]
    y = sa * loc[:, 0] + ca * loc[:, 1]
    return torch.stack([x, y], -1) + c[..., None, :]


def union_bbox(c, s):
    """Exact axis-aligned bounding box of the union (normals 0, 90, 180, 270 deg are in the set).
    c (B, M, T, 2), s (B, M, T, N_H) -> lo (B, T, 2), hi (B, T, 2)."""
    q = N_H // 4
    hi = torch.stack([c[..., 0] + s[..., 0], c[..., 1] + s[..., q]], -1).amax(1)
    lo = torch.stack([c[..., 0] - s[..., 2 * q], c[..., 1] - s[..., 3 * q]], -1).amin(1)
    return lo, hi


def union_area(c, s, n_pts=1024, soft_tau=None, gen=None):
    """Monte-Carlo area of the union per (B, T).  soft_tau -> differentiable soft-membership version."""
    lo, hi = union_bbox(c, s)
    lo, hi = lo.detach(), hi.detach()
    u = torch.rand(*lo.shape[:-1], n_pts, 2, generator=gen)
    p = lo[..., None, :] + u * (hi - lo)[..., None, :]                       # (B, T, P, 2)
    box = (hi - lo).prod(-1)                                                 # (B, T)
    if soft_tau is None:
        inside = (union_violation(p, c, s) <= 0).float()
    else:                                                                    # soft union 1 - prod(1 - m_i)
        m = torch.sigmoid(-violation(p[:, None], c, s) / soft_tau)           # (B, M, T, P)
        inside = 1 - (1 - m).prod(1)
    return box * inside.mean(-1)


def coverage(c, s, fut, head, nx=9, ny=5):
    """Fraction of the TRUE footprint covered by the predicted union.  -> (B, T)."""
    p = footprint_points(fut, head, nx, ny)                                  # (B, T, P, 2)
    return (union_violation(p, c, s) <= 0).float().mean(-1)


# ----------------------------------------------------------------------------------------------- drawing
def polygon_vertices(c: np.ndarray, s: np.ndarray, big=500.0) -> np.ndarray:
    """Exact vertices of {p : n_h.(p-c) <= s_h} by Sutherland-Hodgman clipping of a big square."""
    poly = np.array([[-big, -big], [big, -big], [big, big], [-big, big]], float) + c
    N = NORMALS.numpy().astype(float)
    for h in range(N_H):
        n, b = N[h], float(N[h] @ c + s[h])
        out = []
        for k in range(len(poly)):
            P, Q = poly[k], poly[(k + 1) % len(poly)]
            fp, fq = n @ P - b, n @ Q - b
            if fp <= 0:
                out.append(P)
            if fp * fq < 0:
                out.append(P + (Q - P) * fp / (fp - fq))
        poly = np.array(out) if out else np.zeros((0, 2))
        if len(poly) == 0:
            break
    return poly
