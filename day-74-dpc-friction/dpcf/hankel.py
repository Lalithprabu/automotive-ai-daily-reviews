"""PyTorch core: data-driven multi-step predictor of DPC / DeePC with matched, mismatched and POOLED data.

Behavioural systems view (Willems' fundamental lemma): for an LTI system and persistently exciting data, every
length-L input/output trajectory is a linear combination of the columns of the block-Hankel matrix of recorded data

        H_L(w) = [ w_1  w_2  ...  w_{T-L+1} ]          w_t = (u_t, y_t),  each column = L consecutive samples
                 [ ...                      ]
                 [ w_L  w_{L+1} ... w_T      ]

Split rows into past (T_ini samples) and future (N samples):  U_p, Y_p, U_f, Y_f.  DPC predicts

        find g:   U_p g ~ u_ini,   Y_p g ~ y_ini,   U_f g = u_f            ->     y_hat_f = Y_f g

Noise + nonlinearity make the system inconsistent, so g is regularised (here l2, "ridge" DeePC):

    g* = argmin  || W (Z g - b) ||^2 + lam ||g||^2,      Z = [U_p; Y_p; U_f],  b = [u_ini; y_ini; u_f]
       = Z^T W^2 (Z Z^T W^2 + lam I)^-1 ... (kernel form, see predictor_matrix)

so y_hat_f = K b with a SMALL matrix K (N p  x  rows(Z)), linear in b, computed once per data set.

POOLING (the paper's idea, my implementation): data recorded at different friction levels are not one
trajectory, so their Hankel matrices are concatenated COLUMN-wise ("mosaic" Hankel):

        H_pool = [ H_L(w^{mu_high})  |  H_L(w^{mu_low}) ]

Every column is still a valid trajectory of SOME friction level; the initial-trajectory match decides how the
columns are mixed.  No friction estimate is formed anywhere.

Signals in this repo: u = (delta [rad], kappa_ref [1/m]),  y = (e_y [m], e_psi [rad]).  kappa_ref is a measured
exogenous input whose future values are known from the racing line, so it enters u_f as a fixed value.

Shape legend: m inputs, p outputs, Ti = T_ini, N horizon, C Hankel columns, B batch.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def block_hankel(w: torch.Tensor, L: int) -> torch.Tensor:
    """w (T, q) -> block-Hankel (L*q, T-L+1).  Column j stacks w_j..w_{j+L-1} (time-major, channel-minor)."""
    T, q = w.shape
    cols = w.unfold(0, L, 1)                       # (T-L+1, q, L)
    return cols.permute(0, 2, 1).reshape(T - L + 1, L * q).T.contiguous()


class HankelData(nn.Module):
    """Normalised, split Hankel matrices built from one or several (u, y) recordings.

    Several recordings (e.g. two friction levels, or several runs) -> column-wise concatenation (mosaic).
    """

    def __init__(self, recordings: list[tuple[torch.Tensor, torch.Tensor]], T_ini: int, N: int,
                 stats: dict | None = None):
        super().__init__()
        self.T_ini, self.N = T_ini, N
        u0, y0 = recordings[0]
        self.m, self.p = u0.shape[1], y0.shape[1]
        if stats is None:                          # per-channel scaling (rad vs m vs 1/m)
            U = torch.cat([u for u, _ in recordings]); Y = torch.cat([y for _, y in recordings])
            stats = dict(u_s=U.std(0).clamp_min(1e-6), y_s=Y.std(0).clamp_min(1e-6))
        self.stats = stats
        L = T_ini + N
        Hu, Hy = [], []
        for u, y in recordings:
            Hu.append(block_hankel(u / stats["u_s"], L))     # (L*m, C_k)
            Hy.append(block_hankel(y / stats["y_s"], L))     # (L*p, C_k)
        Hu, Hy = torch.cat(Hu, 1), torch.cat(Hy, 1)          # mosaic: (L*m, C), (L*p, C)
        m, p = self.m, self.p
        self.register_buffer("Up", Hu[: T_ini * m])
        self.register_buffer("Uf", Hu[T_ini * m:])
        self.register_buffer("Yp", Hy[: T_ini * p])
        self.register_buffer("Yf", Hy[T_ini * p:])

    @property
    def n_cols(self) -> int:
        return self.Up.shape[1]


class DPCPredictor(nn.Module):
    """Regularised (l2) data-driven multi-step predictor  y_hat_f = K [u_ini; y_ini; u_f].

    Weights: input rows get w_u (large -> near-hard consistency U g = u, as in DeePC equality constraints),
    output-history rows get w_y (soft: slack on noisy y_ini).  lam is the ridge weight on g.
    """

    def __init__(self, data: HankelData, lam: float = 1.0, w_u: float = 10.0, w_y: float = 1.0):
        super().__init__()
        self.data = data
        Ti, N, m, p = data.T_ini, data.N, data.m, data.p
        Z = torch.cat([data.Up, data.Yp, data.Uf])                          # (R, C),  R = Ti(m+p) + N m
        w = torch.cat([torch.full((Ti * m,), w_u), torch.full((Ti * p,), w_y),
                       torch.full((N * m,), w_u)]).to(Z)
        # kernel form of ridge regression (R << C):  g = Z^T W (W Z Z^T W + lam I)^-1 W b
        Zw = w[:, None] * Z                                                 # (R, C)
        Gram = Zw @ Zw.T + lam * torch.eye(Z.shape[0], dtype=Z.dtype)       # (R, R)
        K = data.Yf @ Zw.T @ torch.linalg.solve(Gram, torch.diag(w))        # (N p, R)
        self.register_buffer("K", K)
        self.Ti, self.N, self.m, self.p = Ti, N, m, p
        self.lam = lam
        # convenience column blocks of K (normalised units)
        self.register_buffer("K_ini", K[:, : Ti * (m + p)])                 # acts on [u_ini; y_ini]
        self.register_buffer("K_uf", K[:, Ti * (m + p):])                   # acts on u_f  (N m)

    def _pack(self, u_ini, y_ini, u_f):
        """(B, Ti, m), (B, Ti, p), (B, N, m) physical units -> b (B, R) normalised, time-major."""
        s = self.data.stats
        return torch.cat([(u_ini / s["u_s"]).flatten(1), (y_ini / s["y_s"]).flatten(1),
                          (u_f / s["u_s"]).flatten(1)], 1)

    def forward(self, u_ini: torch.Tensor, y_ini: torch.Tensor, u_f: torch.Tensor) -> torch.Tensor:
        """Batched multi-step prediction.  -> y_hat_f (B, N, p) in physical units."""
        b = self._pack(u_ini, y_ini, u_f)                                    # (B, R)
        y = (b @ self.K.T).view(-1, self.N, self.p)                          # (B, N, p) normalised
        return y * self.data.stats["y_s"]

    def g(self, u_ini, y_ini, u_f) -> torch.Tensor:
        """The column weights g* themselves (B, C): used to see WHICH friction data the predictor leans on."""
        d = self.data
        Ti, m, p, N = self.Ti, self.m, self.p, self.N
        Z = torch.cat([d.Up, d.Yp, d.Uf])
        w = torch.cat([torch.full((Ti * m,), 10.0), torch.full((Ti * p,), 1.0), torch.full((N * m,), 10.0)]).to(Z)
        Zw = w[:, None] * Z
        b = self._pack(u_ini, y_ini, u_f) * w
        a = torch.linalg.solve(Zw @ Zw.T + self.lam * torch.eye(Z.shape[0], dtype=Z.dtype), b.T)  # (R, B)
        return (Zw.T @ a).T

    def affine_in_delta(self, u_ini, y_ini, kappa_f):
        """Prediction as  y_hat = f0 + G delta_f  for the controller QP (single sample, physical units).

        u = (delta, kappa): delta_f is the decision, kappa_f is known.  Returns f0 (N p,), G (N p, N).
        """
        s = self.data.stats
        N, m, p = self.N, self.m, self.p
        u_f0 = torch.stack([torch.zeros_like(kappa_f), kappa_f], -1)          # delta = 0
        f0 = self.forward(u_ini[None], y_ini[None], u_f0[None])[0].reshape(-1)  # (N p)
        # columns of K_uf belonging to the delta channel (time-major, channel 0), rescaled to physical units
        Gn = self.K_uf[:, 0::m] / s["u_s"][0]                                   # (N p, N)
        G = Gn * s["y_s"].repeat(N)[:, None]
        return f0, G


def multistep_rmse(pred: DPCPredictor, u: torch.Tensor, y: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """Open-loop multi-step RMS error over a test recording, per horizon step and output -> (N, p)."""
    Ti, N = pred.Ti, pred.N
    L = Ti + N
    idx = torch.arange(0, u.shape[0] - L + 1, stride)
    win_u = torch.stack([u[i:i + L] for i in idx.tolist()])                    # (B, L, m)
    win_y = torch.stack([y[i:i + L] for i in idx.tolist()])                    # (B, L, p)
    yh = pred(win_u[:, :Ti], win_y[:, :Ti], win_u[:, Ti:])
    return ((yh - win_y[:, Ti:]) ** 2).mean(0).sqrt()
