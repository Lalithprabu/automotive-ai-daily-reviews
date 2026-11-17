"""Controllers: data-driven predictive control (DPC) on Hankel data, and a model-based linear MPC baseline.

Both solve the same condensed QP in the steering sequence delta_f = (delta_t .. delta_{t+N-1}):

    min  sum_k  q_y e_y,k^2 + q_psi e_psi,k^2 + r delta_k^2 + r_d (delta_k - delta_{k-1})^2
    s.t. |delta_k| <= delta_max,   |delta_k - delta_{k-1}| <= ddelta_max * DT

with the prediction  y_hat = f0 + G delta_f  coming either from the DPC predictor (data only) or from the linear
single-track error model (Rajamani) with the TRUE friction (oracle baseline; the paper compares against NMPC).
"""
from __future__ import annotations

import numpy as np
import osqp
import scipy.linalg as sla
import scipy.sparse as sp
import torch

from . import vehicle as V
from .hankel import DPCPredictor


class BoxQP:
    """Small dense QP with box + rate constraints, solved with OSQP (re-setup each call; N ~ 15 vars)."""

    def __init__(self, N, q=(40.0, 4.0), r=0.2, r_d=6.0):
        self.N, self.q, self.r, self.r_d = N, q, r, r_d
        D = np.eye(N) - np.eye(N, k=-1)
        self.D = D

    def solve(self, f0, G, delta_prev):
        N = self.N
        Q = np.tile(np.array(self.q), N)                         # (N p,) output weights, time-major
        H = G.T @ (Q[:, None] * G) + self.r * np.eye(N) + self.r_d * self.D.T @ self.D
        e0 = np.zeros(N); e0[0] = delta_prev
        f = G.T @ (Q * f0) - self.r_d * self.D.T @ e0
        A = sp.vstack([sp.eye(N), sp.csc_matrix(self.D)]).tocsc()
        dm, rate = V.P["delta_max"], V.P["ddelta_max"] * V.DT
        lo = np.concatenate([-dm * np.ones(N), -rate * np.ones(N)]); lo[N] += delta_prev
        hi = np.concatenate([dm * np.ones(N), rate * np.ones(N)]); hi[N] += delta_prev
        pr = osqp.OSQP()
        pr.setup(sp.csc_matrix(np.triu(H)), f, A, lo, hi, verbose=False, eps_abs=1e-6, eps_rel=1e-6,
                 polishing=False, max_iter=4000)
        res = pr.solve()
        x = res.x if res.x is not None and np.all(np.isfinite(res.x)) else np.full(N, delta_prev)
        return x


class DPCController:
    """Data-driven predictive control. `predictors` maps a friction label to a DPCPredictor; `select`
    decides which one to use each step ('high', 'low', 'pool', or an oracle callback mu -> label)."""

    def __init__(self, predictors: dict[str, DPCPredictor], select, qp: BoxQP):
        self.pred, self.select, self.qp = predictors, select, qp
        any_p = next(iter(predictors.values()))
        self.Ti, self.N = any_p.Ti, any_p.N
        self.last = {}

    def act(self, u_hist, y_hist, kappa_f, delta_prev, mu_here):
        key = self.select(mu_here) if callable(self.select) else self.select
        P = self.pred[key]
        u_ini = torch.as_tensor(np.asarray(u_hist[-self.Ti:]), dtype=torch.float64)
        y_ini = torch.as_tensor(np.asarray(y_hist[-self.Ti:]), dtype=torch.float64)
        kf = torch.as_tensor(kappa_f, dtype=torch.float64)
        f0, G = P.affine_in_delta(u_ini, y_ini, kf)
        f0, G = f0.numpy(), G.numpy()
        d = self.qp.solve(f0, G, delta_prev)
        self.last = dict(plan=d, y_pred=(f0 + G @ d).reshape(self.N, 2), key=key)
        return float(d[0])


def error_model(mu, vx, p=V.P):
    """Continuous lateral error dynamics (Rajamani), state x = (e_y, de_y, e_psi, de_psi), inputs (delta, kappa).

    Per-AXLE stiffness Cf, Cr from the TRUE friction mu -> oracle model.  psi_des_dot = vx * kappa.
    """
    Cf, Cr = V.cornering_stiffness(mu, p)
    m, Iz, lf, lr = p["m"], p["Iz"], p["lf"], p["lr"]
    vx = max(vx, 0.5)
    A = np.array([
        [0, 1, 0, 0],
        [0, -(Cf + Cr) / (m * vx), (Cf + Cr) / m, (-Cf * lf + Cr * lr) / (m * vx)],
        [0, 0, 0, 1],
        [0, -(Cf * lf - Cr * lr) / (Iz * vx), (Cf * lf - Cr * lr) / Iz, -(Cf * lf ** 2 + Cr * lr ** 2) / (Iz * vx)],
    ])
    Bd = np.array([[0], [Cf / m], [0], [Cf * lf / Iz]])
    # disturbance through psi_des_dot = vx * kappa
    Bk = np.array([[0], [-(Cf * lf - Cr * lr) / (m * vx) - vx], [0], [-(Cf * lf ** 2 + Cr * lr ** 2) / (Iz * vx)]]) * vx
    return A, Bd, Bk


class ModelMPC:
    """Linear MPC on the single-track error model with the TRUE local friction and true state (oracle)."""

    def __init__(self, N, qp: BoxQP):
        self.N, self.qp = N, qp
        self.last = {}

    def act(self, state, ey, epsi, kappa_now, kappa_f, delta_prev, mu):
        X, Y, psi, vx, vy, r = state
        x0 = np.array([ey, vy + vx * epsi, epsi, r - vx * kappa_now])
        A, Bd, Bk = error_model(mu, vx)
        n = 4
        M = sla.expm(np.block([[A, np.hstack([Bd, Bk])], [np.zeros((2, n + 2))]]) * V.DT)
        Ad, Bdd, Bkd = M[:n, :n], M[:n, n:n + 1], M[:n, n + 1:n + 2]
        N = self.N
        Cy = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        f0 = np.zeros(2 * N); G = np.zeros((2 * N, N))
        x = x0.copy(); Phi = np.eye(n)
        powers = [np.eye(n)]
        for _ in range(N):
            powers.append(Ad @ powers[-1])
        for k in range(N):                                         # y_{k+1}
            x = Ad @ x + (Bkd[:, 0] * kappa_f[k])
            f0[2 * k:2 * k + 2] = Cy @ x
            for j in range(k + 1):
                G[2 * k:2 * k + 2, j] = (Cy @ powers[k - j] @ Bdd)[:, 0]
        d = self.qp.solve(f0, G, delta_prev)
        self.last = dict(plan=d, y_pred=(f0 + G @ d).reshape(N, 2), key="model")
        return float(d[0])
