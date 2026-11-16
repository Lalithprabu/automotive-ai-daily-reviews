"""Linear MPC on the robot's look-ahead point with aggregated, prediction-error-tightened half-plane constraints.

Robot: unicycle (x, y, θ), look-ahead point  P = (x + ε cosθ, y + ε sinθ).  Input-output linearisation makes
P a single integrator:  ξ_{k+1} = ξ_k + T_s u_k,  u = virtual Cartesian velocity, with
        v = [cosθ, sinθ] u ,     ω = (1/ε) [-sinθ, cosθ] u                       (θ from the previous solution)

QP over U = (u_0..u_{N-1}) plus slacks (OSQP):
    min  Σ_i q ||ξ_i - r_i||² + p ||ξ_N - r_N||² + r ||u_i||² + w_ps Σ s1 + w_hard Σ s2 (+ tiny quadratic)
    s.t. 0 <= v_i <= v_max,  |ω_i| <= ω_max,  ||u_i - u_{i-1}||_∞ <= a_max T_s,  corridor walls on ξ_y
         for every THREATENING group g and step i:   h_gi · ξ_i + s1_gi + s2_gi >= ℓ_gi
                                                     0 <= s1_gi <= personal-space depth,  s2_gi >= 0

Aggregation (my construction; the paper's grouping criterion was not readable): agents whose effective discs
leave a gap smaller than the robot's width are merged into one group (single linkage).  Each group gets ONE
separating line per step:  h = unit vector from the group's closest member to the robot's predicted point,
ℓ = max_j ( h·p̂_j,i + R_j + m_j,i ),  R_j = r_robot + ε + r_j.
Collision-cone screening: a group is a threat at step i if the predicted robot point is inside an inflated
member disc, or the robot-minus-member relative velocity lies inside the cone of tangent rays to that disc.
The Markov margin m_j,i enters BOTH the screening radius and ℓ  (as in the paper's abstract).
"""
from __future__ import annotations

import numpy as np
import osqp
import scipy.sparse as sp

from . import world as Wd


class MPC:
    def __init__(self, N=20, dt=Wd.DT, eps=0.2, v_max=1.2, w_max=1.5, a_max=1.5, v_ref=1.0,
                 q=1.0, p=5.0, r=0.3, w_ps=40.0, w_hard=4000.0, aware=True, slack_decay=0.85):
        self.N, self.dt, self.eps = N, dt, eps
        self.v_max, self.w_max, self.a_max, self.v_ref = v_max, w_max, a_max, v_ref
        self.q, self.p, self.r, self.w_ps, self.w_hard, self.aware = q, p, r, w_ps, w_hard, aware
        self.slack_decay = slack_decay
        n = 2 * N
        L = np.kron(np.tril(np.ones((N, N))), np.eye(2))                      # ξ_i = ξ0 + dt Σ_{j<=i} u_j
        self.L = dt * L                                                        # (2N, 2N)
        Qd = np.full(n, q)
        Qd[-2:] = p
        self.Qd = Qd
        self.P_u = self.L.T @ (Qd[:, None] * self.L) + r * np.eye(n)
        self.U_prev = np.zeros((N, 2))
        self.last = {}

    # ------------------------------------------------------------------------------------------ groups
    def groups(self, pos0: np.ndarray, r_eff: np.ndarray) -> list[list[int]]:
        """Single-linkage grouping on the CURRENT positions: gap < robot width (+ε) -> same group."""
        n = len(pos0)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        thr = 2 * (Wd.R_ROBOT + self.eps)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(pos0[i] - pos0[j]) - r_eff[i] - r_eff[j] < thr:
                    parent[find(i)] = find(j)
        out: dict[int, list[int]] = {}
        for i in range(n):
            out.setdefault(find(i), []).append(i)
        return list(out.values())

    # ------------------------------------------------------------------------------------------ solve
    def solve(self, state: np.ndarray, goal: np.ndarray, pred: np.ndarray, marg: np.ndarray,
              kinds: list[str], radii: np.ndarray, u_applied: np.ndarray):
        """state (x, y, θ); pred (B, N, 2) predicted agent positions; marg (B, N) margins.

        Returns (v, ω) for the unicycle plus the planned look-ahead path (N, 2).
        """
        N, dt, eps = self.N, self.dt, self.eps
        x, y, th = state
        xi0 = np.array([x + eps * np.cos(th), y + eps * np.sin(th)])
        # reference: straight toward the goal at v_ref, y pulled to the corridor centre
        d = goal - xi0
        dist = np.linalg.norm(d)
        steps = np.arange(1, N + 1)
        along = np.minimum(steps * dt * self.v_ref, dist)
        ref = xi0 + along[:, None] * (d / max(dist, 1e-6))[None]
        # warm start / linearisation trajectory: previous plan shifted by one step
        U_ws = np.vstack([self.U_prev[1:], self.U_prev[-1:]])
        xi_bar = xi0 + np.cumsum(U_ws * dt, 0)                                 # (N, 2)
        th_bar = np.concatenate([[th], np.arctan2(U_ws[:-1, 1], U_ws[:-1, 0] + 1e-9)])
        sp_bar = np.linalg.norm(U_ws, axis=1)
        th_bar = np.where(np.concatenate([[1.0], sp_bar[:-1]]) > 0.05, th_bar, th)

        # ---------------- collision rows
        B = len(pred)
        r_phys = Wd.R_ROBOT + eps + radii                                      # (B,)
        ps_depth = np.array([(Wd.PS_PED - Wd.R_PED) if (k == "ped" and self.aware) else 0.0 for k in kinds])
        r_eff = radii + ps_depth
        rows = []                                                              # (i, h, ell, ps_depth)
        if B:
            vel = np.gradient(pred, dt, axis=1)                                # (B, N, 2) predicted velocity
            u_bar = U_ws
            for g in self.groups(pred[:, 0], r_eff):
                g = np.array(g)
                for i in range(N):
                    c = pred[g, i]                                             # (G, 2)
                    R = r_phys[g] + ps_depth[g] + marg[g, i]                   # screening radius (G,)
                    dv = c - xi_bar[i]
                    dn = np.linalg.norm(dv, axis=1)
                    threat = np.any(dn <= R + 0.3)
                    if not threat:
                        vrel = u_bar[i] - vel[g, i]
                        vn = np.linalg.norm(vrel, axis=1) + 1e-9
                        cosang = (vrel * dv).sum(1) / (vn * dn)
                        alpha = np.arcsin(np.clip(R / dn, 0, 1))
                        tt = dn / vn                                           # time to reach (s)
                        threat = np.any((cosang >= np.cos(alpha)) & (tt < 6.0))
                    if not threat:
                        continue
                    jc = np.argmin(dn - R)
                    h = xi_bar[i] - c[jc]
                    nh = np.linalg.norm(h)
                    h = h / nh if nh > 1e-6 else np.array([-np.cos(th), -np.sin(th)])
                    ell = np.max(c @ h + r_phys[g] + ps_depth[g] + marg[g, i])
                    rows.append((i, h, ell, float(np.max(ps_depth[g]))))

        n_u, n_c = 2 * N, len(rows)
        n_var = n_u + 2 * n_c
        # ---------------- cost
        P = np.zeros((n_var, n_var))
        P[:n_u, :n_u] = self.P_u
        P[n_u:, n_u:] = 1e-3 * np.eye(2 * n_c)
        qv = np.zeros(n_var)
        xi0_rep = np.tile(xi0, N)
        qv[:n_u] = self.L.T @ (self.Qd * (xi0_rep - ref.reshape(-1)))
        # slack prices decay along the horizon: an imminent violation is far more expensive than one 2 s out,
        # so infeasible far-horizon rows (large Markov margins) cannot distort the near-term action (my choice)
        decay = np.array([self.slack_decay ** i for i, *_ in rows]) if rows else np.zeros(0)
        qv[n_u:n_u + n_c] = self.w_ps * decay
        qv[n_u + n_c:] = self.w_hard * decay
        # ---------------- constraints
        A, lo, hi = [], [], []
        for i in range(N):
            c_, s_ = np.cos(th_bar[i]), np.sin(th_bar[i])
            rv = np.zeros(n_var); rv[2 * i:2 * i + 2] = [c_, s_]
            A.append(rv); lo.append(0.0); hi.append(self.v_max)
            rw = np.zeros(n_var); rw[2 * i:2 * i + 2] = np.array([-s_, c_]) / eps
            A.append(rw); lo.append(-self.w_max); hi.append(self.w_max)
            for k in range(2):                                                 # rate limit on u
                ra = np.zeros(n_var); ra[2 * i + k] = 1.0
                prev = u_applied[k] if i == 0 else 0.0
                if i > 0:
                    ra[2 * (i - 1) + k] = -1.0
                A.append(ra); lo.append(prev - self.a_max * dt); hi.append(prev + self.a_max * dt)
            ry = np.zeros(n_var); ry[:n_u] = self.L[2 * i + 1]                # walls on ξ_y
            A.append(ry)
            lo.append(-Wd.HALF_W + Wd.R_ROBOT - xi0[1]); hi.append(Wd.HALF_W - Wd.R_ROBOT - xi0[1])
        for k, (i, h, ell, depth) in enumerate(rows):
            rc = np.zeros(n_var)
            rc[:n_u] = h @ self.L[2 * i:2 * i + 2]
            rc[n_u + k] = 1.0
            rc[n_u + n_c + k] = 1.0
            A.append(rc); lo.append(ell - h @ xi0); hi.append(np.inf)
            r1 = np.zeros(n_var); r1[n_u + k] = 1.0
            A.append(r1); lo.append(0.0); hi.append(depth)
            r2 = np.zeros(n_var); r2[n_u + n_c + k] = 1.0
            A.append(r2); lo.append(0.0); hi.append(np.inf)
        A = sp.csc_matrix(np.array(A))
        prob = osqp.OSQP()
        prob.setup(sp.csc_matrix(np.triu(P)), qv, A, np.array(lo), np.array(hi), verbose=False,
                   eps_abs=1e-4, eps_rel=1e-4, max_iter=4000, polishing=False)
        x0 = np.zeros(n_var); x0[:n_u] = U_ws.reshape(-1)
        prob.warm_start(x=x0)
        res = prob.solve()
        if res.x is None or not np.all(np.isfinite(res.x)):
            U = np.zeros((N, 2))                                               # emergency: stop
        else:
            U = res.x[:n_u].reshape(N, 2)
        self.U_prev = U
        u0 = U[0]
        v = float(np.clip(np.cos(th) * u0[0] + np.sin(th) * u0[1], 0.0, self.v_max))
        w = float(np.clip((-np.sin(th) * u0[0] + np.cos(th) * u0[1]) / eps, -self.w_max, self.w_max))
        plan = xi0 + np.cumsum(U * dt, 0)
        self.last = dict(rows=rows, plan=plan, ref=ref, n_rows=len(rows))
        return v, w, u0, plan
