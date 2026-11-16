"""Core PyTorch modules of PET-MPC (toy re-implementation of arXiv 2611.15781).

    observed track z_{t-W..t}  ──► StructuredKoopman  (online, sparse 7x7 operator, regularised LS)
                                     ẑ_{t+i|t} = K_t^i z_t ,  i = 1..N          (positions = ẑ[:2])
    realised position p_t      ──► ResidualMoments    (i-step-ahead errors e_i = p_t - p̂_{t|t-i})
                                     Σ_i ← (1-β) Σ_i + β ||e_i||²               (online second moments)
                                ──► markov_margin      m_i = sqrt( Σ_i / δ )
                                     Markov:  P(||e_i||² ≥ m_i²) ≤ E||e_i||² / m_i² = δ

The margins m_i inflate every obstacle's safety radius at prediction step i (collision-cone screening AND
the separating half-plane constraints of the MPC, see planner.py).

What follows the paper (sections I-II, readable): the 7-D observable [x, y, vx, vy, θ, ω, s], a structured
(sparse) 7x7 operator identified ONLINE by regularised least squares over a sliding window of one-step
transitions with a Frobenius pull toward the previous operator, held fixed over the horizon, and a
constant-velocity / constant-turn-rate prior for new tracks; prediction errors characterised online by
SECOND MOMENTS and turned into Markov-based margins.
My choices (section III onward was not readable): the sparsity pattern, window W = 12, λ = 0.5, the EMA
form of the moment estimator (β = 0.08), the population prior for young tracks, δ = 0.25 and the cap.

Shape legend: B tracks, W window length, N horizon, D = 7 observables.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

D = 7  # [x, y, vx, vy, theta, omega, speed]


def cvtr_prior(dt: float, dtype=torch.float64) -> torch.Tensor:
    """Constant-velocity / constant-turn-rate linear prior K_0 (D, D):  x+=dt*vx, y+=dt*vy, θ+=dt*ω."""
    K = torch.eye(D, dtype=dtype)
    K[0, 2] = dt
    K[1, 3] = dt
    K[4, 5] = dt
    return K


def sparsity_mask(dtype=torch.float64) -> torch.Tensor:
    """Allowed (row, col) entries of K: the structure encodes simple kinematics.

        x'  = x + dt vx         y'  = y + dt vy      (FIXED kinematic rows, not identified)
        vx' <- vx, vy, s        vy' <- vx, vy, s     (identified: local rotation / scaling of the velocity)
        θ'  <- θ, ω             ω'  <- ω             s' <- vx, vy, s
    14 free parameters instead of 49.
    """
    M = torch.zeros(D, D, dtype=dtype)
    allowed = {0: [0, 2], 1: [1, 3], 2: [2, 3, 6], 3: [2, 3, 6], 4: [4, 5], 5: [5], 6: [2, 3, 6]}
    for r, cols in allowed.items():
        M[r, cols] = 1.0
    return M


class StructuredKoopman(nn.Module):
    """Batched online identification of a sparse Koopman operator per track + multi-step propagation.

    Per track b and row r (support S_r given by the mask), solve the ridge problem

        k_r* = argmin_k  Σ_w ( z_{w+1, r} - k · z_w )²  +  λ || k - k_r^prev ||²    s.t. k_j = 0 for j ∉ S_r

    in closed form:  (Zᵀ Z + λ I + Λ_off) k = Zᵀ y_r + λ k_r^prev,   Λ_off = big * diag(1 - mask_r).
    All B x D row problems are solved in one batched torch.linalg.solve call.
    """

    def __init__(self, dt: float = 0.1, window: int = 12, lam: float = 2.0, big: float = 1e8, rho_max: float = 1.0):
        super().__init__()
        self.dt, self.window, self.lam, self.big, self.rho_max = dt, window, lam, big, rho_max
        self.register_buffer("K0", cvtr_prior(dt))
        self.register_buffer("mask", sparsity_mask())

    def identify(self, Z: torch.Tensor, valid: torch.Tensor, K_prev: torch.Tensor) -> torch.Tensor:
        """Z (B, W+1, D) last W+1 observables (oldest first), valid (B, W) transition mask, K_prev (B, D, D).

        Returns K (B, D, D).  Positions are made RELATIVE to the latest sample so the (affine-free) linear
        operator is translation-invariant over the window.
        """
        B = Z.shape[0]
        Zr = Z.clone()
        Zr[..., :2] = Zr[..., :2] - Z[:, -1:, :2]                              # translation invariance
        X, Y = Zr[:, :-1], Zr[:, 1:]                                           # (B, W, D) inputs / targets
        w = valid.to(Z.dtype)[..., None]                                       # (B, W, 1)
        A = (X * w).transpose(1, 2) @ X                                        # (B, D, D)   Σ_w z zᵀ
        C = (X * w).transpose(1, 2) @ Y                                        # (B, D, D)   Σ_w z y_rᵀ
        I = torch.eye(D, dtype=Z.dtype)
        off = self.big * (1.0 - self.mask)                                     # (D_rows, D_cols)
        # per-row system matrices: (B, D_rows, D, D)
        lhs = A[:, None] + self.lam * I + torch.diag_embed(off)[None]
        rhs = C.transpose(1, 2) + self.lam * K_prev                            # (B, D_rows, D)
        K = torch.linalg.solve(lhs, rhs[..., None])[..., 0]                    # (B, D_rows, D)
        K = K * self.mask
        K[:, :2] = self.K0[:2]                                                 # fixed kinematic position rows
        return self.stabilise(K)

    def stabilise(self, K: torch.Tensor) -> torch.Tensor:
        """Spectral projection of the velocity block (my addition, see SOURCING.md).

        A short, noisy window can yield |λ| > 1 for the (vx, vy, s) block, and K^20 then explodes
        (I measured 4.8 km FDE without this).  If ρ(block) > rho_max the block is rescaled by rho_max / ρ,
        so velocities can rotate and decay but never grow geometrically over the horizon.
        """
        idx = torch.tensor([2, 3, 6])
        blk = K[:, idx][:, :, idx]                                             # (B, 3, 3)
        rho = torch.linalg.eigvals(blk).abs().amax(-1)                         # (B,)
        scale = torch.where(rho > self.rho_max, self.rho_max / rho, torch.ones_like(rho))
        K = K.clone()
        K[:, idx[:, None], idx[None, :]] = blk * scale[:, None, None]
        w = K[:, 5, 5]                                                         # turn-rate persistence
        K[:, 5, 5] = w.clamp(-1.0, 1.0)
        return K

    def propagate(self, K: torch.Tensor, z0: torch.Tensor, N: int) -> torch.Tensor:
        """ẑ_{i} = K^i z_0 for i = 1..N.   K (B, D, D), z0 (B, D) -> (B, N, D) (absolute positions)."""
        out, z = [], z0.clone()
        origin = z0[:, :2].clone()
        z[:, :2] = 0.0                                                         # relative frame (see identify)
        for _ in range(N):
            z = (K @ z[..., None])[..., 0]
            out.append(z)
        Zp = torch.stack(out, 1)
        Zp[..., :2] = Zp[..., :2] + origin[:, None]
        return Zp


def constant_velocity(z0: torch.Tensor, N: int, dt: float) -> torch.Tensor:
    """Baseline predictor: straight line at the current velocity.  z0 (B, D) -> positions (B, N, 2)."""
    i = torch.arange(1, N + 1, dtype=z0.dtype)[None, :, None]
    return z0[:, None, :2] + i * dt * z0[:, None, 2:4]


class ResidualMoments(nn.Module):
    """Online second moments of the i-step-ahead position error, per track and per horizon step.

    Every control step the predictor emits p̂_{t+i|t}, i = 1..N.  i steps later the realised position p_{t+i}
    is observed and   e_i = p_{t+i} - p̂_{t+i|t}   updates   Σ_i ← (1-β) Σ_i + β ||e_i||².
    Young tracks (few residuals) borrow a population prior Σ^pop_i via a count-based blend.
    """

    def __init__(self, N: int, beta: float = 0.08, n_prior: float = 10.0, sigma0: float = 0.05):
        super().__init__()
        self.N, self.beta, self.n_prior = N, beta, n_prior
        i = torch.arange(1, N + 1, dtype=torch.float64)
        # initial population prior: error std growing linearly with the step (sigma0 m per 0.1 s step)
        self.register_buffer("pop", (sigma0 * i) ** 2)                          # (N,)
        self.tracks: dict[int, dict] = {}

    def _get(self, tid: int) -> dict:
        if tid not in self.tracks:
            self.tracks[tid] = dict(S=self.pop.clone(), n=torch.zeros(self.N, dtype=torch.float64), buf=[])
        return self.tracks[tid]

    def push_prediction(self, tid: int, t_idx: int, pred_pos: torch.Tensor):
        """Store p̂_{t+i|t} (N, 2) made at control step t_idx."""
        tr = self._get(tid)
        tr["buf"].append((t_idx, pred_pos.detach().clone()))
        if len(tr["buf"]) > self.N + 1:
            tr["buf"].pop(0)

    def update(self, tid: int, t_idx: int, pos: torch.Tensor) -> torch.Tensor:
        """Realised position at step t_idx -> update moments with every matured prediction. Returns e² (N,) or nan."""
        tr = self._get(tid)
        e2 = torch.full((self.N,), float("nan"), dtype=torch.float64)
        for t0, P in tr["buf"]:
            i = t_idx - t0                                                      # horizon index 1..N
            if 1 <= i <= self.N:
                err = float(((pos - P[i - 1]) ** 2).sum())
                e2[i - 1] = err
                tr["S"][i - 1] = (1 - self.beta) * tr["S"][i - 1] + self.beta * err
                tr["n"][i - 1] += 1
                # population prior tracks the average over all agents (slow EMA)
                self.pop[i - 1] = 0.995 * self.pop[i - 1] + 0.005 * err
        return e2

    def second_moment(self, tid: int) -> torch.Tensor:
        """Blended Σ_i (N,):  w Σ_track + (1-w) Σ_pop,  w = n / (n + n_prior)."""
        tr = self._get(tid)
        w = tr["n"] / (tr["n"] + self.n_prior)
        return w * tr["S"] + (1 - w) * self.pop


def markov_margin(second_moment: torch.Tensor, delta: float = 0.25, cap: float = 1.5) -> torch.Tensor:
    """Distribution-free margin from Markov's inequality on the squared error.

        P(||e||² ≥ m²) ≤ E[||e||²] / m²  := δ      =>     m = sqrt( E[||e||²] / δ )

    No Gaussian assumption: holds for any error distribution whose second moment is estimated correctly,
    at the price of conservatism (δ = 0.25 -> m = 2σ_rms; a Gaussian 2-D error would need only ~1.67σ).
    """
    return torch.sqrt(second_moment / delta).clamp(max=cap)


class PETPredictor(nn.Module):
    """Wraps per-track Koopman identification + residual moments for a set of tracked agents."""

    def __init__(self, N: int = 15, dt: float = 0.1, window: int = 12, lam: float = 2.0, delta: float = 0.25,
                 beta: float = 0.08, cap: float = 1.5, mode: str = "koopman"):
        super().__init__()
        self.N, self.dt, self.window, self.delta, self.cap, self.mode = N, dt, window, delta, cap, mode
        self.koop = StructuredKoopman(dt, window, lam)
        self.moments = ResidualMoments(N, beta)
        self.K: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def predict(self, hists: list[list], t_idx: int):
        """hists[b] = list of 7-D observables of track b (oldest first).

        Returns positions (B, N, 2) and Markov margins (B, N), and logs the predictions for later residuals.
        """
        B = len(hists)
        if B == 0:
            return torch.zeros(0, self.N, 2, dtype=torch.float64), torch.zeros(0, self.N, dtype=torch.float64)
        W = self.window
        Z = torch.zeros(B, W + 1, D, dtype=torch.float64)
        valid = torch.zeros(B, W, dtype=torch.bool)
        for b, h in enumerate(hists):
            h = torch.from_numpy(np.asarray(h[-(W + 1):], dtype=np.float64))      # (L, D), L <= W+1
            L = len(h)
            Z[b, W + 1 - L:] = h                                               # right-aligned, pad with oldest
            Z[b, :W + 1 - L] = h[0]
            valid[b] = torch.arange(W) >= W - (L - 1)                          # only the L-1 real transitions
        z0 = Z[:, -1]
        if self.mode == "cv":
            pos = constant_velocity(z0, self.N, self.dt)
        else:
            K_prev = torch.stack([self.K.get(b, self.koop.K0) for b in range(B)])
            K = self.koop.identify(Z, valid, K_prev)
            for b in range(B):
                self.K[b] = K[b]
            pos = self.koop.propagate(K, z0, self.N)[..., :2]
        for b in range(B):
            self.moments.update(b, t_idx, z0[b, :2])
            self.moments.push_prediction(b, t_idx, pos[b])
        S = torch.stack([self.moments.second_moment(b) for b in range(B)])
        return pos, markov_margin(S, self.delta, self.cap)
