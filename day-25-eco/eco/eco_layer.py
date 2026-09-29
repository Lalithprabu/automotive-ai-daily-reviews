"""Endpoint-Constrained Optimization (ECO) layer -- this repo's reconstruction.

Paper (arXiv:2609.31383) describes ECO only at abstract level: a training-free post-processing layer that
  (1) anchors the trajectory to the vehicle's executed history,
  (2) preserves the policy's predicted endpoint, and
  (3) reshapes the intermediate waypoints for feasibility / motion compliance.
The objective/solver below is OUR choice: a closed-form, batched, differentiable quadratic program.

Sequence s = [history (H pts) | ego origin | w_1 ... w_T]   (all in the ego frame, 0.5 s apart)
Unknowns x = w_1..w_{T-1}.  Fixed: history, ego origin, and the endpoint w_T.
    minimise  lam_a * ||D2 s||^2 + lam_j * ||D3 s||^2 + lam_f * ||x - w_raw||^2
where D2/D3 are finite-difference operators (acceleration / jerk) applied across the WHOLE sequence,
so the optimised path is smooth *through* the already-executed history, not just in the future.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def _diff_matrix(n: int, order: int) -> torch.Tensor:
    """(n-order, n) finite-difference operator of the given order."""
    d = torch.eye(n, dtype=torch.float64)
    for _ in range(order):
        d = d[1:] - d[:-1]
    return d


class EndpointConstrainedOptimizer(nn.Module):
    def __init__(self, horizon: int = 8, history: int = 4, lambda_accel: float = 1.0,
                 lambda_jerk: float = 4.0, lambda_fidelity: float = 0.6):
        super().__init__()
        assert horizon >= 3, "need at least 3 future waypoints so an interior exists"
        self.T, self.H = horizon, history
        n = history + 1 + horizon                       # N: full sequence length
        d2, d3 = _diff_matrix(n, 2), _diff_matrix(n, 3)
        q = lambda_accel * d2.T @ d2 + lambda_jerk * d3.T @ d3          # (N,N) smoothness quadratic
        unk = torch.arange(history + 1, history + horizon)               # indices of w_1..w_{T-1}
        fixed = torch.tensor([i for i in range(n) if i not in set(unk.tolist())])
        qu = q[unk][:, unk]                                              # Q_uu
        qf = q[unk][:, fixed]                                            # Q_uf
        a = qu + lambda_fidelity * torch.eye(len(unk), dtype=torch.float64)
        a_inv = torch.linalg.inv(a)                                      # constant SPD system -> precompute
        self.register_buffer("a_inv", a_inv.float())                     # (T-1, T-1)
        self.register_buffer("qf", qf.float())                           # (T-1, |fixed|)
        self.register_buffer("fixed_idx", fixed)
        self.lam_f = lambda_fidelity

    def forward(self, wp_raw: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        """
        wp_raw: (B, T, 2)  raw policy waypoints w_1..w_T, ego frame (ego at the origin)
        hist:   (B, H, 2)  executed history points, oldest first, ego frame (excludes the ego origin)
        returns (B, T, 2)  optimised waypoints; the last one equals wp_raw[:, -1] exactly.
        """
        B = wp_raw.shape[0]
        origin = wp_raw.new_zeros(B, 1, 2)
        # fixed part of the sequence, in the same order as fixed_idx: history, origin, endpoint
        fixed = torch.cat([hist, origin, wp_raw[:, -1:]], dim=1)          # (B, |fixed|, 2)
        rhs = self.lam_f * wp_raw[:, :-1] - torch.einsum("uf,bfc->buc", self.qf, fixed)   # (B,T-1,2)
        x = torch.einsum("uv,bvc->buc", self.a_inv, rhs)                  # (B, T-1, 2) interior waypoints
        return torch.cat([x, wp_raw[:, -1:]], dim=1)                      # endpoint preserved bit-exactly
