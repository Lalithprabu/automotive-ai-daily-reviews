"""Loss utilities: best-of-many mode selection + the three weighted losses
MM-Future trains with (action flow-matching, scene flow-matching, proposal
scoring). Weights are read from config.yaml's `loss:` block, which mirrors
the paper's reported weights (action 1.0, scene 0.1, score 1.0).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def one_step_z1_estimate(z_t: torch.Tensor, v_pred: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Flow matching one-step reconstruction: given z_t = (1-t)z0 + t*z1 and a
    velocity prediction v ~= (z1 - z0), recover an estimate of z1:
        z1_hat = z_t + (1 - t) * v
    (exact when v is exact; used at train time purely to pick the winning
    mode before full multi-step generation.)
    t: broadcastable to z_t's leading dims.
    """
    while t.dim() < z_t.dim():
        t = t.unsqueeze(-1)
    return z_t + (1.0 - t) * v_pred


def select_best_of_many(trajectory_hat: torch.Tensor, gt_trajectory: torch.Tensor) -> torch.Tensor:
    """trajectory_hat: [B, M, Tf, 2], gt_trajectory: [B, Tf, 2] -> m_star: [B]"""
    dist = (trajectory_hat - gt_trajectory.unsqueeze(1)).norm(dim=-1).mean(dim=-1)  # [B, M]
    return dist.argmin(dim=1)


def gather_winner(per_proposal_loss: torch.Tensor, m_star: torch.Tensor) -> torch.Tensor:
    """per_proposal_loss: [B, M] -> mean loss of the winning proposal per batch element."""
    return per_proposal_loss.gather(1, m_star.unsqueeze(1)).squeeze(1).mean()


def flow_matching_loss(v_pred: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    """Per-(batch, proposal) MSE between predicted velocity and the target
    velocity (z1 - z0) of rectified flow matching. Reduces over all trailing
    dims, keeps [B, M]."""
    target = z1 - z0
    return ((v_pred - target) ** 2).mean(dim=tuple(range(2, v_pred.dim())))


def scoring_loss(scores: torch.Tensor, m_star: torch.Tensor) -> torch.Tensor:
    """Cross-entropy ranking loss: encourages the scorer to rank the winning
    (ground-truth-matching) proposal highest among the M candidates."""
    return F.cross_entropy(scores, m_star)
