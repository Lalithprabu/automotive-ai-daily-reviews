"""Selective trajectory replacement logic.

Paper-sourced fact: trajectories are "replaced only when additional predicted
risk triggers intervention and an alternative satisfies component-wise
constraints on predicted risk and trajectory error" -- i.e. a gating rule:
(a) the nominal trajectory's predicted risk must exceed a threshold tau_risk,
AND (b) a candidate must exist whose own predicted risk is below tau_risk AND
whose L2 deviation from the nominal trajectory is below tau_dev. Only then is
the nominal replaced (with the lowest-risk qualifying candidate). This is
implemented as a plain, testable function -- deliberately NOT hidden inside a
training loop or an nn.Module -- since it is planning-time logic applied to
already-computed risk scores, not a differentiable component.

RECONSTRUCTION NOTE: the exact tie-breaking rule (pick the qualifying
candidate with lowest risk; ties broken by lowest deviation) is this repo's
own default -- the abstract states the gating conditions but not the
tie-break policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SelectionResult:
    """Result of the selective-replacement decision for one batch of scenes.

    replaced: (B,) bool tensor, True where the nominal trajectory was swapped.
    selected_index: (B,) long tensor, index into the candidate dimension K of
        the trajectory actually used (arbitrary/undefined convention -1 is
        used to mean "nominal trajectory kept", handled via `replaced`).
    """

    replaced: torch.Tensor
    selected_index: torch.Tensor


def select_trajectory(
    nominal_risk: torch.Tensor,
    candidate_risk: torch.Tensor,
    candidate_deviation: torch.Tensor,
    tau_risk: float,
    tau_dev: float,
) -> SelectionResult:
    """Decides, per batch element, whether to replace the nominal trajectory.

    Args:
        nominal_risk: (B,) predicted risk score of the nominal trajectory.
        candidate_risk: (B, K) predicted risk score of each alternative
            candidate trajectory.
        candidate_deviation: (B, K) L2 deviation (e.g. meters) of each
            candidate from the nominal trajectory.
        tau_risk: risk threshold -- nominal is only reconsidered if its own
            risk exceeds this, and a replacement candidate must have risk
            strictly below this.
        tau_dev: max allowed deviation of a replacement candidate from the
            nominal trajectory.

    Returns:
        SelectionResult with `replaced` (B,) bool and `selected_index` (B,)
        long (index into K; meaningless/ignored where `replaced` is False).
    """
    B, K = candidate_risk.shape
    device = nominal_risk.device

    nominal_unsafe = nominal_risk > tau_risk                          # (B,) condition (a)

    candidate_ok = (candidate_risk < tau_risk) & (candidate_deviation < tau_dev)  # (B, K) condition (b), component-wise

    # Among qualifying candidates, pick the lowest-risk one. Disqualified
    # candidates are masked out with +inf so they're never selected by argmin.
    masked_risk = torch.where(candidate_ok, candidate_risk, torch.full_like(candidate_risk, float("inf")))
    best_val, best_idx = masked_risk.min(dim=1)                          # (B,), (B,)

    any_candidate_qualifies = torch.isfinite(best_val)                     # (B,)

    replaced = nominal_unsafe & any_candidate_qualifies                      # (B,) both gate conditions
    selected_index = torch.where(
        replaced, best_idx, torch.full_like(best_idx, -1)
    )

    return SelectionResult(replaced=replaced, selected_index=selected_index)
