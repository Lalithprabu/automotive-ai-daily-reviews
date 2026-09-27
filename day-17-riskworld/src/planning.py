"""Planning-time glue: candidate trajectory generation + a convenience
function that runs collision scoring and selective replacement together.

RECONSTRUCTION NOTE: the candidate generation strategy (lateral offset "swerve"
candidates plus a "slow down" candidate, built by perturbing the nominal
trajectory) is this repo's own default -- the paper's abstract does not
describe how candidates are generated, only that a fixed set of candidates is
scored per planning step.
"""
from __future__ import annotations

import torch

from .collision_score import CollisionScoreModule
from .trajectory_selection import SelectionResult, select_trajectory


def generate_candidate_trajectories(
    nominal_traj: torch.Tensor, num_candidates: int, lateral_offsets: list[float] | None = None
) -> torch.Tensor:
    """Builds a small set of candidate ego trajectories around the nominal one.

    Args:
        nominal_traj: (B, horizon, 2) nominal ego waypoints in grid-cell coords.
        num_candidates: total number of candidates to return (including a
            "slow down" candidate and lateral-offset "swerve" candidates;
            padded/truncated to this count).
        lateral_offsets: cell offsets (perpendicular to the direction of
            nominal motion, approximated here as a fixed +/- row shift since
            our synthetic ego always drives along a horizontal road) to try.
            Defaults to a spread of small and large offsets.
    Returns:
        (B, num_candidates, horizon, 2) candidate trajectories.
    """
    B, horizon, _ = nominal_traj.shape
    device = nominal_traj.device
    if lateral_offsets is None:
        lateral_offsets = [-3.0, -1.5, 1.5, 3.0]

    candidates = []
    # Swerve candidates: shift the row (y) coordinate by a constant lateral offset.
    for off in lateral_offsets:
        cand = nominal_traj.clone()
        cand[..., 1] = cand[..., 1] + off
        candidates.append(cand)

    # Slow-down candidate: hold x progress back to roughly half-speed while
    # keeping the same lane (y unchanged), i.e. a braking maneuver.
    slow = nominal_traj.clone()
    start_x = nominal_traj[:, 0:1, 0] - (nominal_traj[:, 1, 0] - nominal_traj[:, 0, 0])  # crude x at t=0
    for t in range(horizon):
        slow[:, t, 0] = start_x[:, 0] + 0.5 * (nominal_traj[:, t, 0] - start_x[:, 0])
    candidates.append(slow)

    cand_stack = torch.stack(candidates, dim=1)  # (B, len(candidates), horizon, 2)

    if cand_stack.shape[1] >= num_candidates:
        cand_stack = cand_stack[:, :num_candidates]
    else:
        pad = num_candidates - cand_stack.shape[1]
        extra = nominal_traj.unsqueeze(1).expand(-1, pad, -1, -1)
        cand_stack = torch.cat([cand_stack, extra], dim=1)

    return cand_stack.to(device)


def trajectory_l2_deviation(candidates: torch.Tensor, nominal: torch.Tensor) -> torch.Tensor:
    """Mean per-waypoint L2 deviation of each candidate from the nominal trajectory.

    Args:
        candidates: (B, K, horizon, 2)
        nominal: (B, horizon, 2)
    Returns:
        (B, K) mean L2 deviation.
    """
    diff = candidates - nominal.unsqueeze(1)              # (B, K, horizon, 2)
    dist = torch.linalg.norm(diff, dim=-1)                   # (B, K, horizon)
    return dist.mean(dim=-1)                                    # (B, K)


def plan_step(
    model_out: dict,
    nominal_traj: torch.Tensor,
    collision_scorer: CollisionScoreModule,
    num_candidates: int,
    tau_risk: float,
    tau_dev: float,
    cell_meters: float = 1.0,
) -> dict:
    """End-to-end planning-time convenience: generate candidates, score them
    (forecast + persistence + correction), and apply the selective-replacement
    gate.

    Args:
        model_out: the dict returned by `RiskWorld.forward` (needs
            "forecast_occupancy", "persistence_occupancy", "risk_field").
        nominal_traj: (B, horizon, 2) nominal ego trajectory, grid-cell coords.
        collision_scorer: a CollisionScoreModule instance.
        num_candidates: number of alternative candidates to generate.
        tau_risk: risk gate threshold (in the same units as the collision
            score -- an occupancy+risk weighted average in [0, ~2]).
        tau_dev: deviation gate threshold, in grid CELLS (not meters) since
            trajectories here are in cell coordinates; convert externally if
            you need a meters threshold and `cell_meters != 1.0`.
        cell_meters: meters per grid cell (kept for API clarity; deviation is
            computed in cell units by `trajectory_l2_deviation`).
    Returns:
        dict with candidate trajectories, their scores, the nominal's own
        score, and the SelectionResult.
    """
    B = nominal_traj.shape[0]
    candidates = generate_candidate_trajectories(nominal_traj, num_candidates)  # (B, K, horizon, 2)

    all_trajs = torch.cat([nominal_traj.unsqueeze(1), candidates], dim=1)  # (B, 1+K, horizon, 2)
    scores = collision_scorer(
        model_out["forecast_occupancy"], model_out["persistence_occupancy"], model_out["risk_field"], all_trajs
    )
    nominal_risk = scores["forecast_score"][:, 0]        # (B,)
    candidate_risk = scores["forecast_score"][:, 1:]       # (B, K)
    candidate_correction = scores["correction"][:, 1:]       # (B, K)

    deviation = trajectory_l2_deviation(candidates, nominal_traj)  # (B, K)

    result: SelectionResult = select_trajectory(nominal_risk, candidate_risk, deviation, tau_risk, tau_dev)

    return {
        "candidates": candidates,
        "nominal_risk": nominal_risk,
        "candidate_risk": candidate_risk,
        "candidate_correction": candidate_correction,
        "nominal_correction": scores["correction"][:, 0],
        "deviation": deviation,
        "selection": result,
    }
