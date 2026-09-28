"""
TrustRegionCEM -- refines a rough anchor intent into a concrete, scored ego
trajectory using the Cross-Entropy Method (CEM), with an explicit trust-region
penalty that keeps candidates close to the anchor they were spawned from.

Why a trust region at all: the whole efficiency argument of this reconstruction
(and, qualitatively, of the paper) is that ONE cached other-agent prediction,
computed for an anchor's canonical intent, "stays valid across an entire family of
plans" that share that intent. That validity is not unconditional -- it only holds
for candidates that stay reasonably close to the anchor in intent-space. The trust
region penalty makes that assumption explicit and enforced, rather than silently
assumed: candidates that would require the other agent to react very differently
(i.e. stray far from the anchor) are penalized, pushing CEM's search back toward
the region where the cached prediction is trustworthy.

This module supports two query modes, used to build the "Old vs New" ablation in
train.py / simulate.py:
  - mode="cached": the cached other-agent prediction (computed ONCE outside this
    function, by whoever calls refine()) is reused for every candidate in every
    CEM iteration. This is what BOTH "New Way (INTERACT)" and "Old Way A
    (non-interactive)" use -- they differ only in what prediction was cached
    (anchor-conditioned vs. a single unconditional guess).
  - mode="naive": re-queries the predictor once per candidate per iteration,
    conditioning it on that candidate's own (non-canonical) intent vector. This is
    "Old Way B" -- the accurate-but-expensive alternative the paper's efficiency
    claim is about. predictor_calls counts these queries explicitly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from src.utils.kinematics import rollout_trajectory


@dataclass
class CEMResult:
    best_cond: torch.Tensor          # [2] winning candidate intent vector
    best_traj: torch.Tensor          # [T, 2] winning candidate ego trajectory
    best_cost: float                 # scalar cost of the winner
    cost_history: list = field(default_factory=list)   # best cost seen after each iteration
    predictor_calls: int = 0         # number of predictor forward "queries" this refine() call made
    wall_time_s: float = 0.0


def collision_cost(ego_traj: torch.Tensor, other_traj: torch.Tensor,
                    margin: float = 0.35, w_lat: float = 1.0, w_spd: float = 0.3) -> torch.Tensor:
    """Penalizes ego trajectories that get too close to the (predicted) other agent.

    Args:
        ego_traj:   [..., T, 2]  candidate ego (lat, spd) trajectories
        other_traj: [..., T, 2]  other-agent (lat, spd) trajectory, broadcastable
    Returns:
        cost: [...]  summed squared margin violation over T
    """
    diff_lat = ego_traj[..., 0] - other_traj[..., 0]
    diff_spd = ego_traj[..., 1] - other_traj[..., 1]
    dist = torch.sqrt(diff_lat.pow(2) * w_lat + diff_spd.pow(2) * w_spd + 1e-6)   # [..., T]
    violation = torch.relu(margin - dist)
    return violation.pow(2).sum(dim=-1)   # [...]


def comfort_cost(ego_traj: torch.Tensor) -> torch.Tensor:
    """Penalizes jerky (non-smooth) ego trajectories: sum of squared step-to-step deltas."""
    d = ego_traj[..., 1:, :] - ego_traj[..., :-1, :]     # [..., T-1, 2]
    return d.pow(2).sum(dim=(-1, -2))                     # [...]


def progress_cost(ego_traj: torch.Tensor) -> torch.Tensor:
    """Rewards forward progress (penalizes low average speed)."""
    return -ego_traj[..., 1].mean(dim=-1)   # [...]


def trust_region_penalty(cond: torch.Tensor, anchor_center: torch.Tensor,
                          radius: float = 0.35, lam: float = 5.0) -> torch.Tensor:
    """Quadratic penalty once a candidate's intent vector strays past `radius`
    from the anchor it was spawned from -- enforces that CEM's search stays inside
    the region where the cached prediction is assumed valid.

    Args:
        cond: [..., 2] candidate intent vectors
        anchor_center: [2] the anchor's own canonical intent vector
    Returns:
        penalty: [...]
    """
    dist = torch.norm(cond - anchor_center, dim=-1)          # [...]
    return lam * torch.relu(dist - radius).pow(2)


def total_cost(ego_traj: torch.Tensor, other_traj: torch.Tensor, cond: torch.Tensor,
               anchor_center: torch.Tensor, weights: dict, margin: float, radius: float) -> torch.Tensor:
    c_coll = collision_cost(ego_traj, other_traj, margin=margin)
    c_comf = comfort_cost(ego_traj)
    c_prog = progress_cost(ego_traj)
    c_tr = trust_region_penalty(cond, anchor_center, radius=radius, lam=weights["trust_region"])
    return (weights["collision"] * c_coll + weights["comfort"] * c_comf
            + weights["progress"] * c_prog + c_tr)


class TrustRegionCEM:
    def __init__(self, future_len: int, weights: dict, n_iter: int = 4, n_samples: int = 32,
                 elite_frac: float = 0.2, init_std: float = 0.35, bounds=(-1.5, 1.5),
                 collision_margin: float = 0.35, trust_region_radius: float = 0.35,
                 base_speed: float = 1.0):
        self.future_len = future_len
        self.weights = weights
        self.n_iter = n_iter
        self.n_samples = n_samples
        self.elite_frac = elite_frac
        self.init_std = init_std
        self.bounds = bounds
        self.collision_margin = collision_margin
        self.trust_region_radius = trust_region_radius
        self.base_speed = base_speed

    def refine(self, anchor_cond: torch.Tensor, mode: str = "cached",
               cached_pred: torch.Tensor | None = None, predict_fn=None,
               ego_hist: torch.Tensor | None = None, other_hist: torch.Tensor | None = None) -> CEMResult:
        """Runs CEM around one anchor.

        Args:
            anchor_cond: [2] the anchor's canonical intent vector (CEM's search center / trust-region anchor)
            mode: "cached" (reuse `cached_pred` for every candidate/iteration) or
                  "naive" (re-query `predict_fn` once per candidate per iteration)
            cached_pred: [T, 2] required when mode="cached" -- the single other-agent
                         forecast this whole CEM run will reuse.
            predict_fn: callable(ego_hist_batch[N,H,S], other_hist_batch[N,H,S], cond_batch[N,2]) -> [N,T,2],
                        required when mode="naive".
            ego_hist, other_hist: [H, state_dim] required when mode="naive" (to build predict_fn's batch input).

        Returns:
            CEMResult with the best candidate found, its trajectory/cost, the per-iteration
            best-cost history, and how many predictor queries this call made.
        """
        assert mode in ("cached", "naive")
        if mode == "cached":
            assert cached_pred is not None, "mode='cached' requires a precomputed cached_pred"
        else:
            assert predict_fn is not None and ego_hist is not None and other_hist is not None, \
                "mode='naive' requires predict_fn + ego_hist + other_hist"

        t0 = time.time()
        mu = anchor_cond.clone().float()                       # [2]
        sigma = torch.full_like(mu, self.init_std)              # [2]
        lo, hi = self.bounds

        predictor_calls = 0
        cost_history = []
        best_cond_overall, best_traj_overall = mu.clone(), None
        best_cost_overall = float("inf")

        n_elite = max(1, int(round(self.n_samples * self.elite_frac)))

        for _it in range(self.n_iter):
            # --- sample a population of candidate intent vectors around the current mean ---
            noise = torch.randn(self.n_samples, mu.shape[0])
            candidates = mu.unsqueeze(0) + noise * sigma.unsqueeze(0)   # [N, 2]
            candidates = torch.clamp(candidates, lo, hi)

            ego_trajs = rollout_trajectory(candidates, self.future_len, base_speed=self.base_speed)  # [N, T, 2]

            if mode == "cached":
                other_pred = cached_pred.unsqueeze(0).expand(self.n_samples, -1, -1)   # [N, T, 2], reused, 0 new calls
            else:
                ego_hist_batch = ego_hist.unsqueeze(0).expand(self.n_samples, -1, -1)     # [N, H, state_dim]
                other_hist_batch = other_hist.unsqueeze(0).expand(self.n_samples, -1, -1)  # [N, H, state_dim]
                with torch.no_grad():
                    other_pred = predict_fn(ego_hist_batch, other_hist_batch, candidates)   # [N, T, 2]
                predictor_calls += self.n_samples   # ONE conditioning query per candidate -- the expensive path

            costs = total_cost(ego_trajs, other_pred, candidates, anchor_cond,
                                self.weights, self.collision_margin, self.trust_region_radius)   # [N]

            iter_best_idx = torch.argmin(costs)
            iter_best_cost = costs[iter_best_idx].item()
            cost_history.append(iter_best_cost)
            if iter_best_cost < best_cost_overall:
                best_cost_overall = iter_best_cost
                best_cond_overall = candidates[iter_best_idx].clone()
                best_traj_overall = ego_trajs[iter_best_idx].clone()

            # --- CEM update: refit mean/std from the elite fraction ---
            elite_idx = torch.topk(costs, n_elite, largest=False).indices
            elites = candidates[elite_idx]                      # [n_elite, 2]
            mu = elites.mean(dim=0)
            sigma = elites.std(dim=0, unbiased=False) + 1e-3    # variance floor prevents collapse

        wall_time = time.time() - t0
        return CEMResult(best_cond=best_cond_overall, best_traj=best_traj_overall,
                          best_cost=best_cost_overall, cost_history=cost_history,
                          predictor_calls=predictor_calls, wall_time_s=wall_time)
