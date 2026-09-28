"""
Shared kinematic rollout used by three different places in this reconstruction:
  1. the synthetic data simulator (to generate the ego's own deterministic future
     given the intent it committed to),
  2. the Trust-Region CEM refiner (to turn a *candidate* intent vector into a
     concrete ego waypoint trajectory that can be scored), and
  3. simulate.py (to render the ego's committed plan).

Keeping ONE rollout function guarantees the planner and the data generator agree
on what a given `cond` vector physically means.

State representation (deliberately simplified, this project's own choice, not the
paper's): every trajectory is a sequence of 2D vectors (lat, spd) where
  lat = lateral offset from the current lane center (unitless, positive = toward
        the merge target)
  spd = longitudinal speed (unitless, positive = forward)

`cond` = (target_lat_offset, target_speed_delta) is exactly the same 2D vector
used both as an "anchor" (a canonical point in this space) and as a raw
conditioning input to the predictor -- this symmetry is what lets the "naive
per-candidate re-query" baseline (Old Way B) reuse the very same predictor by
simply feeding it a candidate's own (non-canonical) cond vector instead of an
anchor's.
"""
from __future__ import annotations

import torch


def rollout_trajectory(cond: torch.Tensor, future_len: int, base_speed: float = 1.0) -> torch.Tensor:
    """Deterministic kinematic rollout of an ego intent vector into a waypoint trajectory.

    Args:
        cond: [..., 2] tensor, (target_lat_offset, target_speed_delta). Leading dims
              are batched freely (e.g. [N, 2] for a CEM population).
        future_len: T, number of future steps to roll out.
        base_speed: starting longitudinal speed the ramp departs from.

    Returns:
        traj: [..., T, 2] tensor of (lat, spd) waypoints. lat ramps smoothly
              (smoothstep, so it starts/ends with ~zero lateral rate -- "comfortable")
              from 0 to target_lat_offset; spd ramps linearly from base_speed to
              base_speed + target_speed_delta.
    """
    lead_shape = cond.shape[:-1]
    target_lat = cond[..., 0]   # [...]
    target_spd_delta = cond[..., 1]  # [...]

    t = torch.linspace(1.0 / future_len, 1.0, future_len, device=cond.device, dtype=cond.dtype)  # [T]
    # smoothstep ramp in [0,1]: 3t^2 - 2t^3  -> smooth accel/decel of lateral motion
    smooth_ramp = 3 * t.pow(2) - 2 * t.pow(3)                          # [T]
    linear_ramp = t                                                     # [T]

    # broadcast: [..., 1] * [T] -> [..., T]
    lat_traj = target_lat.unsqueeze(-1) * smooth_ramp.reshape(*([1] * len(lead_shape)), future_len)
    spd_traj = base_speed + target_spd_delta.unsqueeze(-1) * linear_ramp.reshape(*([1] * len(lead_shape)), future_len)

    traj = torch.stack([lat_traj, spd_traj], dim=-1)   # [..., T, 2]
    return traj


def build_history(lat0: float, spd0: float, history_len: int, noise_std: float = 0.02,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """Builds a near-constant-velocity past history ending at the present (t=0).

    Returns:
        hist: [H, 4] tensor, columns = (lat, spd, dlat, dspd) where dlat/dspd are
              first-order finite differences (0 at the first timestep).
    """
    kw = {} if generator is None else {"generator": generator}
    spd = spd0 + noise_std * torch.randn(history_len, **kw)
    lat = lat0 + noise_std * 0.5 * torch.randn(history_len, **kw)
    dlat = torch.cat([torch.zeros(1), lat[1:] - lat[:-1]])
    dspd = torch.cat([torch.zeros(1), spd[1:] - spd[:-1]])
    hist = torch.stack([lat, spd, dlat, dspd], dim=-1)   # [H, 4]
    return hist


def assertiveness(cond: torch.Tensor) -> torch.Tensor:
    """Scalar 'how assertive is this committed intent' signal used ONLY by the
    synthetic ground-truth simulator to decide how the other agent reacts.
    The predictor never sees this scalar directly -- it only sees `cond` and must
    learn the mapping implicitly. This is this project's own reconstruction rule,
    not something the paper specifies.
    """
    return 0.5 * cond[..., 0] + 0.5 * cond[..., 1]
