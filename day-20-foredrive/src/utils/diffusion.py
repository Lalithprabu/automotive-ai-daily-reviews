"""
Shared diffusion-schedule utilities for the DiT trajectory planner.

We use a standard DDPM linear beta schedule for TRAINING (random timestep,
epsilon-prediction MSE — see models/foredrive.py `forward_train`) and DDIM
(eta=0, deterministic) for SAMPLING at inference, since DDIM naturally
supports skipping steps (e.g. 15 sampling steps out of a 50-step training
schedule) without needing a re-derived variance schedule. This is a standard,
well-known simplification for small diffusion demos and is documented here
explicitly as our own choice — the ForeDrive paper's exact sampler is not
known to us (see SOURCING.md).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch


def make_beta_schedule(num_steps: int, beta_start: float, beta_end: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear beta schedule, per DDPM (Ho et al. 2020).

    Returns:
      betas      : (num_steps,)
      alphas      : (num_steps,)       = 1 - betas
      alpha_bars   : (num_steps,)       = cumprod(alphas)   ("alpha_bar_t")
    """
    betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal position/timestep embedding (as in Transformer /
    DDPM). timesteps: (B,) integer or float tensor. Returns (B, dim)."""
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=device) / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)   # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)      # (B, 2*half)
    if emb.shape[-1] < dim:  # pad if dim is odd
        emb = torch.nn.functional.pad(emb, (0, dim - emb.shape[-1]))
    return emb


def q_sample(x0: torch.Tensor, t: torch.Tensor, alpha_bars: torch.Tensor,
             noise: torch.Tensor) -> torch.Tensor:
    """Forward diffusion: sample x_t ~ q(x_t | x0) in closed form.
    x0: (B, T, 2), t: (B,) int64 indices into alpha_bars, noise: same shape as x0."""
    ab = alpha_bars.to(x0.device)[t].view(-1, 1, 1)   # (B, 1, 1)
    return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise


def ddim_respaced_timesteps(num_diffusion_steps: int, num_sampling_steps: int) -> List[int]:
    """Evenly-spaced subset of {0, ..., num_diffusion_steps-1}, descending, used
    to walk the reverse process in fewer steps than it was trained with."""
    num_sampling_steps = min(num_sampling_steps, num_diffusion_steps)
    step_indices = torch.linspace(0, num_diffusion_steps - 1, num_sampling_steps).round().long()
    step_indices = sorted(set(step_indices.tolist()), reverse=True)
    return step_indices


def ddim_step(x_t: torch.Tensor, eps_pred: torch.Tensor, t: int, t_prev: int,
              alpha_bars: torch.Tensor) -> torch.Tensor:
    """One deterministic DDIM (eta=0) reverse step from timestep `t` to `t_prev`
    (t_prev < t, or t_prev == -1 for the final step landing on x0).

    x_t, eps_pred: (B, T, 2)
    """
    device = x_t.device
    ab_t = alpha_bars.to(device)[t]
    ab_prev = alpha_bars.to(device)[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

    x0_pred = (x_t - torch.sqrt(1.0 - ab_t) * eps_pred) / torch.sqrt(ab_t)
    x_prev = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1.0 - ab_prev) * eps_pred
    return x_prev
