"""
euler_sample -- the 10-step Euler ODE solver used at inference to turn
Gaussian noise into a final trajectory, matching the paper's deployment
setup.
"""
from typing import Optional

import torch

from src.models.planning_expert import PlanningExpert
from src.models.vlm_kv_cache import VLMKVCache


@torch.no_grad()
def euler_sample(
    model: PlanningExpert,
    kv_cache: VLMKVCache,
    instruction_embed: torch.Tensor,
    ego_state_embed: torch.Tensor,
    n_waypoints: int,
    n_steps: int = 10,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Integrates the flow-matching ODE dx/dt = x1_pred(x_t, t) - x_t from
    t=0 (pure noise) to t=1 (clean trajectory) using n_steps Euler steps,
    matching the paper's 10-step Euler solver at deployment.
    """
    b = instruction_embed.shape[0]
    device = device or instruction_embed.device
    x_t = torch.randn(b, n_waypoints, 3, device=device)   # [B, T, 3] start from noise
    dt = 1.0 / n_steps

    for step in range(n_steps):
        t_val = torch.full((b,), step * dt, device=device)
        x1_pred = model(x_t, t_val, kv_cache, instruction_embed, ego_state_embed)
        velocity = x1_pred - x_t          # rectified-flow velocity field
        x_t = x_t + dt * velocity          # Euler update

    return x_t   # [B, T, 3] final predicted (x, y, heading) trajectory
