"""
Sinusoidal diffusion-timestep embedding, DDPM/flow-matching convention.
"""
import math

import torch
import torch.nn.functional as F


def sinusoidal_time_embed(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    t:   [B]  flow-matching timestep in [0, 1] (0 = pure noise, 1 = clean data,
              matching the paper's x-prediction convention)
    Returns: [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )                                                   # [half]
    args = t.float()[:, None] * freqs[None, :] * 1000.0  # [B, half] (scale like DDPM convention)
    embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, 2*half]
    if dim % 2:
        embed = F.pad(embed, (0, 1))                     # odd dim -> pad one zero column
    return embed                                          # [B, dim]
