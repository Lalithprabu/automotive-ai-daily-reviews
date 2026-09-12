"""
Adaptive LayerNorm (AdaLN-Zero style) conditioning, used by every
PlanningExpertLayer to inject (diffusion timestep + instruction + ego-state).
"""
import torch
import torch.nn as nn


class AdaLNModulation(nn.Module):
    """
    Maps a conditioning vector c (time + instruction + ego-state, already
    summed/concatenated upstream) into six per-channel modulation vectors:
    (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp).

    Gates are zero-initialized (AdaLN-Zero) so each Planning Expert layer
    starts as an identity function and training is stable from step 0.
    """

    def __init__(self, cond_dim: int, model_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * model_dim),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, cond: torch.Tensor):
        # cond: [B, cond_dim] -> six chunks of [B, model_dim]
        out = self.proj(cond)                             # [B, 6*D]
        return out.chunk(6, dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: [B, T, D], shift/scale: [B, D] -> broadcast over the T (token) axis.
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
