"""
Output heads for SV-WAM:
  - ActionHead: action tokens -> (x, y, heading) waypoints
  - VideoHead:  future-video tokens -> predicted future latents (training only)
"""
import torch
import torch.nn as nn


class ActionHead(nn.Module):
    """Small MLP projecting action tokens to per-waypoint (x, y, heading)."""

    def __init__(self, embed_dim: int, action_dim: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, action_dim),
        )

    def forward(self, action_tokens: torch.Tensor) -> torch.Tensor:
        # action_tokens: [B, A, D] -> [B, A, action_dim]
        return self.mlp(action_tokens)


class VideoHead(nn.Module):
    """Linear projection from video tokens to predicted future latents (training-only)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, video_tokens: torch.Tensor) -> torch.Tensor:
        # video_tokens: [B, V, D] -> [B, V, D]
        return self.proj(video_tokens)
