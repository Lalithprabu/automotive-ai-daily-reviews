"""
Latency conditioning embedding.

ChronoFuse must know *how far ahead* it needs to extrapolate before it can
decide how aggressively to trust the causal cross-time fusion signal versus
the raw current-frame features. We encode the requested latency (expressed
in "frames ahead" — a continuous, not just integer, quantity since real
pipeline latency is a continuous jitter, not a fixed number of ticks) with a
standard sinusoidal positional encoding followed by a 2-layer MLP, in the
same spirit as diffusion-timestep embeddings.

This module is deliberately tiny: the paper reports ChronoFuse adds only
~0.17M parameters over a base detector, so every sub-component here is kept
narrow (a handful of linear layers, no attention) to stay consistent with
that budget while still giving each scale's fusion block a channel-wise
latency signal to condition on.
"""

import math

import torch
import torch.nn as nn


class LatencyEmbedding(nn.Module):
    """Encodes a scalar 'frames-ahead' latency value into a feature vector.

    Args:
        embed_dim: output embedding dimensionality (D).
        max_latency: the largest latency value (in frames) the sinusoidal
            table is tuned for; latencies are normalized against this.
    """

    def __init__(self, embed_dim: int = 32, max_latency: float = 8.0):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even for sin/cos pairing"
        self.embed_dim = embed_dim
        self.max_latency = max_latency

        half_dim = embed_dim // 2
        # Standard geometric frequency schedule (as in Transformer positional
        # encodings / diffusion timestep embeddings).
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs, persistent=False)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, latency_steps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latency_steps: Tensor of shape [B], float, "frames ahead" the
                model must predict for (the estimated pipeline latency
                expressed in units of the input frame interval).

        Returns:
            Tensor of shape [B, embed_dim].
        """
        # Shape: [B] -> [B, 1] -> [B, half_dim]
        x = latency_steps.float().unsqueeze(-1) * self.freqs.unsqueeze(0)
        # Shape: [B, embed_dim]
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return self.mlp(emb)
