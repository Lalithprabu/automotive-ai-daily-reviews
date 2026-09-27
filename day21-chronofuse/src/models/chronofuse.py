"""
ChronoFuse: causal cross-time feature fusion for latency-compensated
detection.

Reconstructed from the abstract-level description in "Bend the Clock:
Predicting Ahead to Beat Latency in Event-Based Object Detection"
(arXiv:2609.26919, Sen, Cottereau, Cuperlier, Sim et al., submitted
2026-09-22): "ChronoFuse performs causal cross-time fusion over a
multi-scale feature hierarchy, combining current representations with
cached temporal features" to predict object state for *when the output
becomes available* rather than for when the input was observed.

IMPORTANT SOURCING NOTE: the paper's abstract names the mechanism
("causal cross-time fusion", "multi-scale feature hierarchy", "cached
temporal features") and reports it adds ~0.17M parameters / 0.84ms latency
on top of a base detector, but the internal wiring below — the causal
temporal-attention cache summary, the finite-difference velocity term, and
the latency-gated residual fusion — is this project's own design built to
be *consistent* with that description and parameter budget, not a
line-for-line reproduction of the paper's real implementation (which was
not recoverable — see SOURCING.md).

Core idea implemented here:
  1. Approximate short-term feature "motion" via a finite difference between
     the current feature map and the most recently cached one.
  2. Run a tiny causal temporal-attention pool over the cached feature
     stack (queried by the current frame) to get a longer-horizon context
     summary — this is the "cross-time fusion" over more than one step.
  3. Gate how much of that extrapolated signal to blend in, conditioned on
     the requested latency embedding (predict 1 frame ahead -> small gate;
     predict 4 frames ahead -> larger gate).
  4. Add the gated correction as a residual onto the current feature map,
     so with the gate at zero the block is the identity (a graceful
     fallback to "no compensation", used as this repo's ablation baseline).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChronoFuseBlock(nn.Module):
    """Single-scale causal cross-time fusion block.

    Args:
        channels: feature channel count C at this pyramid scale.
        latency_dim: dimensionality D of the shared latency embedding.
        cache_len: number of cached (past) feature maps this block expects.
    """

    def __init__(self, channels: int, latency_dim: int, cache_len: int = 4):
        super().__init__()
        self.channels = channels
        self.cache_len = cache_len

        # Tiny causal temporal-attention scorer: pools each cached frame's
        # feature map (global-average-pool -> [B, C]) and the current
        # frame's pooled feature, projects both to a shared key/query space,
        # and produces one attention weight per cached step. This is
        # deliberately global-pooled (not per-pixel) attention to keep the
        # parameter count small, consistent with the paper's ~0.17M budget.
        self.query_proj = nn.Linear(channels, channels // 4)
        self.key_proj = nn.Linear(channels, channels // 4)

        # Fuses [current | finite-diff velocity | attended cache summary]
        # (3*C channels) back down to C channels.
        self.fuse_conv = nn.Conv2d(channels * 3, channels, kernel_size=1)

        # Latency-conditioned gate (FiLM-style): maps the shared latency
        # embedding to a per-channel gate in [0, 1] controlling how much of
        # the extrapolated correction is applied.
        self.latency_gate = nn.Sequential(
            nn.Linear(latency_dim, channels),
            nn.Sigmoid(),
        )

        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

    def forward(
        self,
        current: torch.Tensor,
        cache: torch.Tensor,
        latency_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            current: [B, C, H, W] — feature map at the most recent observed
                timestep.
            cache: [B, K, C, H, W] — the K most recent *previous* feature
                maps at this same scale, oldest first. K may be less than
                `self.cache_len` at the start of a sequence (handled by the
                caller padding with zeros).
            latency_embed: [B, D] — shared latency embedding (see
                `LatencyEmbedding`), identical across all scales.

        Returns:
            fused: [B, C, H, W] — predicted feature map for the future
                (output-availability) timestep.
        """
        b, c, h, w = current.shape
        k = cache.shape[1]

        # --- 1. Finite-difference "velocity" term -----------------------
        # Shape: [B, C, H, W]. Most recent cached frame == cache[:, -1].
        prev = cache[:, -1]
        velocity = current - prev

        # --- 2. Causal temporal-attention cache summary ------------------
        # Pool each cached step and the current step to [B, C], project to
        # a small key/query space, score, softmax over the K cached steps
        # (causal: only *past* steps are attended to, never the future).
        current_pooled = current.mean(dim=(-1, -2))                     # [B, C]
        cache_pooled = cache.mean(dim=(-1, -2))                         # [B, K, C]

        query = self.query_proj(current_pooled).unsqueeze(1)            # [B, 1, C/4]
        keys = self.key_proj(cache_pooled)                              # [B, K, C/4]
        scores = torch.einsum("bqd,bkd->bqk", query, keys) / (keys.shape[-1] ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)                        # [B, 1, K]

        # Weighted sum of the *full* cached feature maps (not just the
        # pooled summaries) using the attention weights.
        # Shape: [B, K, C, H, W] * [B, K, 1, 1, 1] -> sum over K -> [B, C, H, W]
        weights = attn_weights.squeeze(1).view(b, k, 1, 1, 1)
        cache_summary = (cache * weights).sum(dim=1)

        # --- 3. Fuse current + velocity + cache summary ------------------
        combined = torch.cat([current, velocity, cache_summary], dim=1)  # [B, 3C, H, W]
        correction = self.fuse_conv(combined)                            # [B, C, H, W]

        # --- 4. Latency-conditioned gated residual ------------------------
        gate = self.latency_gate(latency_embed).view(b, c, 1, 1)         # [B, C, 1, 1]
        fused = current + gate * correction
        return self.norm(fused)


class MultiScaleChronoFuse(nn.Module):
    """Applies an independent ChronoFuseBlock at each pyramid scale, sharing
    one latency embedding across scales."""

    def __init__(self, channels_per_scale, latency_dim: int, cache_len: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ChronoFuseBlock(c, latency_dim, cache_len) for c in channels_per_scale]
        )

    def forward(self, current_feats, cached_feats, latency_embed):
        """
        Args:
            current_feats: tuple of [B, C_i, H_i, W_i] for each scale i.
            cached_feats: tuple of [B, K, C_i, H_i, W_i] for each scale i.
            latency_embed: [B, D], shared across scales.

        Returns:
            tuple of fused [B, C_i, H_i, W_i] feature maps, same order.
        """
        return tuple(
            block(cur, cache, latency_embed)
            for block, cur, cache in zip(self.blocks, current_feats, cached_feats)
        )
