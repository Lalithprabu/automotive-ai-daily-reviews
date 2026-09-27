"""
ChronoFuseDetector: end-to-end model wiring the event-frame backbone, the
per-scale feature cache, the ChronoFuse causal cross-time fusion module, an
FPN merge, and the center-point detection head.

The critical training-time detail (this is the whole point of the paper):
targets are the ground-truth object positions at t_obs + latency_steps
(a *future* time relative to the last observed input frame), not at
t_obs itself. The model only ever sees inputs up to t_obs; it must predict
where objects will be once its own output is actually available.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.backbone import EventFrameEncoder
from src.models.chronofuse import MultiScaleChronoFuse
from src.models.detection_head import FPNMerge, CenterDetectionHead
from src.utils.latency_embed import LatencyEmbedding


class ChronoFuseDetector(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        backbone_channels=(32, 64, 128),
        fpn_channels: int = 64,
        num_classes: int = 2,
        cache_len: int = 4,
        latency_dim: int = 32,
        max_latency: float = 8.0,
    ):
        super().__init__()
        self.cache_len = cache_len

        self.backbone = EventFrameEncoder(in_channels, backbone_channels)
        self.latency_embed = LatencyEmbedding(latency_dim, max_latency)
        self.chronofuse = MultiScaleChronoFuse(
            channels_per_scale=self.backbone.out_channels,
            latency_dim=latency_dim,
            cache_len=cache_len,
        )
        self.fpn = FPNMerge(self.backbone.out_channels, fpn_channels)
        self.head = CenterDetectionHead(fpn_channels, num_classes)

        n_params = sum(p.numel() for p in self.chronofuse.parameters())
        self._chronofuse_param_count = n_params  # exposed for the README's param-budget claim

    def encode_sequence(self, event_seq: torch.Tensor):
        """Runs the backbone over every timestep of an input sequence.

        Args:
            event_seq: [B, T, 2, H, W] — T observed event frames, oldest
                first, most recent last (index T-1).

        Returns:
            current_feats: tuple of [B, C_i, H_i, W_i] for the *last*
                timestep (T-1) at each of the 3 pyramid scales.
            cached_feats: tuple of [B, K, C_i, H_i, W_i] — the K frames
                immediately *before* the last one (T-2 ... T-1-K), oldest
                first, zero-padded at the start of short sequences.
        """
        b, t, c_in, h, w = event_seq.shape
        # Run the (shared-weight) backbone independently per timestep.
        per_step_feats = [self.backbone(event_seq[:, i]) for i in range(t)]

        current_feats = per_step_feats[-1]  # tuple of 3 scales, each [B, C, H, W]

        k = self.cache_len
        cached_feats = []
        for scale_idx in range(3):
            steps = [per_step_feats[i][scale_idx] for i in range(max(0, t - 1 - k), t - 1)]
            if len(steps) < k:
                pad = [torch.zeros_like(current_feats[scale_idx]) for _ in range(k - len(steps))]
                steps = pad + steps
            cached_feats.append(torch.stack(steps, dim=1))  # [B, K, C, H, W]
        return current_feats, tuple(cached_feats)

    def forward(
        self,
        event_seq: torch.Tensor,
        latency_steps: torch.Tensor,
        enable_chronofuse: bool = True,
    ):
        """
        Args:
            event_seq: [B, T, 2, H, W]
            latency_steps: [B] float — requested "frames ahead" to predict.
            enable_chronofuse: when False, the ChronoFuse fusion is skipped
                entirely and only the last observed frame's raw features
                are used — this is this repo's "old way" ablation baseline
                (a standard event detector with *no* latency compensation),
                reusing the exact same trained weights.

        Returns:
            heatmap, size, offset — see `CenterDetectionHead`.
        """
        current_feats, cached_feats = self.encode_sequence(event_seq)

        if enable_chronofuse:
            latency_embed = self.latency_embed(latency_steps)
            fused_feats = self.chronofuse(current_feats, cached_feats, latency_embed)
        else:
            fused_feats = current_feats

        merged = self.fpn(*fused_feats)
        heatmap, size, offset = self.head(merged)
        return heatmap, size, offset

    @property
    def chronofuse_param_count(self) -> int:
        return self._chronofuse_param_count
