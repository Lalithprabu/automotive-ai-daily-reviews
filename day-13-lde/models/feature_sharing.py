"""
AdaptiveFeatureGate: bandwidth-budgeted top-k SPATIAL gating.

The paper's "adaptation-oriented feature sharing" -- selects which BEV
cells' feature vectors get shared over the (bandwidth-limited) V2X link.
"""

import torch
import torch.nn as nn


class AdaptiveFeatureGate(nn.Module):
    """
    Deliberately purely spatial: it decides WHICH cells survive the
    bandwidth budget and zeroes out the rest. It has NO learnable
    parameters and must never reproject the surviving cells through any
    channel-mixing operator (e.g. an untrained 1x1 "channel_reduce" conv).

    Bug #3: an earlier version of this module ran gated features through
    such an untrained conv before the teacher's head decoded them. That
    conv's random weights scrambled the feature basis the (already
    EMA-trained) teacher cls_head expects, collapsing pseudo-confidence
    even when the teacher was genuinely confident pre-gating. The fix is
    this module: selection + zeroing only, nothing learned, nothing that
    mixes channels.
    """

    def __init__(self, budget_ratio: float = 0.4):
        super().__init__()
        assert 0.0 < budget_ratio <= 1.0
        self.budget_ratio = budget_ratio

    @staticmethod
    def _saliency(features: torch.Tensor) -> torch.Tensor:
        """Per-cell L2 norm across channels: an unlearned proxy for 'how
        much signal is in this cell', computed only from feature values
        already present -- no extra parameters.
        features: (B, C, H, W) -> (B, H, W)"""
        return features.norm(p=2, dim=1)

    def forward(self, features: torch.Tensor, budget_ratio: float = None):
        """
        Args:
            features: (B, C, H, W) collaborator feature map
        Returns:
            gated_features: (B, C, H, W), non-selected cells zeroed
            mask: (B, 1, H, W) binary, 1.0 = selected/shared, 0.0 = dropped
        """
        ratio = self.budget_ratio if budget_ratio is None else budget_ratio
        B, C, H, W = features.shape
        num_cells = H * W
        k = max(1, int(round(num_cells * ratio)))

        saliency = self._saliency(features).reshape(B, -1)          # (B, HW)
        topk_idx = saliency.topk(k, dim=1).indices                   # (B, k)

        mask_flat = torch.zeros(B, num_cells, device=features.device, dtype=features.dtype)
        mask_flat.scatter_(1, topk_idx, 1.0)
        mask = mask_flat.reshape(B, 1, H, W)

        gated_features = features * mask  # purely spatial multiply, no conv
        return gated_features, mask
