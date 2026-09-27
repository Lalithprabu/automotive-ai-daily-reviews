"""
Top-down FPN merge + anchor-free (CenterNet-style) detection head.

Design choice (this project's own default, not paper-sourced): the fused
multi-scale features coming out of ChronoFuse are merged top-down (stride16
-> stride8 -> stride4, upsample + add, like a standard FPN) into a single
stride-4 map, which a small head turns into a heatmap (object-center
likelihood per class), a size regression, and a sub-pixel offset
regression — the standard anchor-free detection recipe, chosen because it
keeps the parameter count small and makes the "predicted center position"
directly readable for the latency-compensation evaluation in `simulate.py`
and `train.py`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNMerge(nn.Module):
    def __init__(self, channels_per_scale, out_channels: int):
        super().__init__()
        c4, c8, c16 = channels_per_scale
        self.lat16 = nn.Conv2d(c16, out_channels, kernel_size=1)
        self.lat8 = nn.Conv2d(c8, out_channels, kernel_size=1)
        self.lat4 = nn.Conv2d(c4, out_channels, kernel_size=1)
        self.smooth = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, p4, p8, p16):
        # Shape trace (H, W = the stride-4 map's spatial size):
        t16 = self.lat16(p16)                                         # [B, Cout, H/16, W/16]
        t8 = self.lat8(p8) + F.interpolate(t16, size=p8.shape[-2:], mode="nearest")   # [B, Cout, H/8, W/8]
        t4 = self.lat4(p4) + F.interpolate(t8, size=p4.shape[-2:], mode="nearest")    # [B, Cout, H/4, W/4]
        return self.smooth(t4)                                        # [B, Cout, H/4, W/4]


class CenterDetectionHead(nn.Module):
    """Anchor-free center-point detection head.

    Outputs, all at stride 4 relative to the input event frame:
        heatmap: [B, num_classes, H/4, W/4] — per-class object-center logits
        size:    [B, 2, H/4, W/4]           — (w, h) in stride-4 pixel units
        offset:  [B, 2, H/4, W/4]           — sub-pixel (dx, dy) refinement
    """

    def __init__(self, in_channels: int, num_classes: int = 2, head_channels: int = 64):
        super().__init__()

        def head(out_ch):
            return nn.Sequential(
                nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(head_channels, out_ch, kernel_size=1),
            )

        self.heatmap_head = head(num_classes)
        self.size_head = head(2)
        self.offset_head = head(2)

        # CenterNet convention: bias the heatmap head's last-layer bias
        # negative so training starts from "mostly background", which
        # stabilizes early focal-loss optimization.
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)  # sigmoid(-2.19) ~= 0.10

    def forward(self, feat: torch.Tensor):
        heatmap = self.heatmap_head(feat)   # [B, num_classes, H/4, W/4]
        size = self.size_head(feat)         # [B, 2, H/4, W/4]
        offset = self.offset_head(feat)     # [B, 2, H/4, W/4]
        return heatmap, size, offset
