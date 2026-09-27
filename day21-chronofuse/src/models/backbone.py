"""
Lightweight multi-scale event-frame encoder.

Real event-camera pipelines (e.g. the 1 Mpx automotive detection setting
ChronoFuse's source paper evaluates on) bin asynchronous events into dense
"event frames" / voxel grids (typically 2 polarity channels: positive and
negative brightness-change counts per pixel-time-bin) before feeding a
standard CNN. We reconstruct that convention here: each timestep's input is
a [B, 2, H, W] event frame, and the backbone produces a 3-level feature
pyramid, which is what ChronoFuse's "multi-scale feature hierarchy" fuses
across time.

This backbone is this project's own reconstruction default — the paper
discloses that ChronoFuse itself is a small add-on module, not the backbone,
so backbone capacity/design was not paper-sourced.
"""

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        nn.SiLU(inplace=True),
    )


class EventFrameEncoder(nn.Module):
    """Produces a 3-scale feature pyramid from a single event frame.

    Scales (relative to input H, W):
        P4:  stride 4  -> channels[0]
        P8:  stride 8  -> channels[1]
        P16: stride 16 -> channels[2]
    """

    def __init__(self, in_channels: int = 2, channels=(32, 64, 128)):
        super().__init__()
        c1, c2, c3 = channels

        # Shape: [B, 2, H, W] -> [B, c1, H/2, W/2] -> [B, c1, H/4, W/4]
        self.stem = nn.Sequential(
            conv_block(in_channels, c1, stride=2),
            conv_block(c1, c1, stride=2),
        )
        # Shape: [B, c1, H/4, W/4] -> [B, c2, H/8, W/8]
        self.stage2 = conv_block(c1, c2, stride=2)
        # Shape: [B, c2, H/8, W/8] -> [B, c3, H/16, W/16]
        self.stage3 = conv_block(c2, c3, stride=2)

        self.out_channels = (c1, c2, c3)

    def forward(self, event_frame: torch.Tensor):
        """
        Args:
            event_frame: [B, 2, H, W]

        Returns:
            (p4, p8, p16): feature maps at stride 4 / 8 / 16, channel counts
            as given by `self.out_channels`.
        """
        p4 = self.stem(event_frame)      # [B, c1, H/4,  W/4]
        p8 = self.stage2(p4)             # [B, c2, H/8,  W/8]
        p16 = self.stage3(p8)            # [B, c3, H/16, W/16]
        return p4, p8, p16
