"""Flow-guided occupancy evolution module.

Paper-sourced facts (from the alphaxiv.org secondary summary, see README):
  "Flow-guided evolution transports occupancy and scene features, with signed
  residuals correcting occupancy after transport" -- i.e. (1) a predicted 2D
  flow field warps the previous occupancy grid forward, then (2) a separate
  small network predicts a *signed* residual added on top of the warped
  occupancy to correct for whatever the warp gets wrong (new entrants, warp
  artifacts, etc). The abstract also says one forecast is generated per
  planning step and reused across trajectory candidates -- this module is
  called once per step, never once per candidate.

RECONSTRUCTION NOTE (everything below the two bullet facts above is this
repo's own default, not paper-verified): the exact flow-head/residual-head
architectures, the use of `grid_sample` bilinear warping, the tanh-bounded
residual scale, and the autoregressive per-horizon-step loop (with a simple
one-hot step embedding to let the network distinguish which future step it is
predicting) are all this repo's own reconstruction choices.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import make_base_grid


class FlowGuidedOccupancyEvolution(nn.Module):
    """Warps the previous occupancy grid with a predicted flow field, then
    applies a learned signed residual correction to produce the forecast
    occupancy for one future planning step.

    Call this once per planning step (not once per trajectory candidate) --
    the caller is responsible for reusing the resulting forecast across all
    candidates when computing collision scores.
    """

    def __init__(self, in_channels: int, grid_size: int, flow_hidden: int, residual_hidden: int,
                 max_horizon: int, residual_scale: float = 0.5):
        super().__init__()
        self.grid_size = grid_size
        self.residual_scale = residual_scale
        # +max_horizon for a one-hot "which future step" embedding channel,
        # so a single module can be reused autoregressively across steps.
        step_channels = max_horizon
        self.flow_head = nn.Sequential(
            nn.Conv2d(in_channels + step_channels, flow_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(flow_hidden, 2, kernel_size=3, padding=1),  # 2 = (dx, dy)
        )
        # Residual head sees the fused features + the step embedding + the
        # warped occupancy itself, so it can correct based on what the warp
        # actually produced.
        self.residual_head = nn.Sequential(
            nn.Conv2d(in_channels + step_channels + 1, residual_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(residual_hidden, 1, kernel_size=3, padding=1),
        )
        self.max_horizon = max_horizon
        self.register_buffer("_base_grid_cache", torch.zeros(1, grid_size, grid_size, 2), persistent=False)

    def _base_grid(self, device: torch.device) -> torch.Tensor:
        return make_base_grid(self.grid_size, device)

    def warp(self, occupancy: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Bilinearly warps `occupancy` forward by the predicted flow field.

        Args:
            occupancy: (B, 1, H, W) occupancy probability grid.
            flow: (B, 2, H, W) predicted per-cell displacement, in normalized
                  grid_sample units (roughly [-1, 1] scale after the internal
                  damping below).
        Returns:
            warped: (B, 1, H, W) occupancy after flow-guided transport.
        """
        B = occupancy.shape[0]
        base = self._base_grid(occupancy.device).expand(B, -1, -1, -1)  # (B, H, W, 2)
        # Flow is predicted in raw conv-output scale; damp it into a sane
        # normalized-coordinate displacement range (a few cells at most) so
        # training doesn't immediately sample garbage off-grid.
        flow_norm = torch.tanh(flow) * (2.0 / self.grid_size) * 4.0  # cap ~4 cells/step
        flow_norm = flow_norm.permute(0, 2, 3, 1)  # (B, H, W, 2)
        sample_grid = base + flow_norm               # (B, H, W, 2)
        warped = F.grid_sample(
            occupancy, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )  # (B, 1, H, W)
        return warped

    def forward(self, fused_feat: torch.Tensor, prev_occupancy: torch.Tensor, step_idx: int) -> dict:
        """Predicts one future occupancy step from the previous occupancy grid.

        Args:
            fused_feat: (B, C, H, W) fused BEV+actor(+risk-conditioning) features.
            prev_occupancy: (B, 1, H, W) occupancy grid from the previous step
                (t=0 previous observed frame, or the model's own last forecast
                step when called autoregressively).
            step_idx: int in [0, max_horizon), which future step is being
                predicted -- encoded as a one-hot conditioning channel so a
                single module instance handles the whole horizon.
        Returns:
            dict with:
              "flow": (B, 2, H, W) predicted raw flow field,
              "warped_occupancy": (B, 1, H, W),
              "residual": (B, 1, H, W) signed residual (already scaled),
              "occupancy": (B, 1, H, W) final forecast, clamped to [0, 1].
        """
        B, _, H, W = fused_feat.shape
        step_onehot = torch.zeros(B, self.max_horizon, H, W, device=fused_feat.device, dtype=fused_feat.dtype)
        step_onehot[:, step_idx, :, :] = 1.0

        flow_in = torch.cat([fused_feat, step_onehot], dim=1)           # (B, C+max_horizon, H, W)
        flow = self.flow_head(flow_in)                                    # (B, 2, H, W)

        warped = self.warp(prev_occupancy, flow)                            # (B, 1, H, W)

        residual_in = torch.cat([fused_feat, step_onehot, warped], dim=1)     # (B, C+max_horizon+1, H, W)
        residual_raw = self.residual_head(residual_in)                          # (B, 1, H, W)
        residual = torch.tanh(residual_raw) * self.residual_scale                 # signed, bounded correction

        occupancy = torch.clamp(warped + residual, 0.0, 1.0)                        # (B, 1, H, W), valid prob range

        return {
            "flow": flow,
            "warped_occupancy": warped,
            "residual": residual,
            "occupancy": occupancy,
        }
