"""
Three parallel heads reading the shared BEV feature F_t, plus the pairwise
ranking machinery that is DRiF's core contribution.

  - StaticMapHead    : F_t -> per-cell map class logits (drivable / boundary / bg)
  - DynamicRiskHead   : F_t -> dense scalar risk field R_hat (paper's core head)
  - PlanningHead       : F_t -> future ego waypoints

`pairwise_ranking_loss` and `sample_risk_at_points` are reproduced close to
verbatim from the verified reference implementation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Module-level constant: BEV window is [-BEV_RANGE_M/2, +BEV_RANGE_M/2] meters
# in both x (forward) and y (lateral), centered on the ego vehicle.
BEV_RANGE_M = 50.0


class UpsampleBlock(nn.Module):
    """2x spatial upsample + conv + norm + activation.

    Uses nearest-neighbor upsample followed by a 3x3 conv (rather than a
    transposed conv) to avoid the checkerboard artifacts that strided
    deconvolutions are prone to -- relevant here because the dynamic risk
    head is exactly the branch we found rendering as checkerboard noise
    before the TV-smoothness fix (see DynamicRiskHead / compute_losses).

    Input / Output: [B, C_in, H, W] -> [B, C_out, 2H, 2W]
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x)
        return self.act(x)


class StaticMapHead(nn.Module):
    """Psi_map: F_t -> per-cell static map segmentation logits.

    Input:  f_t     # Shape: [B, C_in, H/4, W/4]
    Output: logits   # Shape: [B, num_classes, H, W]
    """

    def __init__(self, in_channels: int = 128, num_classes: int = 3):
        super().__init__()
        self.up1 = UpsampleBlock(in_channels, in_channels // 2)
        self.up2 = UpsampleBlock(in_channels // 2, in_channels // 4)
        self.head = nn.Conv2d(in_channels // 4, num_classes, kernel_size=1)

    def forward(self, f_t: torch.Tensor) -> torch.Tensor:
        x = self.up1(f_t)
        x = self.up2(x)
        return self.head(x)


class DynamicRiskHead(nn.Module):
    """Psi_risk: F_t -> ego-conditioned dynamic risk field R_hat in R^{H x W}.
    No sigmoid: the pairwise ranking loss only needs correct *ordering*, not
    a calibrated absolute scale.
    """

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.up1 = UpsampleBlock(in_channels, in_channels // 2)
        self.up2 = UpsampleBlock(in_channels // 2, in_channels // 4)
        self.head = nn.Conv2d(in_channels // 4, 1, kernel_size=1)

    def forward(self, f_t: torch.Tensor) -> torch.Tensor:
        x = self.up1(f_t)
        x = self.up2(x)
        risk = self.head(x)
        return risk.squeeze(1)


class PlanningHead(nn.Module):
    """Psi_plan: F_t -> future ego trajectory waypoints, read off the same
    shared feature as the map and risk heads (single-BEV-feature design).

    Input:  f_t          # Shape: [B, C_in, H/4, W/4]
    Output: waypoints    # Shape: [B, horizon, 2]   (x_forward, y_lateral) meters
    """

    def __init__(self, in_channels: int = 128, horizon: int = 8, out_dim: int = 2):
        super().__init__()
        self.horizon = horizon
        self.out_dim = out_dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.SiLU(inplace=True),
            nn.Linear(in_channels, in_channels // 2),
            nn.SiLU(inplace=True),
            nn.Linear(in_channels // 2, horizon * out_dim),
        )

    def forward(self, f_t: torch.Tensor) -> torch.Tensor:
        B = f_t.shape[0]
        pooled = self.pool(f_t).flatten(1)          # [B, C_in]
        out = self.mlp(pooled)                        # [B, horizon * out_dim]
        return out.view(B, self.horizon, self.out_dim)


def pairwise_ranking_loss(
    r_hat: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    y_ij: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """L_rank = mean( max(0, margin - y_ij * (r_i - r_j)) ), a pairwise hinge
    ranking loss -- only penalizes pairs whose predicted risk *ordering*
    disagrees with the priority-classed label, by at least `margin`. Pairs
    labeled y_ij = 0 (statistically indistinguishable risk) contribute
    nothing, correctly encoding "no preference."

    Input:  r_hat    # Shape: [B, N]   predicted risk at each of N sampled points
            idx_i     # Shape: [B, P]   index into N for point i of each pair
            idx_j     # Shape: [B, P]   index into N for point j of each pair
            y_ij      # Shape: [B, P]   in {+1, 0, -1}
    Output: scalar loss
    """
    r_i = torch.gather(r_hat, dim=1, index=idx_i)
    r_j = torch.gather(r_hat, dim=1, index=idx_j)
    hinge = F.relu(margin - y_ij * (r_i - r_j))
    return hinge.mean()


def sample_risk_at_points(risk_map: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample a dense [B, H, W] risk map at world-coordinate points.
    Input:  risk_map    # Shape: [B, H, W]
            points_xy    # Shape: [B, N, 2]   (x_forward, y_lateral) meters, ego-centered
    Output: sampled       # Shape: [B, N]
    """
    B, H, W = risk_map.shape
    norm_x = (points_xy[..., 1] / (BEV_RANGE_M / 2)).clamp(-1, 1)
    norm_y = (points_xy[..., 0] / (BEV_RANGE_M / 2)).clamp(-1, 1)
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(1)
    sampled = F.grid_sample(
        risk_map.unsqueeze(1), grid, mode="bilinear", align_corners=False, padding_mode="border"
    )
    return sampled.view(B, -1)
