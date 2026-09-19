"""
FoVAligner: differentiable affine warp of the collaborator's shared
features into the ego's own BEV frame, plus range/FoV masking (the
paper's "FoV filtering"): cells outside the ego's actual field of view
are masked out even after alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FoVAligner(nn.Module):
    def __init__(self, fov_radius_cells: float = 16.0):
        super().__init__()
        self.fov_radius_cells = fov_radius_cells

    @staticmethod
    def _affine_theta(dx_norm: torch.Tensor, dy_norm: torch.Tensor, dtheta: torch.Tensor) -> torch.Tensor:
        """(B,) x3 -> (B, 2, 3) affine theta for F.affine_grid such that,
        for each output (ego-frame) location p, the sampled input
        (collaborator-local-frame) location is R(dtheta) @ p + (dx, dy)
        in normalized [-1, 1] coordinates."""
        cos_t = torch.cos(dtheta)
        sin_t = torch.sin(dtheta)
        theta = torch.stack([
            torch.stack([cos_t, -sin_t, dx_norm], dim=-1),
            torch.stack([sin_t, cos_t, dy_norm], dim=-1),
        ], dim=-2)
        return theta  # (B, 2, 3)

    def compute_grid(self, shape, dx_cells: torch.Tensor, dy_cells: torch.Tensor, dtheta: torch.Tensor = None):
        B, C, H, W = shape
        device = dx_cells.device
        if dtheta is None:
            dtheta = torch.zeros(B, device=device)
        dx_norm = (2.0 * dx_cells / W).to(device)
        dy_norm = (2.0 * dy_cells / H).to(device)
        theta = self._affine_theta(dx_norm, dy_norm, dtheta.to(device))
        return F.affine_grid(theta, shape, align_corners=False)

    def range_mask(self, shape, device, dtype) -> torch.Tensor:
        """(B, 1, H, W) binary mask, 1.0 within fov_radius_cells of the
        ego's own grid center, 0.0 outside -- applied in the ego's OWN
        frame (destination grid), independent of any warp."""
        B, C, H, W = shape
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        dist = torch.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        mask = (dist <= self.fov_radius_cells).to(dtype)
        return mask.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)

    def warp(self, tensor: torch.Tensor, grid: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
        return F.grid_sample(tensor, grid, align_corners=False, mode=mode, padding_mode="zeros")

    def forward(self, features: torch.Tensor, dx_cells: torch.Tensor, dy_cells: torch.Tensor,
                dtheta: torch.Tensor = None):
        """
        Args:
            features: (B, C, H, W) collaborator features (already gated),
                in the collaborator's own local frame
            dx_cells, dy_cells: (B,) the collaborator's local-frame origin
                offset relative to the ego frame, in grid cells (positive
                = collaborator is further along +x/+y than the ego)
            dtheta: (B,) relative rotation in radians (default 0)
        Returns:
            warped: (B, C, H, W) features resampled into the ego's frame
            fov_mask: (B, 1, H, W) binary, 1 = inside ego's trusted FoV
        """
        grid = self.compute_grid(features.shape, dx_cells, dy_cells, dtheta)
        warped = self.warp(features, grid, mode="bilinear")
        fov_mask = self.range_mask(features.shape, features.device, features.dtype)
        return warped, fov_mask

    def warp_mask(self, mask: torch.Tensor, dx_cells: torch.Tensor, dy_cells: torch.Tensor,
                  dtheta: torch.Tensor = None) -> torch.Tensor:
        """Warp a binary mask (e.g. the feature-gate's selection mask)
        with the SAME grid used for the features, so downstream code can
        tell, per ego-frame cell, whether the value sitting there actually
        came from a cell the bandwidth budget let through."""
        grid = self.compute_grid(mask.shape, dx_cells, dy_cells, dtheta)
        return self.warp(mask, grid, mode="nearest")
