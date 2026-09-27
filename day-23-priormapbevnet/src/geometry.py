"""
Camera-rig and BEV-grid geometry utilities.

This project's own reconstruction default (not paper-sourced): the source
paper (arXiv:2609.26325) does not publish its camera rig, voxel size, or
BEV grid resolution in the abstract-level material this project could
retrieve (see SOURCING.md). The pinhole model, grid convention, and all
numeric defaults below are this repo's own choice, picked to keep a small
synthetic scene fully visible to all three cameras.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class BEVGrid:
    """Ego-centric bird's-eye-view grid.

    The ego vehicle sits at grid row `ego_row`, centered in x. Grid cell
    (r, c) covers world coordinates:
        x_world = (c - grid_w / 2 + 0.5) * cell_size
        y_world = (ego_row - r + 0.5) * cell_size   (y grows forward)
    """

    grid_h: int = 32
    grid_w: int = 32
    cell_size: float = 1.0  # meters / cell
    ego_row: int = 24  # ego sits near the bottom of the grid (most range is ahead)

    def cell_centers(self) -> torch.Tensor:
        """Returns world-frame (x, y) center of every BEV cell.

        Shape: [grid_h, grid_w, 2]
        """
        rows = torch.arange(self.grid_h).float()
        cols = torch.arange(self.grid_w).float()
        y = (self.ego_row - rows + 0.5) * self.cell_size  # [grid_h]
        x = (cols - self.grid_w / 2 + 0.5) * self.cell_size  # [grid_w]
        yy = y.view(-1, 1).expand(self.grid_h, self.grid_w)
        xx = x.view(1, -1).expand(self.grid_h, self.grid_w)
        return torch.stack([xx, yy], dim=-1)  # [H, W, 2]

    def world_to_cell(self, xy: torch.Tensor) -> torch.Tensor:
        """World (x, y) -> continuous (row, col) grid coordinates.

        xy: [..., 2] -> returns [..., 2] as (row, col), float (not rounded).
        """
        x, y = xy[..., 0], xy[..., 1]
        col = x / self.cell_size + self.grid_w / 2 - 0.5
        row = self.ego_row - (y / self.cell_size - 0.5)
        return torch.stack([row, col], dim=-1)


@dataclass
class PinholeCamera:
    """A single pinhole camera mounted on the ego vehicle (own reconstruction default)."""

    name: str
    yaw_deg: float  # 0 = forward, +90 = left, -90 = right (right-handed, y-forward)
    height: float = 1.4  # meters, mount height above ground
    focal_px: float = 60.0  # px-equivalent focal length (wide, dashcam-like FOV)
    img_h: int = 48
    img_w: int = 64
    fov_deg: float = 100.0  # horizontal field of view used for visibility gating

    def world_to_pixel(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project ground-plane world points into this camera's image plane.

        xy: [..., 2] world (x, y), y = forward from ego origin.
        Returns (u, v, valid) each shaped [...]; valid = in front of camera and
        within the horizontal FOV.
        """
        yaw = math.radians(self.yaw_deg)
        cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
        x, y = xy[..., 0], xy[..., 1]
        # Rotate into camera-facing frame (camera looks down its own +y')
        xc = x * cos_y - y * sin_y
        yc = x * sin_y + y * cos_y
        valid = yc > 0.5
        u = self.focal_px * (xc / yc.clamp(min=1e-3)) + self.img_w / 2
        v = self.focal_px * (self.height / yc.clamp(min=1e-3)) + self.img_h / 2
        half_fov = math.radians(self.fov_deg / 2)
        ang = torch.atan2(xc, yc)
        valid = valid & (ang.abs() < half_fov)
        return u, v, valid


def default_camera_rig() -> list[PinholeCamera]:
    """Front / left / right synthetic camera trio (this repo's own default)."""
    return [
        PinholeCamera(name="front", yaw_deg=0.0),
        PinholeCamera(name="left", yaw_deg=75.0),
        PinholeCamera(name="right", yaw_deg=-75.0),
    ]
