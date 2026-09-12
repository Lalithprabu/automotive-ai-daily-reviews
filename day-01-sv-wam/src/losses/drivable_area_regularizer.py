"""
DrivableAreaRegularizer -- differentiable penalty that keeps all 4 predicted
vehicle-footprint corners inside a rasterized BEV drivable-area mask.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.geometry import vehicle_corners_to_world, sample_sdf_at_points


class DrivableAreaRegularizer(nn.Module):
    """
    Penalizes predicted trajectories whose vehicle footprint drifts outside
    the drivable area, using a precomputed signed-distance field (SDF) of
    the BEV drivable-area mask so the penalty is smooth and differentiable
    (a hard mask lookup would have zero gradient almost everywhere).

    sdf_map convention: negative inside the drivable area, positive outside,
    magnitude = distance in meters to the nearest boundary.
    """

    def __init__(self, vehicle_length: float = 4.8, vehicle_width: float = 1.9):
        super().__init__()
        self.length = vehicle_length
        self.width = vehicle_width
        # Footprint corners in the vehicle's own local frame, centered on
        # the rear axle: (front-left, front-right, rear-left, rear-right).
        hw, hl = vehicle_width / 2.0, vehicle_length / 2.0
        self.register_buffer(
            "local_corners",
            torch.tensor(
                [[hl, hw], [hl, -hw], [-hl, hw], [-hl, -hw]]
            ),  # [4, 2]  (x_forward, y_left)
        )

    def forward(self, trajectory: torch.Tensor, sdf_map: torch.Tensor,
                map_resolution: float, map_origin: torch.Tensor) -> torch.Tensor:
        """
        trajectory:   [B, T, 3]      predicted (x, y, heading) waypoints, meters
        sdf_map:      [B, 1, H, W]   per-sample signed-distance field, meters
        map_resolution: scalar, meters per pixel
        map_origin:   [B, 2]         world (x, y) of pixel (0, 0)
        """
        corners = vehicle_corners_to_world(trajectory, self.local_corners)   # [B, T, 4, 2]
        sampled_sdf = sample_sdf_at_points(sdf_map, corners, map_resolution, map_origin)  # [B, T, 4]

        # Only positive SDF (outside drivable area) incurs a penalty; ReLU
        # keeps the loss zero (and gradient-quiet) for fully compliant corners.
        violation = F.relu(sampled_sdf)                           # [B, T, 4]
        return violation.pow(2).mean()                            # scalar penalty
