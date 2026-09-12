"""
Geometry helpers shared by the drivable-area regularizer:
  - projecting the vehicle's local footprint corners into world coordinates
    given a predicted (x, y, heading) trajectory
  - sampling a per-sample signed-distance field (SDF) at those world points
"""
import torch
import torch.nn.functional as F


def vehicle_corners_to_world(trajectory: torch.Tensor, local_corners: torch.Tensor) -> torch.Tensor:
    """
    trajectory:    [B, T, 3]  (x, y, heading) at each predicted step, meters/radians
    local_corners: [4, 2]     footprint corners in the vehicle's own local frame
                              (x_forward, y_left), centered on the rear axle

    Returns:
        world corners: [B, T, 4, 2]
    """
    x, y, theta = trajectory[..., 0], trajectory[..., 1], trajectory[..., 2]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)

    lc = local_corners.view(1, 1, 4, 2)
    rot_x = lc[..., 0] * cos_t.unsqueeze(-1) - lc[..., 1] * sin_t.unsqueeze(-1)
    rot_y = lc[..., 0] * sin_t.unsqueeze(-1) + lc[..., 1] * cos_t.unsqueeze(-1)

    world_x = rot_x + x.unsqueeze(-1)      # [B, T, 4]
    world_y = rot_y + y.unsqueeze(-1)      # [B, T, 4]
    return torch.stack([world_x, world_y], dim=-1)   # [B, T, 4, 2]


def sample_sdf_at_points(
    sdf_map: torch.Tensor,
    points_xy: torch.Tensor,
    map_resolution: float,
    map_origin: torch.Tensor,
) -> torch.Tensor:
    """
    Bilinearly samples a per-batch signed-distance field at a set of world
    (x, y) points via grid_sample.

    sdf_map:       [B, 1, H, W]   per-sample SDF, meters (negative = inside
                                  the drivable area, positive = outside)
    points_xy:     [B, T, K, 2]   world (x, y) points to sample (e.g. T
                                  timesteps x K=4 footprint corners)
    map_resolution: scalar, meters per pixel
    map_origin:    [B, 2]        world (x, y) of pixel (0, 0)

    Returns:
        sampled values: [B, T, K]
    """
    px = (points_xy[..., 0] - map_origin[:, None, None, 0]) / map_resolution
    py = (points_xy[..., 1] - map_origin[:, None, None, 1]) / map_resolution
    h, w = sdf_map.shape[-2:]
    gx = (px / (w - 1)) * 2 - 1
    gy = (py / (h - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)                     # [B, T, K, 2]
    b, t, k, _ = grid.shape
    grid = grid.reshape(b, t * k, 1, 2)                      # grid_sample wants [B, Hg, Wg, 2]

    sampled = F.grid_sample(sdf_map, grid, mode="bilinear", align_corners=True)  # [B, 1, T*K, 1]
    return sampled.view(b, t, k)                              # [B, T, K]
