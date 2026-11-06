"""Ego-motion warping of BEV tensors (the 'ego state evolution' half of EvoOcc)."""
import torch
import torch.nn.functional as F


def ego_theta(ego_delta: torch.Tensor, extent_m: float) -> torch.Tensor:
    """Build the 2x3 affine matrices used by ``F.affine_grid``.

    ego_delta: [B, 3] = (dx, dy, dyaw) -- pose of the CURRENT ego frame expressed in the
               PREVIOUS ego frame (metres, metres, radians).
    extent_m:  full side length of the (square) BEV grid in metres.

    For an output cell at p_cur (current frame), its source location in the previous
    frame is  p_prev = R(dyaw) @ p_cur + t.  In normalised [-1, 1] coordinates the
    translation is divided by half the grid extent.   -> theta: [B, 2, 3]
    """
    dx, dy, yaw = ego_delta[:, 0], ego_delta[:, 1], ego_delta[:, 2]
    c, s = torch.cos(yaw), torch.sin(yaw)
    half = extent_m / 2.0
    theta = torch.stack(
        [torch.stack([c, -s, dx / half], dim=-1),
         torch.stack([s, c, dy / half], dim=-1)], dim=1)
    return theta


def warp_bev(x: torch.Tensor, ego_delta: torch.Tensor, extent_m: float,
             mode: str = "bilinear") -> torch.Tensor:
    """Warp a previous-frame BEV tensor [B, C, H, W] into the current ego frame.
    Cells that fall outside the previous field of view become 0 ('unknown')."""
    theta = ego_theta(ego_delta, extent_m).to(x.dtype)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=False)
