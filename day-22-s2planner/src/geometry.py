"""
Pinhole camera geometry utilities.

S2Planner's core innovation (per the paper abstract) is projecting trajectory
waypoints into the camera image plane using known geometry, then sampling
multi-scale image features at those projected locations ("camera-projected
cross-attention"). To make that mechanism real (not a stand-in), this module
implements an actual pinhole projection for a 3-camera rig (left / center /
right, matching the paper's "three front-facing cameras").

Coordinate convention (ego / BEV frame):
    x: forward (meters)
    y: left    (meters)
    z: up      (meters, usually 0 for ground-plane points)

This is this project's own reconstruction default, not paper-sourced.
"""
from __future__ import annotations
import math
import torch


def make_camera_rig(
    yaws_deg=(-32.0, 0.0, 32.0),
    focal_px: float = 60.0,
    image_size: int = 64,
    cam_height_m: float = 1.4,
):
    """
    Build a rig of `len(yaws_deg)` pinhole cameras mounted at the ego origin,
    each yawed by `yaws_deg[i]` degrees from the ego forward axis (positive =
    turned toward +y / left, matching a right-handed z-up frame).

    Returns a dict of tensors describing the rig, consumed by `project_points`.
    """
    yaws = torch.tensor([math.radians(d) for d in yaws_deg], dtype=torch.float32)
    n_cam = yaws.shape[0]

    # Rotation from ego frame -> camera frame for each camera (yaw only).
    # Camera looks down its own +x axis; image-plane u is along camera -y,
    # image-plane v is along camera -z (down), which is the usual pinhole
    # convention once we swap axes below.
    cos_y = torch.cos(yaws)
    sin_y = torch.sin(yaws)
    # R rotates a world/ego vector into the camera's forward-facing frame.
    R = torch.stack(
        [
            torch.stack([cos_y, sin_y], dim=-1),
            torch.stack([-sin_y, cos_y], dim=-1),
        ],
        dim=-2,
    )  # (n_cam, 2, 2), acts on (x, y)

    K = focal_px
    return {
        "n_cam": n_cam,
        "yaws_deg": torch.tensor(yaws_deg, dtype=torch.float32),
        "R": R,  # (n_cam, 2, 2) planar rotation for (x, y)
        "focal_px": torch.tensor(float(K)),
        "image_size": image_size,
        "cam_height_m": torch.tensor(float(cam_height_m)),
    }


def project_points(rig: dict, points_xyz: torch.Tensor):
    """
    Project 3D ego-frame points into every camera of the rig.

    Args:
        rig: output of `make_camera_rig`.
        points_xyz: (..., 3) points in ego frame (x fwd, y left, z up).

    Returns:
        pixel_uv: (n_cam, ..., 2) pixel coordinates (u right, v down), in
            [0, image_size). Not clamped -- may fall outside the image.
        valid: (n_cam, ...) bool mask, True where the point is in front of
            that camera (local x > eps) i.e. a real, well-defined projection.
    """
    n_cam = rig["n_cam"]
    R = rig["R"]  # (n_cam, 2, 2)
    f = rig["focal_px"]
    S = rig["image_size"]
    cam_h = rig["cam_height_m"]

    xy = points_xyz[..., :2]  # (..., 2)
    z = points_xyz[..., 2] - cam_h  # camera mounted at cam_height above ground

    # Broadcast rotation across all leading dims of points.
    # xy: (..., 2) -> (1, ..., 2); R: (n_cam, 2, 2)
    xy_exp = xy.unsqueeze(0)  # (1, ..., 2)
    # local_x = R[...,0,0]*x + R[...,0,1]*y ; local_y = R[...,1,0]*x + R[...,1,1]*y
    R0 = R[:, 0, :].reshape(n_cam, *([1] * (xy_exp.dim() - 2)), 2)
    R1 = R[:, 1, :].reshape(n_cam, *([1] * (xy_exp.dim() - 2)), 2)
    local_x = (xy_exp * R0).sum(-1)  # forward distance in camera frame, (n_cam, ...)
    local_y = (xy_exp * R1).sum(-1)  # lateral offset in camera frame, (n_cam, ...)
    local_z = z.unsqueeze(0).expand_as(local_x)

    eps = 1e-3
    valid = local_x > eps
    safe_x = torch.clamp(local_x, min=eps)

    u = S / 2.0 - f * (local_y / safe_x)  # lateral -> horizontal pixel (left = +y -> smaller u)
    v = S / 2.0 - f * (local_z / safe_x)  # height -> vertical pixel (up = +z -> smaller v)

    pixel_uv = torch.stack([u, v], dim=-1)
    return pixel_uv, valid


def normalize_pixels_for_grid_sample(pixel_uv: torch.Tensor, image_size: int) -> torch.Tensor:
    """Map pixel coords in [0, image_size) to grid_sample's [-1, 1] range."""
    return (pixel_uv / (image_size - 1)) * 2.0 - 1.0
