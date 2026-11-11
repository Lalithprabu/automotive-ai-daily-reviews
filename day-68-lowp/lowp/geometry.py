"""
Calibration-driven geometry used by both LOWP readers (batched, differentiable, torch).

Conventions
-----------
ego frame    : x forward, y left, z up (metres)
camera frame : x right, y down, z forward (OpenCV)
R_n          : (3,3) camera -> ego rotation, t_n: camera centre in ego frame
Projection   : p_cam = R_n^T (p - t_n);  [u, v, 1]^T ~ K_n p_cam   (paper: K_n R_n^T (p - t_n))
grid_sample  : normalised coords g = 2 * u / W - 1  (align_corners=False, u in [0, W])
"""
from __future__ import annotations

import torch


def project_points(p: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor,
                   img_hw: tuple[int, int], min_depth: float = 0.5):
    """Project ego-frame points into every camera.

    Args:
        p:  (B, M, 3) ego-frame points
        K:  (B, N, 3, 3) intrinsics (pixel units of the *image*, not of the feature map)
        R:  (B, N, 3, 3) camera->ego rotations
        t:  (B, N, 3)    camera centres
        img_hw: (H, W) of the image the intrinsics refer to
    Returns:
        grid:  (B, N, M, 2) normalised grid_sample coordinates in [-1, 1] when visible
        valid: (B, N, M)    bool, in front of the camera and inside the image
        depth: (B, N, M)    camera z-depth (metres)
    """
    H, W = img_hw
    d = p[:, None, :, :] - t[:, :, None, :]                          # (B, N, M, 3)
    p_cam = torch.einsum("bnji,bnmj->bnmi", R, d)                     # R^T d  -> (B, N, M, 3)
    z = p_cam[..., 2]
    uvw = torch.einsum("bnij,bnmj->bnmi", K, p_cam)                   # (B, N, M, 3)
    uv = uvw[..., :2] / z.clamp(min=1e-3)[..., None]                  # pixels
    grid = torch.stack([2 * uv[..., 0] / W - 1, 2 * uv[..., 1] / H - 1], -1)
    valid = (z > min_depth) & (grid.abs() <= 1).all(-1)
    return grid, valid, z


def ray_points(K: torch.Tensor, R: torch.Tensor, t: torch.Tensor, img_hw: tuple[int, int],
               feat_hw: tuple[int, int], depths: torch.Tensor) -> torch.Tensor:
    """3D points along each feature cell's viewing ray (the global reader's geometry label).

    For feature cell (i, j) at stride s, the pixel centre is u = (j + 0.5) * W / Wf, v = (i + 0.5) * H / Hf.
    Its ray direction in the camera is K^-1 [u, v, 1]^T; the point at z-depth d is d * K^-1 [u, v, 1]^T,
    mapped to ego by p = R p_cam + t.

    Args:
        K, R: (B, N, 3, 3); t: (B, N, 3); depths: (D,) z-depths in metres
    Returns:
        (B, N, Hf, Wf, D, 3) ego-frame points
    """
    H, W = img_hw
    Hf, Wf = feat_hw
    dev, dt = K.device, K.dtype
    v, u = torch.meshgrid((torch.arange(Hf, device=dev, dtype=dt) + 0.5) * H / Hf,
                          (torch.arange(Wf, device=dev, dtype=dt) + 0.5) * W / Wf, indexing="ij")
    pix = torch.stack([u, v, torch.ones_like(u)], -1)                 # (Hf, Wf, 3)
    ray_cam = torch.einsum("bnij,hwj->bnhwi", torch.linalg.inv(K), pix)  # (B, N, Hf, Wf, 3), z = 1
    p_cam = ray_cam[..., None, :] * depths.view(1, 1, 1, 1, -1, 1)     # (B, N, Hf, Wf, D, 3)
    p_ego = torch.einsum("bnij,bnhwdj->bnhwdi", R, p_cam) + t[:, :, None, None, None, :]
    return p_ego
