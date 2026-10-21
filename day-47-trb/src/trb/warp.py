"""Rigid pose warp of a BEV feature map between two collaborator poses.

Frame convention: x -> columns (W), y -> rows (H), cell centres at (j+0.5)*cell - half.
Output frame = collaborator pose at "now"; input frame = pose at the (stale) send time.
q_in = R(dyaw) q_out + t_in   (all in metres), converted to grid_sample's normalised coords.
"""
import math
import torch
import torch.nn.functional as F


def rigid_warp(feat, rel_pose, half_extent):
    """feat: [B,C,H,W] in the STALE frame. rel_pose: [B,3] = (tx, ty, dyaw) so that
    q_stale = R(dyaw) q_now + (tx, ty). Returns feat resampled into the NOW frame."""
    B = feat.shape[0]
    tx, ty, yaw = rel_pose[:, 0], rel_pose[:, 1], rel_pose[:, 2]
    c, s = torch.cos(yaw), torch.sin(yaw)
    theta = torch.zeros(B, 2, 3, dtype=feat.dtype, device=feat.device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = c, -s, tx / half_extent
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = s, c, ty / half_extent
    grid = F.affine_grid(theta, list(feat.shape), align_corners=False)
    return F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
