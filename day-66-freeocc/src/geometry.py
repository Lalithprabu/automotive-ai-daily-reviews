"""SE(2) helpers shared by the model, the synthetic world and the tests.

Conventions (everything is metric, right-handed): x = forward, y = left, yaw counter-clockwise.
BEV tensors are (B, C, G, G) with  dim-2 index i <-> x (forward)  and  dim-3 index j <-> y (left):
    x = (i + 0.5 - G/2) * CELL ,  y = (j + 0.5 - G/2) * CELL   (ego sits at the grid centre).
"""
import torch
import torch.nn.functional as F

G = 48          # grid cells per side
CELL = 1.0      # metres per cell  -> 48 m x 48 m window


def rel_pose(a, b):
    """Pose of frame b expressed in frame a.  a, b: (..., 3) = [x, y, yaw] in the world. Returns (..., 3)."""
    d = b[..., :2] - a[..., :2]
    c, s = torch.cos(a[..., 2]), torch.sin(a[..., 2])
    x = c * d[..., 0] + s * d[..., 1]
    y = -s * d[..., 0] + c * d[..., 1]
    return torch.stack([x, y, b[..., 2] - a[..., 2]], -1)


def warp_bev(feat, pose_b_in_a, mode="bilinear"):
    """Resample a BEV map that lives in frame A so that it lives in frame B (the SE(2) 'ego-motion removal').

    feat: (N, C, G, G) in frame A ; pose_b_in_a: (N, 3) = pose of B in A.
    out(q) = feat( R(yaw) q + t )  for every output cell centre q (metres, frame B).  Bilinear, zero padding.
    """
    n = feat.shape[0]
    half = G * CELL / 2.0
    x, y, th = pose_b_in_a[:, 0] / half, pose_b_in_a[:, 1] / half, pose_b_in_a[:, 2]
    c, s = torch.cos(th), torch.sin(th)
    # affine_grid works on (g_j, g_i) = (y, x) normalised coordinates, see module docstring
    theta = torch.stack([torch.stack([c, s, y], -1), torch.stack([-s, c, x], -1)], 1)   # (N, 2, 3)
    grid = F.affine_grid(theta, list(feat.shape), align_corners=False)
    return F.grid_sample(feat, grid, mode=mode, padding_mode="zeros", align_corners=False)


def cell_centres(device="cpu"):
    """(G, G, 2) metric (x, y) of every cell centre in the ego frame."""
    idx = (torch.arange(G, device=device).float() + 0.5 - G / 2) * CELL
    xx, yy = torch.meshgrid(idx, idx, indexing="ij")
    return torch.stack([xx, yy], -1)


def inv_pose(p):
    """Invert a relative pose: pose of b in a  ->  pose of a in b.  p: (..., 3)."""
    c, s = torch.cos(p[..., 2]), torch.sin(p[..., 2])
    x = -(c * p[..., 0] + s * p[..., 1])
    y = -(-s * p[..., 0] + c * p[..., 1])
    return torch.stack([x, y, -p[..., 2]], -1)
