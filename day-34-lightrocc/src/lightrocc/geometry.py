"""Pinhole camera + voxel-grid geometry shared by the data generator and the model.

Coordinate frames (ego, right-handed): x forward, y LEFT, z up. Camera sits at (0, 0, cam_height)
looking along +x. Image coordinates (u, v): u right, v down, in *feature-map pixels*.
"""
import math
import torch


def voxel_centers(cfg):
    """[X*Y*Z, 3] world coordinates of voxel centers, flattened in (x, y, z) C-order."""
    w = cfg["world"]
    xs = w["x_min"] + (torch.arange(w["grid_x"]) + 0.5) * w["cell"]
    ys = w["y_min"] + (torch.arange(w["grid_y"]) + 0.5) * w["cell"]
    zs = (torch.arange(w["grid_z"]) + 0.5) * w["cell_z"]
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    return torch.stack([gx, gy, gz], -1).reshape(-1, 3)


def project(xyz, cfg):
    """World points [...,3] -> (u, v, depth). depth = forward distance x (optical axis)."""
    c = cfg["camera"]
    x = xyz[..., 0].clamp(min=1e-3)
    u = c["w"] / 2.0 - c["focal"] * xyz[..., 1] / x
    v = c["h"] / 2.0 + c["focal"] * (c["cam_height"] - xyz[..., 2]) / x
    return u, v, xyz[..., 0]


def unproject(u, v, depth, cfg):
    """Forward lifting: pixel (u, v) + depth along optical axis -> world (x, y, z). Differentiable."""
    c = cfg["camera"]
    x = depth
    y = -(u - c["w"] / 2.0) * depth / c["focal"]
    z = c["cam_height"] - (v - c["h"] / 2.0) * depth / c["focal"]
    return torch.stack([x, y, z], -1)


def rot2(yaw):
    """[...,2,2] rotation matrices for yaw angles."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
