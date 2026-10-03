"""Synthetic road-scene generator + pinhole camera for the MapLightning reconstruction.

World frame: x forward (m), y left (m), z up. Everything here is this repo's own synthetic
setup (NOT the paper's nuScenes / Argoverse 2 data).
"""
import math
import torch
import torch.nn.functional as F

X_MIN, X_MAX, Y_HALF = 6.0, 36.0, 10.0     # mapped region in front of the ego car
N_PTS = 20                                  # points per polyline
IMG_H, IMG_W = 48, 96
NOM = dict(h=1.5, pitch=math.radians(8.0), yaw=0.0, fx=60.0)   # nominal calibration
CLS_DIVIDER, CLS_BOUNDARY, CLS_NONE = 0, 1, 2
N_LINES = 4                                 # left bnd, right bnd, divider A, divider B (optional)


def camera_basis(pitch, yaw):
    """pitch/yaw: (B,) radians (pitch positive = looking down). Returns forward, right, down: (B,3)."""
    f = torch.stack([torch.cos(pitch) * torch.cos(yaw), torch.cos(pitch) * torch.sin(yaw), -torch.sin(pitch)], -1)
    up = torch.tensor([0., 0., 1.], dtype=f.dtype).expand_as(f)
    r = F.normalize(torch.cross(f, up, dim=-1), dim=-1)
    d = torch.cross(f, r, dim=-1)
    return f, r, d


def project(pts, h, pitch, yaw, fx=NOM["fx"], W=IMG_W, H=IMG_H):
    """pts (B,N,3) world -> pixel (u,v) (B,N) each, plus depth. Pinhole, cx=W/2, cy=H/2."""
    f, r, d = camera_basis(pitch, yaw)
    p = pts - torch.stack([torch.zeros_like(h), torch.zeros_like(h), h], -1)[:, None, :]
    zc = (p * f[:, None]).sum(-1)
    zs = zc.clamp(min=1e-2)
    u = W / 2 + fx * (p * r[:, None]).sum(-1) / zs
    v = H / 2 + fx * (p * d[:, None]).sum(-1) / zs
    return u, v, zc


def sample_camera(B, s, gen=None):
    """Extrinsic perturbation at severity s (degrees of pitch/yaw std; height std = 0.02*s m)."""
    g = dict(generator=gen)
    return dict(
        h=NOM["h"] + 0.02 * s * torch.randn(B, **g),
        pitch=NOM["pitch"] + math.radians(s) * torch.randn(B, **g),
        yaw=NOM["yaw"] + math.radians(s) * torch.randn(B, **g))


def nominal_camera(B):
    return dict(h=torch.full((B,), NOM["h"]), pitch=torch.full((B,), NOM["pitch"]), yaw=torch.full((B,), NOM["yaw"]))


def sample_scene(B, gen=None):
    """Returns polys (B,4,N,2) metres, cls (B,4) long, present (B,4) bool."""
    u = lambda lo, hi, *sh: lo + (hi - lo) * torch.rand(*sh, generator=gen)
    xs = torch.linspace(X_MIN, X_MAX, N_PTS)
    dx = xs - X_MIN
    a, b, off, w = u(-.08, .08, B, 1), u(-.0015, .0015, B, 1), u(-1., 1., B, 1), u(5., 7., B, 1)
    yc = off + a * dx + b * dx ** 2                                   # (B,N)
    offs = torch.cat([w / 2, -w / 2, u(-1.2, 0., B, 1), u(.8, 2., B, 1)], 1)   # (B,4)
    polys = torch.stack([xs.expand(B, -1).clone().unsqueeze(1).expand(B, 4, -1),
                         yc[:, None, :] + offs[:, :, None]], -1)       # (B,4,N,2)
    cls = torch.tensor([CLS_BOUNDARY, CLS_BOUNDARY, CLS_DIVIDER, CLS_DIVIDER]).expand(B, -1).clone()
    present = torch.ones(B, 4, dtype=torch.bool)
    present[:, 3] = torch.rand(B, generator=gen) < 0.5
    return polys, cls, present


def _splat(img, u, v, wgt, ch):
    """Bilinear splat points into img (B,C,H,W). u,v,wgt: (B,P); ch: (B,P) long."""
    B, C, H, W = img.shape
    flat = img.view(-1)
    bidx = torch.arange(B)[:, None].expand_as(u)
    u0, v0 = u.floor(), v.floor()
    for du in (0, 1):
        for dv in (0, 1):
            uu, vv = u0 + du, v0 + dv
            w = wgt * (1 - (u - uu).abs()) * (1 - (v - vv).abs())
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            idx = ((bidx * C + ch) * H + vv.clamp(0, H - 1).long()) * W + uu.clamp(0, W - 1).long()
            flat.index_put_((idx[ok],), w[ok], accumulate=True)
    return img


_BLUR = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.


def render(polys, cls, present, cam, noise=0.04, clutter=True, gen=None):
    """Render a synthetic 3-channel camera image (ch0 dividers, ch1 boundaries, ch2 road texture)."""
    B, L, N, _ = polys.shape
    dense = F.interpolate(polys.reshape(B * L, N, 2).transpose(1, 2), size=160, mode="linear",
                          align_corners=True).transpose(1, 2).reshape(B, L * 160, 2)
    pts = torch.cat([dense, torch.zeros(B, L * 160, 1)], -1)
    u, v, zc = project(pts, cam["h"], cam["pitch"], cam["yaw"])
    ch = (cls == CLS_BOUNDARY).long()[:, :, None].expand(B, L, 160).reshape(B, -1)
    wgt = present[:, :, None].expand(B, L, 160).reshape(B, -1).float() * (zc > 0.5).float() * 0.5
    img = torch.zeros(B, 3, IMG_H, IMG_W)
    _splat(img, u, v, wgt, ch)
    img = F.conv2d(img, _BLUR.expand(3, 1, 3, 3).contiguous(), padding=1, groups=3).clamp(0, 1) * 1.8
    img = img.clamp(0, 1)
    tex = F.interpolate(torch.rand(B, 1, 6, 12, generator=gen), size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)
    img[:, 2:3] = 0.3 * tex
    if clutter:   # random bright blobs in the line channels = false-positive bait
        for _ in range(3):
            cu = torch.rand(B, 1, generator=gen) * IMG_W
            cv = torch.rand(B, 1, generator=gen) * IMG_H
            cc = (torch.rand(B, 1, generator=gen) < .5).long()
            blob = torch.zeros(B, 3, IMG_H, IMG_W)
            _splat(blob, cu, cv, torch.full_like(cu, 1.0), cc)
            img = (img + F.conv2d(blob, _BLUR.expand(3, 1, 3, 3).contiguous(), padding=1, groups=3) * 1.5).clamp(0, 1)
    return (img + noise * torch.randn(B, 3, IMG_H, IMG_W, generator=gen)).clamp(0, 1)


def make_batch(B, s=1.5, gen=None):
    polys, cls, present = sample_scene(B, gen)
    cam = sample_camera(B, s, gen)
    return dict(img=render(polys, cls, present, cam, gen=gen), polys=polys, cls=cls, present=present, cam=cam)
