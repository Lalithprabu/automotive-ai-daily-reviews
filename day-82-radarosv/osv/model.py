"""PyTorch core of a RadarOSV-style network: two polar radar spectra -> unified Cartesian voxels with occupancy,
dynamic-object semantics and ground-relative velocity.  Re-implementation from the ABSTRACT of arXiv 2611.30078
(Fang, Zhuang, Wei, Jiang, Shi, Julier, Chetty, "RadarOSV: Unified 3D Scene and Motion Perception from mmWave
Radars").  What the abstract states, and how each item maps to this file:

  (1) Polar-to-Cartesian Geometry (PCG) backbone: polar imaging-radar features -> Cartesian voxels with FIXED and
      DEFORMABLE sampling                                                         -> class PCG
  (2) single-chip detections supply class evidence; raw Doppler spectra supply motion features and RADIAL-VELOCITY
      CANDIDATES; the velocity readout SELECTS among candidates to resolve aliasing -> class VelocityReadout
  (3) ego-motion compensation uses RADAR-ESTIMATED velocity, no external odometry -> class EgoVelocityHead
  Everything else (layer sizes, losses, 2D BEV instead of 3D voxels, candidate parameterisation) is my construction.

Shapes (B batch, G = 64 BEV cells per side, C = 32 feature channels, K = 9 aliasing candidates):

    img   (B, 16, 64, 64)  imaging-radar log power, (Doppler, range, azimuth); Doppler wraps every 5 m/s
    sc    (B, 32, 32, 16)  single-chip log power, (Doppler, range, azimuth); +-24 m/s, no wrapping
    polar encoders         Doppler bins are channels -> conv over (range, azimuth)       (B, C, R, A)
    PCG                    grid_sample at each BEV cell's (r, theta) + K_s learned offsets  (B, C, G, G)
    BEV trunk              concat(img_bev, sc_bev, geometry maps, v_ego cos(theta)) -> U-shaped convs (B, 40, G, G)
    heads                  semantics (B, 6, G, G)  [free, static, car, truck, cyclist, pedestrian]
                           wrapped phase (B, 2, G, G)  -> v_w = V_I * atan2(s, c) / pi  in [-2.5, 2.5)
                           candidate logits (B, 9, G, G) -> v_rel = v_w + 2 V_I k,  k in {-4..4}
                           ground-relative radial velocity  v_gr = v_rel + v_ego cos(theta)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .radar import AI, AS, DI, DS, RI, RS, VI
from .scene import FOV, G, GR, GTH, R_MAX

K_CAND = 4                                     # candidates k = -4..4 -> relative radial velocity up to +-22.5 m/s
N_CLS = 6


def conv(ci, co, d=1):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=d, dilation=d), nn.GroupNorm(8, co), nn.GELU())


class PolarEncoder(nn.Module):
    """Doppler bins as input channels; 2D convs over (range, azimuth) stay in the sensor's native polar grid."""

    def __init__(self, c_in, c=32):
        super().__init__()
        self.net = nn.Sequential(conv(c_in, c), conv(c, c))

    def forward(self, x):                      # (B, D, R, A) -> (B, C, R, A)
        return self.net(x)


class PCG(nn.Module):
    """Polar-to-Cartesian Geometry block.

    Fixed sampling: every BEV cell (x, y) has a known polar location (r, theta). We sample the polar feature map there
    with bilinear grid_sample:  u = theta / FOV  (azimuth axis = width),  w = 2 r / R_MAX - 1  (range axis = height).
    Deformable sampling: from the fixed sample, a 1x1 conv predicts n_s offsets (du, dw) and n_s weights per cell; the
    extra samples are softmax-weighted and added as a residual. This lets a cell pull features from neighbouring
    azimuth bins where angular blur smeared its return (the imaging radar's beam is wider far away than near).
    Cells outside the field of view get zeros plus a validity channel.
    """

    def __init__(self, c, n_s=4, deformable=True, max_off=0.08):
        super().__init__()
        r = torch.tensor(GR, dtype=torch.float32)
        th = torch.tensor(GTH, dtype=torch.float32)
        grid = torch.stack([th / FOV, 2 * r / R_MAX - 1], -1)                 # (G, G, 2) (x=azimuth, y=range)
        self.register_buffer("grid", grid[None])
        self.register_buffer("valid", ((th.abs() <= FOV) & (r < R_MAX)).float()[None, None])
        self.deformable, self.n_s, self.max_off = deformable, n_s, max_off
        if deformable:
            self.off = nn.Conv2d(c, 3 * n_s, 1)
            nn.init.zeros_(self.off.weight)
            nn.init.zeros_(self.off.bias)

    def forward(self, f):                      # f (B, C, R, A) -> (B, C, G, G)
        B = f.shape[0]
        g = self.grid.expand(B, -1, -1, -1)
        out = F.grid_sample(f, g, align_corners=False, padding_mode="zeros")    # fixed sampling
        if self.deformable:
            o = self.off(out)                                                     # (B, 3 n_s, G, G)
            d = torch.tanh(o[:, : 2 * self.n_s]) * self.max_off                   # offsets in normalised units
            w = torch.softmax(o[:, 2 * self.n_s:], 1)                             # (B, n_s, G, G)
            acc = 0
            for k in range(self.n_s):
                gk = g + d[:, 2 * k: 2 * k + 2].permute(0, 2, 3, 1)               # (B, G, G, 2)
                acc = acc + w[:, k: k + 1] * F.grid_sample(f, gk, align_corners=False, padding_mode="zeros")
            out = out + acc
        return out * self.valid


class EgoVelocityHead(nn.Module):
    """Radar-estimated ego speed. Static returns satisfy v_rel(theta) = -v_ego cos(theta), so the Doppler-vs-azimuth
    profile encodes v_ego. We average features over range (keeping azimuth) and regress v_ego with an MLP."""

    def __init__(self, c, n_az):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c * n_az, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, f):                      # (B, C, R, A) -> (B,)
        return self.mlp(f.mean(2).flatten(1))[:, 0] * 4.0 + 4.0


class VelocityReadout(nn.Module):
    """Aliasing-aware velocity readout.

    The imaging radar measures v_rel only modulo 2 V_I = 5 m/s. We predict the wrapped value as a phase
    phi = pi v_w / V_I (regressing (cos phi, sin phi) avoids the discontinuity at +-V_I), then build the candidate set
        v_rel^(k) = v_w + 2 V_I k,   k = -K..K
    and a classifier picks k from fused features. With the single-chip branch the classifier sees an unaliased but
    coarse Doppler estimate, so picking k becomes easy; without it, the model can only guess k from priors.
    """

    def __init__(self, c):
        super().__init__()
        self.phase = nn.Conv2d(c, 2, 1)
        self.cand = nn.Sequential(nn.Conv2d(c, c, 1), nn.GELU(), nn.Conv2d(c, 2 * K_CAND + 1, 1))

    def forward(self, h):
        ph = self.phase(h)                                                        # (B, 2, G, G)
        logit = self.cand(h)                                                      # (B, 9, G, G)
        return ph, logit

    @staticmethod
    def decode(ph, logit):
        v_w = VI * torch.atan2(ph[:, 1], ph[:, 0]) / math.pi                      # (B, G, G) in [-V_I, V_I)
        k = logit.argmax(1) - K_CAND                                              # (B, G, G)
        return v_w + 2 * VI * k.float(), v_w, k


class RadarOSVNet(nn.Module):
    """use_sc: add the single-chip branch.  ego: 'estimated' (from radar) or 'external' (given, e.g. odometry)."""

    def __init__(self, use_sc=True, ego="estimated", c=32, w=40, deformable=True):
        super().__init__()
        self.use_sc, self.ego = use_sc, ego
        self.enc_i = PolarEncoder(DI, c)
        self.pcg_i = PCG(c, deformable=deformable)
        self.ego_head = EgoVelocityHead(c * (2 if use_sc else 1), AI)
        if use_sc:
            self.enc_s = PolarEncoder(DS, c)
            self.pcg_s = PCG(c, deformable=deformable)
        cin = c * (2 if use_sc else 1) + 5
        # U-shaped BEV trunk: full-res stem -> stride-2 dilated context (receptive field ~ 40 cells) -> upsample + skip
        self.stem = conv(cin, w)
        self.down = nn.Sequential(nn.Conv2d(w, w, 3, stride=2, padding=1), nn.GroupNorm(8, w), nn.GELU(),
                                  conv(w, w, 2), conv(w, w, 4))
        self.fuse = conv(2 * w, w)
        self.sem = nn.Conv2d(w, N_CLS, 1)
        self.vel = VelocityReadout(w)
        self.register_buffer("geo", torch.tensor(np.stack([np.cos(GTH), np.sin(GTH), GR / R_MAX]),
                                                 dtype=torch.float32)[None])

    def forward(self, img, sc, v_ego_ext=None):
        B = img.shape[0]
        fi = self.enc_i(img)                                                      # (B, C, 64, 64) polar
        bev = [self.pcg_i(fi)]                                                    # (B, C, G, G)
        ego_feat = fi
        if self.use_sc:
            fs = self.enc_s(sc)                                                   # (B, C, 32, 16) polar
            bev.append(self.pcg_s(fs))
            fs_up = F.interpolate(fs, size=fi.shape[-2:], mode="bilinear", align_corners=False)
            ego_feat = torch.cat([fi, fs_up], 1)
        v_hat = self.ego_head(ego_feat)                                           # (B,) radar-estimated ego speed
        v_use = v_ego_ext if (self.ego == "external" and v_ego_ext is not None) else v_hat
        comp = (v_use[:, None, None] * self.geo[:, 0]) / 8.0                      # ego-motion compensation map
        x = torch.cat(bev + [self.geo.expand(B, -1, -1, -1), comp[:, None],
                             (bev[0].abs().sum(1, keepdim=True) > 0).float()], 1)
        h0 = self.stem(x)                                                         # (B, w, G, G)
        h1 = F.interpolate(self.down(h0), scale_factor=2, mode="bilinear", align_corners=False)  # (B, w, G, G)
        h = self.fuse(torch.cat([h0, h1], 1))                                     # (B, w, G, G)
        ph, logit = self.vel(h)
        return dict(sem=self.sem(h), phase=ph, cand=logit, v_ego=v_hat, v_used=v_use)

    def predict(self, img, sc, v_ego_ext=None):
        out = self(img, sc, v_ego_ext)
        v_rel, v_w, k = VelocityReadout.decode(out["phase"], out["cand"])
        v_gr = v_rel + out["v_used"][:, None, None] * self.geo[:, 0]              # ground-relative radial velocity
        return dict(sem=out["sem"].argmax(1), v_rel=v_rel, v_gr=v_gr, k=k, v_ego=out["v_ego"],
                    prob=out["sem"].softmax(1))


def losses(out, sem, vrel, vego, fov, cls_w):
    """sem (B,G,G) long, vrel (B,G,G), vego (B,), fov (G,G) bool.  Velocity terms on occupied cells, dynamic x4."""
    l_sem = (F.cross_entropy(out["sem"], sem, weight=cls_w, reduction="none") * fov).sum() / fov.sum() / len(sem)
    occ = (sem > 0) & fov
    wt = occ.float() * (1 + 3 * (sem >= 2).float())
    phi = math.pi * ((vrel + VI) % (2 * VI) - VI) / VI
    ph = F.normalize(out["phase"], dim=1, eps=1e-6)
    l_ph = ((1 - (ph[:, 0] * torch.cos(phi) + ph[:, 1] * torch.sin(phi))) * wt).sum() / wt.sum().clamp(min=1)
    k = torch.round((vrel - ((vrel + VI) % (2 * VI) - VI)) / (2 * VI)).long().clamp(-K_CAND, K_CAND) + K_CAND
    l_k = (F.cross_entropy(out["cand"], k, reduction="none") * wt).sum() / wt.sum().clamp(min=1)
    l_ego = F.smooth_l1_loss(out["v_ego"], vego)
    tot = l_sem + l_ph + 0.5 * l_k + 0.2 * l_ego
    return tot, dict(sem=l_sem.item(), ph=l_ph.item(), k=l_k.item(), ego=l_ego.item())


def n_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for cfg in (dict(use_sc=False, ego="external"), dict(use_sc=True, ego="estimated")):
        m = RadarOSVNet(**cfg)
        img, sc = torch.randn(2, DI, RI, AI), torch.randn(2, DS, RS, AS)
        o = m.predict(img, sc, torch.tensor([3.0, 5.0]))
        print(cfg, f"{n_params(m):,} params", {k: tuple(v.shape) for k, v in o.items()})
