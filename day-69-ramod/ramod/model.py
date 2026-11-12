"""
RaMOD - Moving-Object Detection from Raw mmWave Radar Cubes under Ego Motion (arXiv 2611.13271)
Toy PyTorch re-implementation of the CORE pipeline described in the abstract:

    raw RAD cubes (T frames) --> [1] per-frame ego-motion estimate
                             --> [2] predict + suppress the Doppler returns of STATIC structure
                             --> [3] multi-scale temporal features
                             --> [4] class heat-map / box detection
                             --> [5] per object: radial + forward velocity under the raw-Doppler and the
                                     Doppler-unwrapped hypotheses, plus a wrapping probability (>= 0.5 -> unwrapped)

Paper facts followed (arXiv abs + alphaXiv overview of the full text): frame-wise translational ego velocity
from each unmasked cube; a GLOBAL static-consistent Doppler mask that includes wrapped copies (not a
zero-Doppler cut); T = 3 suppressed cubes encoded together with their differences from the target frame;
multi-scale features -> box + class; two velocity hypotheses + wrap probability, threshold 0.5.
Everything else (layer sizes, the Hough-voting ego estimator, the soft multiplicative mask, range-azimuth
CenterNet boxes instead of 3-D RAD boxes) is MY toy design.  It is not the authors' code.

Physics used
------------
Radar at the ego origin, forward speed v_e, yaw rate w.  A static point p (range r, azimuth theta) has
relative velocity (-v_e + w*y, -w*x); its radial component is

        v_r = [(-v_e + w y) x - w x y] / r = -v_e * cos(theta)          (yaw rate cancels!)

and the radar measures it folded:   v_meas = wrap(v_r) = ((v_r + V) mod 2V) - V,   V = v_max.
So every azimuth column of the cube has its static clutter on ONE predictable Doppler bin, and the
shape of that curve across azimuth (cos(theta) spans 0.5..1 over +-60 deg) pins down v_e even when
the clutter itself is wrapped.

Shape legend: B batch, T frames, R range bins, A azimuth bins, D Doppler bins, C channels, K classes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def wrap_t(v: torch.Tensor, vmax: float) -> torch.Tensor:
    """Fold radial velocity into [-vmax, vmax)."""
    return torch.remainder(v + vmax, 2 * vmax) - vmax


def conv_bn(i: int, o: int, s: int = 1, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


# =========================================================================== [1] ego motion
class EgoMotionHough(nn.Module):
    """Per-frame ego-speed estimate from the cube by learned Hough voting along the static-Doppler curve.

    1. Collapse range:            X (B, R, A, D) -> azimuth-Doppler map  (B, 2, A, D)  [mean, max over R]
    2. 1x1/3x3 convs learn a "static-clutter evidence" map E (B, 1, A, D).
    3. For each candidate speed v_j on a grid (0..v_hi), sample E along the curve
       d_a(v_j) = wrap(-v_j cos(theta_a)) for every azimuth a and sum  ->  vote(v_j)       (B, J)
    4. soft-argmax over the votes -> v_e (differentiable), plus the vote logits for a CE loss.
    """

    def __init__(self, s_centers: torch.Tensor, vmax: float, v_hi: float = 8.5, n_grid: int = 86):
        super().__init__()
        self.vmax = vmax
        self.register_buffer("cos_a", torch.sqrt(1 - s_centers ** 2))                 # (A,)
        self.register_buffer("v_grid", torch.linspace(0.0, v_hi, n_grid))             # (J,)
        self.evidence = nn.Sequential(conv_bn(2, 16), conv_bn(16, 16), nn.Conv2d(16, 1, 1))
        self.temp = nn.Parameter(torch.tensor(4.0))                                    # vote sharpness

    def forward(self, cube: torch.Tensor):
        """cube (B, R, A, D) -> v_e (B,), logits (B, J), evidence (B, A, D)."""
        B, R, A, D = cube.shape
        ad = torch.stack([cube.mean(1), cube.amax(1)], 1)                              # (B, 2, A, D)
        E = self.evidence(ad)                                                          # (B, 1, A, D)
        # static Doppler on the curve for every (v_j, a): (J, A), mapped to grid_sample coords in [-1, 1]
        v_s = wrap_t(-self.v_grid[:, None] * self.cos_a[None], self.vmax)               # (J, A)
        gx = v_s / self.vmax                                                            # Doppler -> x in [-1, 1)
        gy = (torch.arange(A, device=cube.device) + 0.5) / A * 2 - 1                    # azimuth -> y
        grid = torch.stack([gx, gy[None].expand_as(gx)], -1)                            # (J, A, 2)
        # NB: grid_sample is not circular; a curve point at the +V/-V seam is sampled with zero padding
        samp = F.grid_sample(E, grid[None].expand(B, -1, -1, -1), align_corners=False,
                             padding_mode="border")                                     # (B, 1, J, A)
        logits = samp[:, 0].mean(-1) * self.temp                                        # (B, J)
        v_e = (logits.softmax(-1) * self.v_grid).sum(-1)                                # (B,)
        return v_e, logits, E[:, 0]


# =========================================================================== [2] static suppression
class StaticDopplerSuppression(nn.Module):
    """Predict where static structure lands in Doppler and attenuate it.

    mask M(a, d) = exp(-0.5 * circ(V_d - wrap(-v_e cos theta_a))^2 / sigma^2)          (B, A, D)
    X'           = X * (1 - gamma * M)          (broadcast over range)                   (B, R, A, D)
    sigma (m/s) and gamma in (0, 1) are learned.  The mask is also returned so the backbone can see it.
    """

    def __init__(self, s_centers: torch.Tensor, v_centers: torch.Tensor, vmax: float):
        super().__init__()
        self.vmax = vmax
        self.register_buffer("cos_a", torch.sqrt(1 - s_centers ** 2))                  # (A,)
        self.register_buffer("v_c", v_centers)                                          # (D,)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(0.35)))
        self.gamma_logit = nn.Parameter(torch.tensor(2.0))

    def forward(self, cube: torch.Tensor, v_e: torch.Tensor):
        v_s = wrap_t(-v_e[:, None] * self.cos_a[None], self.vmax)                       # (B, A)
        dv = wrap_t(self.v_c[None, None] - v_s[..., None], self.vmax)                   # (B, A, D) circular
        M = torch.exp(-0.5 * (dv / self.log_sigma.exp()) ** 2)
        out = cube * (1 - torch.sigmoid(self.gamma_logit) * M[:, None])                 # (B, R, A, D)
        return out, M


# =========================================================================== [3] temporal backbone
class FrameEncoder(nn.Module):
    """One frame (B, Din, R, A) [Doppler bins as channels] -> 3 scales: strides 1, 2, 4."""

    def __init__(self, d_in: int, c: int = 32):
        super().__init__()
        # Doppler encoder: per range-azimuth cell, mix the Doppler spectrum into c channels
        self.dop = nn.Sequential(nn.Conv2d(d_in, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.s1 = conv_bn(c, c)
        self.s2 = nn.Sequential(conv_bn(c, 2 * c, 2), conv_bn(2 * c, 2 * c))
        self.s4 = nn.Sequential(conv_bn(2 * c, 4 * c, 2), conv_bn(4 * c, 4 * c))

    def forward(self, x):
        f1 = self.s1(self.dop(x))
        f2 = self.s2(f1)
        return [f1, f2, self.s4(f2)]


class TemporalMultiScale(nn.Module):
    """Shared frame encoder over T frames, temporal fusion per scale, then a top-down FPN merge to stride 1."""

    def __init__(self, d_in: int, T: int, c: int = 32):
        super().__init__()
        self.T = T
        self.enc = FrameEncoder(d_in, c)
        chans = [c, 2 * c, 4 * c]
        # temporal fusion: concat T frames' features at each scale -> 3x3 conv (learns motion cues)
        self.tfuse = nn.ModuleList([conv_bn(T * ch, ch) for ch in chans])
        self.lat = nn.ModuleList([nn.Conv2d(ch, c, 1) for ch in chans])
        self.out = conv_bn(c, c)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames (B, T, Din, R, A) -> (B, c, R, A)."""
        B, T = frames.shape[:2]
        feats = self.enc(frames.flatten(0, 1))                                          # 3 x (B*T, ch, h, w)
        fused = [f(x.reshape(B, T * x.shape[1], *x.shape[2:])) for f, x in zip(self.tfuse, feats)]
        y = self.lat[2](fused[2])
        for i in (1, 0):                                                                # top-down merge
            y = F.interpolate(y, size=fused[i].shape[-2:], mode="bilinear", align_corners=False) + self.lat[i](fused[i])
        return self.out(y)


# =========================================================================== [4]+[5] heads
class RaMODHead(nn.Module):
    """CenterNet-style dense heads on the range-azimuth grid.

    heat    (K) : class centre heat-maps (sigmoid)
    size    (2) : log box extent (range bins, azimuth bins)
    off     (2) : sub-bin centre offset
    -- velocity under TWO hypotheses (paper: "raw-Doppler" and "unwrapped") --
    v_raw   (1) : radial velocity if the Doppler is NOT wrapped, bounded to [-V, V)   (tanh * V)
    f_raw   (1) : forward (ego-x) ground velocity under the raw hypothesis
    v_unw   (1) : radial velocity under the UNWRAPPED hypothesis (unbounded regression)
    f_unw   (1) : forward velocity under the unwrapped hypothesis
    wrap    (1) : logit of the Doppler-wrapping probability p_wrap
    Inference rule (paper): use (v_unw, f_unw) if p_wrap >= 0.5, else (v_raw, f_raw).
    """

    def __init__(self, c: int, n_cls: int, vmax: float):
        super().__init__()
        self.vmax = vmax

        def head(o, bias=0.0):
            h = nn.Sequential(conv_bn(c, c), nn.Conv2d(c, o, 1))
            nn.init.constant_(h[-1].bias, bias)
            return h

        self.heat = head(n_cls, -2.2)
        self.size = head(2)
        self.off = head(2)
        self.vel = head(4)            # v_raw, f_raw, v_unw, f_unw
        self.wrap = head(1, -1.0)

    def forward(self, f):
        v = self.vel(f)                                                                 # (B, 4, R, A)
        return dict(heat=self.heat(f), size=self.size(f), off=self.off(f),
                    v_raw=torch.tanh(v[:, 0]) * self.vmax, f_raw=v[:, 1] * 5.0,
                    v_unw=v[:, 2] * 8.0, f_unw=v[:, 3] * 5.0, wrap=self.wrap(f)[:, 0])


# =========================================================================== full model
class RaMOD(nn.Module):
    """
    Args:
        geom: dict with s_centers (A,), v_centers (D,), vmax
        T: number of frames (paper default 3: target frame + 2 past)
        suppress: enable [1]+[2] (False -> plain RAD-cube detector)
        wrap_head: False -> only the raw-Doppler hypothesis exists (typical RAD detector)
        frame_diff: feed each past frame's difference from the target frame (paper: "encoded together
                    with their differences from the target frame")
    Input:
        cubes (B, T, R, A, D) normalised log-power, index T-1 is the target (current) frame
    """

    def __init__(self, geom: dict, n_cls: int = 3, T: int = 3, c: int = 32,
                 suppress: bool = True, wrap_head: bool = True, frame_diff: bool = True):
        super().__init__()
        s_c = torch.as_tensor(geom["s_centers"], dtype=torch.float32)
        v_c = torch.as_tensor(geom["v_centers"], dtype=torch.float32)
        self.vmax, self.T = float(geom["vmax"]), T
        self.suppress, self.wrap_head, self.frame_diff = suppress, wrap_head, frame_diff and T > 1
        D = len(v_c)
        if suppress:
            self.ego = EgoMotionHough(s_c, self.vmax)
            self.sup = StaticDopplerSuppression(s_c, v_c, self.vmax)
        d_in = D * (1 + int(suppress) + int(self.frame_diff))
        self.backbone = TemporalMultiScale(d_in, T, c)
        self.head = RaMODHead(c, n_cls, self.vmax)

    def forward(self, cubes: torch.Tensor) -> dict:
        B, T, R, A, D = cubes.shape
        flat = cubes.flatten(0, 1)                                                      # (B*T, R, A, D)
        out = {}
        if self.suppress:
            v_e, logits, E = self.ego(flat)                                             # [1] per frame
            x, M = self.sup(flat, v_e)                                                  # [2]
            out.update(v_e=v_e.view(B, T), ego_logits=logits.view(B, T, -1), mask=M.view(B, T, A, D))
        else:
            x = flat
        x = x.view(B, T, R, A, D)
        parts = [x]
        if self.suppress:                                                               # what was removed
            parts.append(out["mask"][:, :, None].expand(-1, -1, R, -1, -1))
        if self.frame_diff:                                                             # x_t - x_target
            parts.append(x - x[:, -1:])
        x = torch.cat(parts, -1)                                                        # (B, T, R, A, Din)
        x = x.permute(0, 1, 4, 2, 3)                                                    # (B, T, Din, R, A)
        out.update(self.head(self.backbone(x)))                                         # [3] + [4] + [5]
        if not self.wrap_head:
            out["wrap"] = None
        return out


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
