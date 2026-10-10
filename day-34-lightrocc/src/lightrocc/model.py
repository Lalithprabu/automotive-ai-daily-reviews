"""LighTROcc-style instance-centric Gaussian occupancy forecaster (this repo's reconstruction).

Paper-level idea (abstract only): a compact set of learned queries, one per movable object, localized by
image cross-attention + query-specific depth + camera geometry ("forward lifting"); each instance is a mixture
of anisotropic 3D Gaussians; instances are propagated to future steps via predicted displacements, giving
continuous, identity-consistent occupancy forecasts in one forward pass.
Every dimension / loss / head below is OUR choice (see SOURCING.md).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import unproject, rot2, voxel_centers


class ImageEncoder(nn.Module):
    """12-ch (2 past frames x 6 feature channels) camera features -> [B, N=H/2*W/2, d] tokens."""

    def __init__(self, cfg, c_in=12):
        super().__init__()
        d = cfg["model"]["d_model"]
        h, w = cfg["camera"]["h"] // 2, cfg["camera"]["w"] // 2
        self.net = nn.Sequential(
            nn.Conv2d(c_in, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, d, 3, stride=2, padding=1), nn.GroupNorm(4, d), nn.GELU(),
            nn.Conv2d(d, d, 3, padding=1), nn.GroupNorm(4, d), nn.GELU())
        self.pos = nn.Parameter(torch.randn(1, h * w, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg["model"]["n_heads"], cfg["model"]["ff"], 0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 1)

    def forward(self, img):                       # img [B,12,H,W]
        f = self.net(img)                         # [B,d,H/2,W/2]
        tok = f.flatten(2).transpose(1, 2)        # [B,N,d]
        return self.enc(tok + self.pos)


def gaussians_to_occupancy(centers, yaws, offsets, sigmas, amps, exist, vox):
    """Instance Gaussian mixtures -> voxel occupancy probabilities.

    centers [B,T1,Q,3]  instance centers per time step      yaws [B,T1,Q]
    offsets [B,Q,K,3]   per-Gaussian offsets in object frame sigmas [B,Q,K,3] per-axis std (object frame)
    amps    [B,Q,K] in (0,1)   exist [B,Q] in (0,1)         vox [V,3]
    Returns probs [B,T1,V] = noisy-OR over instances of noisy-OR over Gaussians.
    """
    R = rot2(yaws)                                                   # [B,T1,Q,2,2]
    off_xy = torch.einsum("btqij,bqkj->btqki", R, offsets[..., :2])  # rotate offsets with the instance
    mu = centers[:, :, :, None, :].clone()                           # [B,T1,Q,1,3]
    mu = torch.cat([mu[..., :2] + off_xy, mu[..., 2:] + offsets[:, None, :, :, 2:]], -1)  # [B,T1,Q,K,3]
    d = vox[None, None, None, None] - mu[:, :, :, :, None, :]        # [B,T1,Q,K,V,3]
    c, s = torch.cos(yaws)[..., None, None], torch.sin(yaws)[..., None, None]
    lx = c * d[..., 0] + s * d[..., 1]                               # world -> object frame
    ly = -s * d[..., 0] + c * d[..., 1]
    sg = sigmas[:, None, :, :, None, :]                              # [B,1,Q,K,1,3]
    m = (lx / sg[..., 0]) ** 2 + (ly / sg[..., 1]) ** 2 + (d[..., 2] / sg[..., 2]) ** 2
    g = amps[:, None, :, :, None] * torch.exp(-0.5 * m)              # [B,T1,Q,K,V]
    log_not_k = torch.log1p(-g.clamp(max=1 - 1e-6)).sum(3)           # sum over Gaussians
    o_q = 1 - torch.exp(log_not_k)                                   # [B,T1,Q,V]
    log_not_q = torch.log1p(-(exist[:, None, :, None] * o_q).clamp(max=1 - 1e-6)).sum(2)
    return 1 - torch.exp(log_not_q)                                  # [B,T1,V]


class LighTROcc(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        m, w = cfg["model"], cfg["world"]
        d, self.Q, self.K, self.T = m["d_model"], m["n_queries"], m["n_gauss"], w["t_future"]
        self.encoder = ImageEncoder(cfg)
        self.query = nn.Parameter(torch.randn(1, self.Q, d) * 0.5)
        layer = nn.TransformerDecoderLayer(d, m["n_heads"], m["ff"], 0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, m["dec_layers"])
        self.norm = nn.LayerNorm(d)
        # per-query head: exist(1) uv(2) depth(1) yaw(2) size(3) gauss K*(off3+logsig3+amp1) disp T*(dx,dy,dyaw)
        self.n_out = 1 + 2 + 1 + 2 + 3 + self.K * 7 + self.T * 3
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.n_out))
        # learned reference points spread over the image (logit space) so queries start diverse
        ref = torch.stack([torch.linspace(-1.2, 1.2, self.Q), torch.zeros(self.Q) + 0.2], -1)
        self.ref_uv = nn.Parameter(ref)
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            b = self.head[-1].bias
            b[3] = math.log(16.0)                                   # depth init ~16 m
            b[6:9] = torch.tensor([math.log(4.5), math.log(1.9), math.log(1.6)])
            for k in range(self.K):                                  # Gaussian log-sigma init
                b[9 + k * 7 + 3: 9 + k * 7 + 6] = math.log(1.0)
        self.register_buffer("vox", voxel_centers(cfg), persistent=False)
        self.register_buffer("gauss_off_scale", torch.tensor([2.0, 0.8, 0.5]), persistent=False)

    def forward(self, img, return_instances=False, zero_disp=False):
        B = img.shape[0]
        mem = self.encoder(img)                                       # [B,N,d]
        q = self.decoder(self.query.expand(B, -1, -1), mem)           # [B,Q,d]
        o = self.head(self.norm(q))                                   # [B,Q,n_out]
        cam = self.cfg["camera"]
        exist = torch.sigmoid(o[..., 0])
        u = torch.sigmoid(o[..., 1] + self.ref_uv[:, 0]) * cam["w"]
        v = torch.sigmoid(o[..., 2] + self.ref_uv[:, 1]) * cam["h"]
        depth = torch.exp(o[..., 3].clamp(-1, 4.5))                   # query-specific depth along optical axis
        c0 = unproject(u, v, depth, self.cfg)                         # forward lifting -> [B,Q,3]
        yaw0 = torch.atan2(o[..., 4] + 1e-6, o[..., 5] + 1e-6)         # heading from (sin, cos) logits
        size = torch.exp(o[..., 6:9].clamp(-1, 2.5))
        g = o[..., 9:9 + self.K * 7].reshape(B, self.Q, self.K, 7)
        offsets = torch.tanh(g[..., 0:3]) * self.gauss_off_scale
        sigmas = torch.exp(g[..., 3:6].clamp(-2.0, 1.5))
        amps = torch.sigmoid(g[..., 6] + 2.0)
        disp = o[..., 9 + self.K * 7:].reshape(B, self.Q, self.T, 3)
        if zero_disp:
            disp = disp * 0
        steps = torch.cumsum(disp * torch.tensor([3.0, 3.0, 0.3], device=disp.device), 2)   # [B,Q,T,3]
        zero = torch.zeros(B, self.Q, 1, 3, device=img.device)
        steps = torch.cat([zero, steps], 2).permute(0, 2, 1, 3)       # [B,T1,Q,3]
        centers = torch.stack([c0[..., 0], c0[..., 1], c0[..., 2]], -1)[:, None] + torch.cat(
            [steps[..., :2], torch.zeros_like(steps[..., :1])], -1)   # rigid translation of the whole instance
        yaws = yaw0[:, None] + steps[..., 2]
        probs = gaussians_to_occupancy(centers, yaws, offsets, sigmas, amps, exist, self.vox)
        out = dict(probs=probs)
        if return_instances:
            out.update(exist=exist, centers=centers, yaws=yaws, size=size, depth=depth, uv=torch.stack([u, v], -1))
        return out
