"""PyTorch core: a sensor-agnostic video world model that forecasts RGB, thermal and LiDAR-depth video, alone or as
ONE tri-sensor video.

Re-implementation of the idea in arXiv 2611.22669 (Chowdhury, Rahman, Hasan, Gupta, Haque, Shannon, "Beyond the
Camera: A Tri-Modal Driving World Model for Future RGB, Thermal, and LiDAR Forecasting in Adverse Weather"), built from
the abstract only.  What the abstract states:

    (1) a PRETRAINED video world model is POST-TRAINED for each sensor
    (2) LiDAR is represented as a DEPTH VIDEO ALIGNED WITH THE RGB CAMERA
    (3) a TRI-SENSOR model sees all three sensors AS ONE VIDEO
    (4) LiDAR is scored in metres against measured returns; forecasts are compared with persistence

Everything else (the tokenizer, ConvGRU + global attention dynamics, sizes, losses, tiling layout) is my construction;
the authors post-train a large pretrained video model, this is a ~0.6M-parameter stand-in.  See SOURCING.md.

Shapes (B clips, T = 14 frames, n tiles side by side, each tile 3 x 48 x 64, latent D = 96 at 1/4 resolution):

    canvas   x (B, T, 3, 48, 64 n)       tiles = [RGB | thermal x3 | LiDAR (d, d, valid)] concatenated along width
    tokenize z_t = Enc(x_t)  (B, D, 12, 16 n)      3 convs, stride 4, GroupNorm
             + P[:, :, :16 n]                     learned 2-D position table over the whole canvas (no sensor ids:
                                                  the model must learn which columns are which sensor, as in a video)
    dynamics h_t = ConvGRU(z_t, h_{t-1})          (B, D, 12, 16 n)   local motion memory, 3 x 3 gates
             h_t = h_t + Attn(LN(h_t))           global self-attention over all 12 x 16 n tokens of the canvas:
                                                  the main path by which the RGB tile informs the LiDAR tile (the 3x3 ConvGRU
                                                  only mixes neighbouring tiles at their seams)
             h_t = h_t + MLP(LN(h_t))
    decode   x_hat_{t+1} = x_t + Dec(h_t)        (B, 3, 48, 64 n)   residual on the current frame
    rollout  t < T_ctx: teacher forcing (feed the observed frame); t >= T_ctx: feed the model's own forecast back,
             so the 8 future frames (2 s) are generated autoregressively from 6 observed frames (1.5 s)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TILE_H, TILE_W = 48, 64
LAT_H, LAT_W = 12, 16                       # per tile, after stride 4
MAX_TILES = 3


class ConvGRUCell(nn.Module):
    def __init__(self, c_in: int, c_h: int, k: int = 3):
        super().__init__()
        self.c_h = c_h
        self.gates = nn.Conv2d(c_in + c_h, 2 * c_h, k, padding=k // 2)
        self.cand = nn.Conv2d(c_in + c_h, c_h, k, padding=k // 2)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """x (B, C_in, h, w), h (B, C_h, h, w) -> (B, C_h, h, w)."""
        zr = torch.sigmoid(self.gates(torch.cat([x, h], 1)))
        z, r = zr.chunk(2, 1)                                             # update gate, reset gate
        n = torch.tanh(self.cand(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


class CanvasAttention(nn.Module):
    """Pre-norm transformer block over every latent token of the canvas (all tiles at once)."""

    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.last_attn: torch.Tensor | None = None                       # kept for visualisation when requested
        self.keep = False

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, D, hh, ww = h.shape
        x = h.flatten(2).transpose(1, 2)                                  # (B, N = hh * ww, D)
        y = self.n1(x)
        y, w = self.att(y, y, y, need_weights=self.keep, average_attn_weights=True)
        if self.keep:
            self.last_attn = w.detach()                                   # (B, N, N)
        x = x + y
        x = x + self.mlp(self.n2(x))
        return x.transpose(1, 2).reshape(B, D, hh, ww)


class Tokenizer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 24, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(24, 48, 4, stride=2, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
                                 nn.Conv2d(48, d, 4, stride=2, padding=1), nn.GroupNorm(8, d))

    def forward(self, x):                                                 # (B, 3, 48, 64 n) -> (B, D, 12, 16 n)
        return self.net(x)


class Detokenizer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.ConvTranspose2d(d, 48, 4, stride=2, padding=1), nn.GELU(),
                                 nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1), nn.GELU(),
                                 nn.Conv2d(24, 3, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)                               # start as exact persistence
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):                                                 # (B, D, 12, 16 n) -> (B, 3, 48, 64 n)
        return self.net(h)


class VideoWorldModel(nn.Module):
    """Same weights for 1 tile (single-sensor) or 3 tiles (tri-sensor "one video"); n is read from the input width."""

    def __init__(self, d: int = 64, layers: int = 1):
        super().__init__()
        self.d = d
        self.enc = Tokenizer(d)
        self.pos = nn.Parameter(torch.randn(1, d, LAT_H, LAT_W * MAX_TILES) * 0.02)
        self.gru = ConvGRUCell(d, d)
        self.attn = nn.ModuleList([CanvasAttention(d) for _ in range(layers)])
        self.dec = Detokenizer(d)

    def step(self, x: torch.Tensor, h: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """One frame in -> (forecast of the next frame, new state)."""
        z = self.enc(x)
        z = z + self.pos[..., :z.shape[-1]]
        if h is None:
            h = torch.zeros_like(z)
        h = self.gru(z, h)
        for a in self.attn:
            h = a(h)
        return x + self.dec(h), h

    def forward(self, clip: torch.Tensor, t_ctx: int, blank: torch.Tensor | None = None) -> torch.Tensor:
        """clip (B, T, 3, 48, 64 n) -> forecasts (B, T - 1, 3, 48, 64 n) of frames 1 .. T-1.

        Frames 1 .. t_ctx-1 are one-step forecasts under teacher forcing; frames t_ctx .. T-1 are a free autoregressive
        rollout.  `blank` (B, 1, 1, 64 n) or None: a 0/1 multiplier applied to every frame the model SEES (observed or
        fed back), used to knock out one sensor at test time.
        """
        B, T = clip.shape[:2]
        h, out = None, []
        x = clip[:, 0]
        for t in range(T - 1):
            if blank is not None:
                x = x * blank
            y, h = self.step(x, h)
            out.append(y)
            x = clip[:, t + 1] if t + 1 < t_ctx else y
        return torch.stack(out, 1)


# ------------------------------------------------------------------------------------------------ losses / metrics
def tile_loss(pred: torch.Tensor, tgt: torch.Tensor, sensors: list[int]) -> tuple[torch.Tensor, dict]:
    """pred, tgt (B, T', 3, 48, 64 n); sensors[i] in {0 rgb, 1 thermal, 2 lidar} is the content of tile i.

    RGB / thermal: L1.  LiDAR: L1 on the depth channels only where the target has a measured return (sparse, x4 weight)
    plus a light L1 on the validity channel (expected return probability).
    """
    parts, total = {}, 0.0
    for i, s in enumerate(sensors):
        p = pred[..., i * TILE_W:(i + 1) * TILE_W]
        g = tgt[..., i * TILE_W:(i + 1) * TILE_W]
        if s in (0, 1):
            l = (p - g).abs().mean()
        else:
            v = g[:, :, 2:3]
            ld = ((p[:, :, :2] - g[:, :, :2]).abs() * v).sum() / (2 * v.sum()).clamp(min=1)
            l = 4.0 * ld + 0.2 * (p[:, :, 2:3] - v).abs().mean()
        parts[["rgb", "thermal", "lidar"][s]] = float(l.detach())
        total = total + l
    return total, parts


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = VideoWorldModel()
    print(f"params {n_params(m):,}")
    for n in (1, 3):
        x = torch.rand(2, 14, 3, TILE_H, TILE_W * n)
        y = m(x, 6)
        loss, parts = tile_loss(y, x[:, 1:], [0, 1, 2][:n] if n == 3 else [2])
        loss.backward()
        print(n, tuple(y.shape), float(loss), parts)
