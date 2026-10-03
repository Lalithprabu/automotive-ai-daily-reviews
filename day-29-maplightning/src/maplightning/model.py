"""MapLightning-style 1D-map-token mapper + a dense-BEV (IPM-lifting) baseline sharing the same decoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import X_MIN, X_MAX, Y_HALF, N_PTS, IMG_H, IMG_W, NOM, project, nominal_camera


class Stem(nn.Module):
    """(B,3,48,96) -> (B,C,12,24). Shared by both models so the comparison isolates the view transform."""
    def __init__(self, c=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.GELU(),
                                 nn.Conv2d(32, c, 3, 2, 1), nn.GroupNorm(8, c), nn.GELU())

    def forward(self, x):
        return self.net(x)


class PolylineDecoder(nn.Module):
    """DETR-style set decoder: Q learned queries cross-attend (FULL attention, not deformable) to memory tokens,
    then predict a class (divider / boundary / none) and N_PTS (x,y) points each."""
    def __init__(self, d=64, q=6, layers=2, heads=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(q, d) * 0.1)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, 2 * d, 0.0, batch_first=True, norm_first=True), layers)
        self.cls = nn.Linear(d, 3)
        self.pts = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, N_PTS * 2))
        self.register_buffer("xs", torch.linspace(X_MIN, X_MAX, N_PTS))

    def forward(self, mem):                       # mem (B,T,d)
        q = self.queries.expand(mem.shape[0], -1, -1)
        h = self.dec(q, mem)                      # (B,Q,d)
        raw = self.pts(h).view(*h.shape[:2], N_PTS, 2)
        x = self.xs + 2.0 * torch.tanh(raw[..., 0])          # start from a fixed forward grid
        y = Y_HALF * torch.tanh(raw[..., 1])
        return self.cls(h), torch.stack([x, y], -1)           # (B,Q,3), (B,Q,N,2)


class MapLightning(nn.Module):
    """Image tokens + M learned 1D map tokens -> joint full self-attention -> DISCARD image tokens -> decode.
    No camera intrinsics/extrinsics anywhere in forward()."""
    def __init__(self, d=64, n_map=24, enc_layers=3, heads=4):
        super().__init__()
        self.stem = Stem(d)
        self.patch = nn.Conv2d(d, d, 2, 2)                    # 12x24 -> 6x12 = 72 image tokens
        self.pos = nn.Parameter(torch.randn(1, 72, d) * 0.02)
        self.map_tokens = nn.Parameter(torch.randn(1, n_map, d) * 0.02)
        self.n_map = n_map
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, 0.0, batch_first=True, norm_first=True), enc_layers)
        self.dec = PolylineDecoder(d, heads=heads)

    def map_tokens_out(self, img):
        t = self.patch(self.stem(img)).flatten(2).transpose(1, 2) + self.pos       # (B,72,d)
        z = torch.cat([self.map_tokens.expand(t.shape[0], -1, -1), t], 1)         # (B,M+72,d)
        z = self.enc(z)                                                            # full self-attention
        return z[:, :self.n_map]                                                   # image tokens discarded

    def forward(self, img):
        return self.dec(self.map_tokens_out(img))


class BEVBaseline(nn.Module):
    """'Old way': lift image features onto a dense BEV grid through the NOMINAL camera calibration
    (inverse-perspective grid_sample, LSS/BEVFormer-style spirit), conv, then the same decoder."""
    def __init__(self, d=64, gx=24, gy=16, heads=4):
        super().__init__()
        self.stem = Stem(d)
        self.gx, self.gy = gx, gy
        self.bev = nn.Sequential(nn.Conv2d(d, d, 3, 1, 1), nn.GroupNorm(8, d), nn.GELU(), nn.Conv2d(d, d, 3, 1, 1), nn.GELU())
        self.pos = nn.Parameter(torch.randn(1, gx * gy, d) * 0.02)
        self.dec = PolylineDecoder(d, heads=heads)
        xs = torch.linspace(X_MIN, X_MAX, gx)
        ys = torch.linspace(Y_HALF, -Y_HALF, gy)
        gxx, gyy = torch.meshgrid(xs, ys, indexing="ij")
        self.register_buffer("ground", torch.stack([gxx, gyy, torch.zeros_like(gxx)], -1).view(1, -1, 3))

    def lift(self, feat, cam):
        B = feat.shape[0]
        u, v, zc = project(self.ground.expand(B, -1, -1), cam["h"], cam["pitch"], cam["yaw"])
        grid = torch.stack([u / IMG_W * 2 - 1, v / IMG_H * 2 - 1], -1).view(B, self.gx, self.gy, 2)
        out = F.grid_sample(feat, grid, align_corners=False, padding_mode="zeros")   # (B,C,gx,gy)
        return out

    def forward(self, img, cam=None):
        feat = self.stem(img)
        cam = cam or nominal_camera(img.shape[0])             # deployed with the NOMINAL calibration
        bev = self.bev(self.lift(feat, cam)).flatten(2).transpose(1, 2) + self.pos   # (B,gx*gy,d)
        return self.dec(bev)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
