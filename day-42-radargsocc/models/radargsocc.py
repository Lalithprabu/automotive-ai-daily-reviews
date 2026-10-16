"""RadarGSOcc-style model (this repo's own small reconstruction from the paper's abstract-level description).

Paper-described mechanism: 4D-radar tensor -> separate GEOMETRIC and SEMANTIC(+Doppler) feature extraction,
each with deformable self-attention; semantic features refine geometric ones via cross-attention; a second
deformable-attention step updates occupancy features using the refined geometry; voxel classification head.
All dimensions, dense (not sparse) 3D convs, and the K-offset deformable attention below are OUR defaults.
Shapes: B batch, C feature dim, grid (X,Y,Z) = (48,32,8).
"""
import torch, torch.nn as nn, torch.nn.functional as F


def conv_block(i, o):
    return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.GELU())


class DeformAttn3D(nn.Module):
    """Deformable attention on a 3D voxel grid.
    Each query voxel predicts K sampling offsets (in voxels) + K softmax weights, bilinearly samples the
    VALUE map at ref+offset, and returns the weighted sum:  out(q) = sum_k a_qk * V(p_q + d_qk).
    query: (B,C,X,Y,Z), value: (B,C,X,Y,Z)  ->  (B,C,X,Y,Z). value==query gives self-attention."""
    def __init__(self, c, k=4):
        super().__init__(); self.k = k
        self.off = nn.Conv3d(c, 3 * k, 1); self.w = nn.Conv3d(c, k, 1)
        self.v = nn.Conv3d(c, c, 1); self.o = nn.Conv3d(c, c, 1)
        nn.init.zeros_(self.off.weight); nn.init.uniform_(self.off.bias, -1.5, 1.5)   # spread initial taps
    def forward(self, query, value):
        B, C, X, Y, Z = query.shape
        off = self.off(query).view(B, self.k, 3, X, Y, Z)             # (B,K,3,X,Y,Z) offsets in voxels
        a = F.softmax(self.w(query), 1)                                # (B,K,X,Y,Z)
        v = self.v(value)
        ax = [torch.arange(n, device=query.device, dtype=query.dtype) for n in (X, Y, Z)]
        gx, gy, gz = torch.meshgrid(*ax, indexing="ij")
        ref = torch.stack([gx, gy, gz], 0)[None, None]                 # (1,1,3,X,Y,Z)
        pos = ref + off                                                # absolute voxel coords
        sz = torch.tensor([X, Y, Z], device=query.device, dtype=query.dtype).view(1, 1, 3, 1, 1, 1)
        norm = 2 * pos / (sz - 1) - 1                                  # -> [-1,1]
        # grid_sample 5-D wants (B,Xo,Yo,Zo,3) with last dim ordered (z,y,x) for tensor dims (X,Y,Z)->(D,H,W)=(X,Y,Z)
        # so last-dim order is (W=Z, H=Y, D=X)
        out = 0
        for i in range(self.k):
            g = norm[:, i].permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]      # (B,X,Y,Z,3)
            s = F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=True)
            out = out + a[:, i:i + 1] * s
        return self.o(out)


class AttnBlock(nn.Module):
    """Residual deformable attention + FFN. cross=True takes values from a second feature map."""
    def __init__(self, c, k=4):
        super().__init__(); self.n1 = nn.GroupNorm(4, c); self.n2 = nn.GroupNorm(4, c)
        self.attn = DeformAttn3D(c, k)
        self.ffn = nn.Sequential(nn.Conv3d(c, 2 * c, 1), nn.GELU(), nn.Conv3d(2 * c, c, 1))
    def forward(self, x, ctx=None):
        x = x + self.attn(self.n1(x), self.n1(x) if ctx is None else ctx)
        return x + self.ffn(self.n2(x))


class RadarGSOcc(nn.Module):
    """Dual-branch geometric/semantic radar occupancy network.
    in: radar (B,8,X,Y,Z) = [p1,p2,p3,d1,d2,d3,mean,var]; out: logits (B,NC,X,Y,Z)."""
    def __init__(self, c=24, n_classes=5, k=4):
        super().__init__()
        # geometric branch: power-structure channels  p1,p2,p3,mean,var  (5)
        self.geo_in = nn.Sequential(conv_block(5, c), conv_block(c, c)); self.geo_attn = AttnBlock(c, k)
        # semantic/motion branch: Doppler indices d1..d3 + p1 (power gate)  (4)
        self.sem_in = nn.Sequential(conv_block(4, c), conv_block(c, c)); self.sem_attn = AttnBlock(c, k)
        self.cross = AttnBlock(c, k)                    # geometry queries, semantic values (cross-attention)
        self.fuse = conv_block(2 * c, c)                # concat geo-refined + sem
        self.occ_refine = AttnBlock(c, k)               # 2nd deformable step: refined geometry updates occupancy features
        self.head = nn.Conv3d(c, n_classes, 1)
    def forward(self, r):
        geo = r[:, [0, 1, 2, 6, 7]]; sem = r[:, [3, 4, 5, 0]]
        g = self.geo_attn(self.geo_in(geo)); s = self.sem_attn(self.sem_in(sem))
        g = self.cross(g, s)                                            # semantic refines geometric
        f = self.fuse(torch.cat([g, s], 1))                             # (B,C,X,Y,Z)
        f = self.occ_refine(f, g)                                       # occupancy feats attend refined geometry
        return self.head(f)


class RadarOccBaseline(nn.Module):
    """'Old way' (RadarOcc-style stand-in, NOT the real RadarOcc): single early-fused stream, plain 3D convs,
    no branch split, no deformable attention. Width chosen so parameter count is in the same range."""
    def __init__(self, c=40, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(conv_block(8, c), conv_block(c, c), conv_block(c, c), conv_block(c, c),
                                 nn.Conv3d(c, n_classes, 1))
    def forward(self, r): return self.net(r)


def n_params(m): return sum(p.numel() for p in m.parameters())
