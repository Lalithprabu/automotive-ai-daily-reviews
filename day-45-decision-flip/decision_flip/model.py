"""Persistent Gaussian bank for semantic occupancy (GaussianFormer-style, heavily simplified).

Each of N persistent rows = one 3D Gaussian: mean (3), log-scale (3), opacity logit, class logits (C).
Per scene, every row (a) samples the BEV feature map at its mean, (b) refines its mean/opacity/class
via a shared MLP, (c) is splatted into voxels. Voxel class mass = sum_i alpha_i * N(v; mu_i, S_i) * p_i(c),
plus a learned constant for the 'empty' class. Because the readout is an *additive* sum over rows,
the exact contribution of any row to any voxel decision is known and removable (see flip.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianOccBank(nn.Module):
    def __init__(self, n, num_classes, feat_ch, hidden, grid, max_shift=1.5):
        super().__init__()
        self.C, self.grid, self.max_shift = num_classes, grid, max_shift
        X, Y, Z = grid
        g = torch.Generator().manual_seed(1234)
        self.mean = nn.Parameter(torch.rand(n, 3, generator=g) * torch.tensor([X, Y, Z]).float())  # (N,3) voxel coords
        self.log_scale = nn.Parameter(torch.full((n, 3), 0.4))                                       # (N,3)
        self.opa = nn.Parameter(torch.zeros(n))                                                      # (N,)
        self.cls = nn.Parameter(torch.randn(n, num_classes, generator=g) * 0.1)                      # (N,C)
        # shared refinement MLP: sampled feature + row state -> (d_mean 3, d_opa 1, d_cls C)
        self.refine = nn.Sequential(nn.Linear(feat_ch + 3, hidden), nn.ReLU(), nn.Linear(hidden, 3 + 1 + num_classes))
        self.empty_bias = nn.Parameter(torch.tensor(0.0))
        vox = torch.stack(torch.meshgrid(*[torch.arange(s).float() + 0.5 for s in grid], indexing="ij"), -1)
        self.register_buffer("vox", vox.reshape(-1, 3))                                              # (V,3), V = X*Y*Z

    @property
    def n(self):
        return self.mean.shape[0]

    def row_contribs(self, feat):
        """feat (B,F,X,Y) -> per-row voxel class mass contrib (B,N,V,C)."""
        B = feat.shape[0]; X, Y, Z = self.grid
        xy = self.mean[:, :2] / torch.tensor([X, Y], device=feat.device) * 2 - 1           # (N,2) in [-1,1]
        # grid_sample uses (x=W, y=H) order; feat dims are (X,Y) -> H=X, W=Y, so swap
        samp = F.grid_sample(feat, xy[:, [1, 0]].unsqueeze(0).expand(B, -1, -1).unsqueeze(2),
                             align_corners=False).squeeze(-1).transpose(1, 2)               # (B,N,F)
        d = self.refine(torch.cat([samp, self.mean.unsqueeze(0).expand(B, -1, -1) / 8 - 1], -1))   # (B,N,4+C)
        mu = self.mean.unsqueeze(0) + self.max_shift * torch.tanh(d[..., :3])                # (B,N,3)
        alpha = torch.sigmoid(self.opa.unsqueeze(0) + d[..., 3])                              # (B,N)
        p = torch.softmax(self.cls.unsqueeze(0) + d[..., 4:], -1)                             # (B,N,C)
        inv = torch.exp(-2 * self.log_scale).unsqueeze(0)                                     # (1,N,3)
        diff = self.vox[None, None] - mu.unsqueeze(2)                                         # (B,N,V,3)
        w = torch.exp(-0.5 * (diff.pow(2) * inv.unsqueeze(2)).sum(-1))                        # (B,N,V)
        return (alpha.unsqueeze(-1) * w).unsqueeze(-1) * p.unsqueeze(2)                       # (B,N,V,C)

    def mass(self, feat, gate=None):
        c = self.row_contribs(feat)                                                           # (B,N,V,C)
        if gate is not None:
            c = c * gate.view(1, -1, 1, 1)
        m = c.sum(1)                                                                          # (B,V,C)
        return m, c

    def forward(self, feat, gate=None):
        m, _ = self.mass(feat, gate)
        m = m.clone()
        m[..., 0] = m[..., 0] + F.softplus(self.empty_bias)
        return torch.log(m + 1e-3)                                                            # (B,V,C) logits


class GaussianOccModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        g = cfg["grid"]; self.grid = (g["x"], g["y"], g["z"])
        m = cfg["model"]
        self.bank = GaussianOccBank(m["num_gaussians"], cfg["num_classes"], cfg["feat_channels"], m["hidden"],
                                    self.grid, m["max_shift"])
        self.cfg = cfg

    def forward(self, feat, gate=None):
        return self.bank(feat, gate)
