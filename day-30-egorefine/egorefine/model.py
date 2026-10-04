"""EgoRefine-style asynchronous collaborative-perception fusion (PyTorch reconstruction).

Paper-sourced (abstract level only): (1) an ego-referenced predictive alignment module that uses the
RECEIVING agent's current features to guide trajectory-field prediction and refine sampling offsets along an
ego-referenced direction; (2) a trajectory-conditioned reliability-aware fusion module that uses trajectory
discrepancy and refinement magnitude as alignment-quality indicators to adaptively reweight fusion.
Everything below (dims, K, the exact form of direction/step/gate) is THIS REPO's own reconstruction.

Modes (the ablation ladder, each trained as its own model):
  ego_only   : no collaborator at all
  naive      : delayed collaborator features fused as-is (no alignment)
  traf       : TraF-Align-style  -> trajectory field from collaborator features alone, no ego reference, no gate
  egorefine  : trajectory field + ego-referenced refinement + reliability gate
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("ego_only", "naive", "traf", "egorefine")


def conv_gn(cin, cout, k=3, d=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=d * (k // 2), dilation=d),
                         nn.GroupNorm(4, cout), nn.ReLU(inplace=True))


class AsyncFusionNet(nn.Module):
    def __init__(self, mode="egorefine", C=24, K=3, max_off=8.0, max_step=4.0, ego_ch=5, col_ch=5):
        super().__init__()
        assert mode in MODES
        self.mode, self.K, self.max_off, self.max_step = mode, K, max_off, max_step
        self.ego_enc = nn.Sequential(conv_gn(ego_ch, C), conv_gn(C, C))
        self.col_enc = nn.Sequential(conv_gn(col_ch, C), conv_gn(C, C))
        # (1) base trajectory field from collaborator features (+delay plane): K sampling points, each (dx,dy)+logit
        self.traj = nn.Sequential(conv_gn(C + 1, C), conv_gn(C, C, d=2), conv_gn(C, C, d=4),
                                  nn.Conv2d(C, 3 * K, 1))
        # (2) ego-referenced refinement: sees ego + collab + base offsets; a global-context branch pools cross-view evidence
        self.ref_in = nn.Sequential(conv_gn(2 * C + 1 + 2 * K, C), conv_gn(C, C, d=2), conv_gn(C, C, d=4))
        self.ref_glob = nn.Sequential(nn.Linear(C, C), nn.ReLU(inplace=True))
        self.ref_out = nn.Conv2d(2 * C, 2 + K + K, 1)         # direction(2) + signed step per point (K) + attn delta (K)
        # (3) reliability gate from [trajectory discrepancy, refinement magnitude]
        self.gate = nn.Sequential(nn.Conv2d(2, 8, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(8, 1, 3, padding=1))
        # fusion + detection head
        self.fuse = nn.Sequential(conv_gn(2 * C, C), conv_gn(C, C, d=2))
        self.head = nn.Conv2d(C, 1, 1)
        nn.init.constant_(self.head.bias, -4.0)
        nn.init.zeros_(self.traj[-1].weight); nn.init.zeros_(self.traj[-1].bias)
        nn.init.zeros_(self.ref_out.weight); nn.init.zeros_(self.ref_out.bias)

    @staticmethod
    def _base_grid(B, H, W, dev):
        ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                                torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
        return torch.stack([xs, ys], 0)[None].expand(B, -1, -1, -1)       # (B,2,H,W) pixel coords

    def _sample(self, feat, off):
        """feat (B,C,H,W); off (B,K,2,H,W) in cells -> (B,K,C,H,W) bilinear samples at p+off_k."""
        B, C, H, W = feat.shape
        base = self._base_grid(B, H, W, feat.device)
        outs = []
        for k in range(off.shape[1]):
            p = base + off[:, k]
            gx = 2 * p[:, 0] / (W - 1) - 1
            gy = 2 * p[:, 1] / (H - 1) - 1
            outs.append(F.grid_sample(feat, torch.stack([gx, gy], -1), align_corners=True, padding_mode="zeros"))
        return torch.stack(outs, 1)

    def align(self, ego_f, col_f, delay_plane):
        """Returns aligned collaborator feature (B,C,H,W) and a dict of diagnostics."""
        B, C, H, W = col_f.shape
        K = self.K
        if self.mode == "naive":
            return col_f, {}
        raw = self.traj(torch.cat([col_f, delay_plane], 1)).view(B, K, 3, H, W)
        o_base = torch.tanh(raw[:, :, :2]) * self.max_off                # (B,K,2,H,W)
        a_base = raw[:, :, 2]                                            # (B,K,H,W) attention logits
        s_base = self._sample(col_f, o_base)                             # (B,K,C,H,W)
        if self.mode == "traf":
            w = torch.softmax(a_base, 1)[:, :, None]
            return (w * s_base).sum(1), dict(o_base=o_base, o_ref=o_base, w=w[:, :, 0])
        # ---- ego-referenced refinement ----
        h = self.ref_in(torch.cat([col_f, ego_f, delay_plane, o_base.flatten(1, 2) / self.max_off], 1))
        g = self.ref_glob(h.mean((2, 3)))[:, :, None, None].expand(-1, -1, H, W)
        r = self.ref_out(torch.cat([h, g], 1))
        d = r[:, :2] + 1e-3 * torch.tensor([1.0, 0.0], device=r.device)[None, :, None, None]
        d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-6)                # unit ego-referenced direction (B,2,H,W)
        step = torch.tanh(r[:, 2:2 + K]) * self.max_step                   # signed magnitude per sampling point (B,K,H,W)
        o_ref = o_base + step[:, :, None] * d[:, None]                      # refine ALONG the direction
        a_ref = a_base + r[:, 2 + K:]
        s_ref = self._sample(col_f, o_ref)
        w = torch.softmax(a_ref, 1)[:, :, None]
        aligned = (w * s_ref).sum(1)
        # ---- reliability indicators ----
        disc = (s_ref - s_base).abs().mean((1, 2))[:, None]                # feature-space trajectory discrepancy (B,1,H,W)
        mag = step.abs().mean(1, keepdim=True) / self.max_step              # refinement magnitude (B,1,H,W)
        gate = torch.sigmoid(self.gate(torch.cat([disc, mag], 1)))
        return aligned * gate, dict(o_base=o_base, o_ref=o_ref, w=w[:, :, 0], gate=gate, disc=disc, mag=mag, dir=d)

    def forward(self, ego, col, return_aux=False):
        ego_f = self.ego_enc(ego)
        if self.mode == "ego_only":
            fused_in = torch.cat([ego_f, torch.zeros_like(ego_f)], 1); aux = {}
        else:
            col_f = self.col_enc(col)
            aligned, aux = self.align(ego_f, col_f, col[:, 4:5])
            fused_in = torch.cat([ego_f, aligned], 1)
        logit = self.head(self.fuse(fused_in))
        return (logit, aux) if return_aux else logit
