"""Sparse2comm-style cooperative detector (this repo's own reconstruction from an abstract-level description).

Pipeline restores, in order:  semantic completeness -> temporal consistency -> spatial alignment, then fuses.
  1. SparseFeatureDecoder   : dense BEV features reconstructed from randomly mask-sampled foreground cells.
  2. LatencyAwareAlignment  : predicts a motion-flow field and warps the (stale) message forward to time t.
  3. SelfCalibratingFusion  : estimates the residual rigid offset *without labels* (soft-argmax over a shift
                              correlation volume vs. the ego's own features), shifts, then gated-fuses.
Tensor shapes use B=batch, C=raw feature channels, H=hidden channels, (Hh,Ww)=BEV grid.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelCfg:
    C: int = 8
    hidden: int = 24
    max_flow: float = 4.0     # cells
    shift_range: float = 3.0  # calibration search range (cells)
    shift_step: float = 0.5
    prior_sigma: float = 2.0  # Gaussian prior on the offset (keeps estimate ~0 when ego has no evidence)
    cooperate: bool = True
    use_recon: bool = True
    use_flow: bool = True
    use_calib: bool = True


def warp(feat, flow):
    """Backward warp: out(x) = feat(x - flow(x)).  feat [B,C,Hh,Ww], flow [B,2,Hh,Ww] in cells (dx,dy)."""
    B, _, Hh, Ww = feat.shape
    ys, xs = torch.meshgrid(torch.arange(Hh, device=feat.device, dtype=feat.dtype),
                            torch.arange(Ww, device=feat.device, dtype=feat.dtype), indexing="ij")
    sx = xs[None] - flow[:, 0]
    sy = ys[None] - flow[:, 1]
    grid = torch.stack([2 * sx / (Ww - 1) - 1, 2 * sy / (Hh - 1) - 1], -1)  # [B,Hh,Ww,2]
    return F.grid_sample(feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


class SparseFeatureDecoder(nn.Module):
    """Completes a sparse message. Received cells are kept verbatim; only missing cells are filled in."""
    def __init__(self, C, hid):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C + 1, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=2, dilation=2), nn.ReLU(),     # dilations grow the receptive field so
            nn.Conv2d(hid, hid, 3, padding=3, dilation=3), nn.ReLU(),     # a blob can be inferred from a few cells
            nn.Conv2d(hid, C, 1))

    def forward(self, sparse, mask):            # [B,C,Hh,Ww], [B,1,Hh,Ww]
        rec = self.net(torch.cat([sparse, mask], 1))
        return mask * sparse + (1 - mask) * rec, rec


class LatencyAwareAlignment(nn.Module):
    """Motion-flow predictor. Input = two consecutive reconstructed messages (t-d-1, t-d) + normalised delay.
    Output flow = displacement over the delay d; the newer message is warped forward by it (forward in time)."""
    def __init__(self, C, hid, max_flow, max_delay=3):
        super().__init__()
        self.max_flow, self.max_delay = max_flow, max_delay
        self.net = nn.Sequential(
            nn.Conv2d(2 * C + 1, hid, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(hid, hid, 3, padding=3, dilation=3), nn.ReLU(),
            nn.Conv2d(hid, 2, 1))

    def forward(self, f_old, f_new, delay):     # delay [B] (frames)
        dmap = (delay / self.max_delay)[:, None, None, None].expand(-1, 1, *f_new.shape[2:])
        flow = torch.tanh(self.net(torch.cat([f_old, f_new, dmap], 1))) * self.max_flow   # [B,2,Hh,Ww]
        return warp(f_new, flow), flow


class SelfCalibratingFusion(nn.Module):
    """Label-free residual-offset estimation + adaptive gated fusion."""
    def __init__(self, C, hid, cfg: ModelCfg):
        super().__init__()
        r, st = cfg.shift_range, cfg.shift_step
        n = int(round(2 * r / st)) + 1
        ax = torch.linspace(-r, r, n)
        sy, sx = torch.meshgrid(ax, ax, indexing="ij")
        self.register_buffer("shifts", torch.stack([sx.flatten(), sy.flatten()], 1))   # [S,2] candidate offsets
        self.register_buffer("prior", -(self.shifts ** 2).sum(1) / (2 * cfg.prior_sigma ** 2))
        self.log_tau = nn.Parameter(torch.tensor(1.0))      # learnable softmax temperature
        self.emb_e = nn.Conv2d(C, hid, 1)
        self.emb_c = nn.Conv2d(C, hid, 1)
        self.gate = nn.Sequential(nn.Conv2d(2 * hid, hid, 3, padding=1), nn.ReLU(), nn.Conv2d(hid, hid, 1), nn.Sigmoid())

    def estimate_offset(self, ego_raw, collab_raw):
        """collab(x + o) ~ ego(x)  =>  o = offset of collab features vs ego frame. Soft-argmax over candidates."""
        B, C, Hh, Ww = ego_raw.shape
        m = (ego_raw.norm(dim=1, keepdim=True) > 0.5).float()                # cells where ego sees a vehicle
        S = self.shifts.shape[0]
        with torch.no_grad():   # the S-way correlation volume is a search; only the temperature needs gradients
            cand = collab_raw[:, None].expand(B, S, C, Hh, Ww).reshape(B * S, C, Hh, Ww)
            fl = (-self.shifts)[None].expand(B, S, 2).reshape(B * S, 2)[:, :, None, None].expand(-1, -1, Hh, Ww)
            sh = warp(cand, fl).reshape(B, S, C, Hh, Ww)                     # sh_s(x) = collab(x + s)
            err = (m[:, None] * (sh - ego_raw[:, None]) ** 2).sum((2, 3, 4))  # [B,S]
        logits = -err / torch.exp(self.log_tau) + self.prior[None]
        w = torch.softmax(logits, 1)
        return w @ self.shifts                                               # [B,2]

    def forward(self, ego_raw, collab_raw, ego_emb, use_calib=True):
        B, _, Hh, Ww = ego_raw.shape
        off = self.estimate_offset(ego_raw, collab_raw) if use_calib else torch.zeros(B, 2, device=ego_raw.device)
        shifted = warp(collab_raw, (-off)[:, :, None, None].expand(-1, -1, Hh, Ww))
        ce = self.emb_c(shifted)
        g = self.gate(torch.cat([ego_emb, ce], 1))
        return ego_emb + g * ce, off, shifted


class Sparse2comm(nn.Module):
    def __init__(self, cfg: ModelCfg = ModelCfg()):
        super().__init__()
        self.cfg = cfg
        C, h = cfg.C, cfg.hidden
        self.ego_enc = nn.Sequential(nn.Conv2d(C, h, 3, padding=1), nn.ReLU(), nn.Conv2d(h, h, 3, padding=1), nn.ReLU())
        self.decoder = SparseFeatureDecoder(C, h)
        self.align = LatencyAwareAlignment(C, h, cfg.max_flow)
        self.fusion = SelfCalibratingFusion(C, h, cfg)
        self.head = nn.Sequential(nn.Conv2d(h, h, 3, padding=1), nn.ReLU(), nn.Conv2d(h, h, 3, padding=2, dilation=2),
                                  nn.ReLU(), nn.Conv2d(h, 1, 1))
        nn.init.constant_(self.head[-1].bias, -2.19)   # focal-loss prior p~0.1

    def forward(self, batch):
        cfg = self.cfg
        ego = batch["ego"]
        e = self.ego_enc(ego)
        out = {}
        if not cfg.cooperate:
            out["logits"] = self.head(e)
            return out
        sp, mk = batch["sparse"], batch["mask"]        # [B,2,C,Hh,Ww], [B,2,1,Hh,Ww]  (frame0=t-d-1, frame1=t-d)
        if cfg.use_recon:                              # 1) semantic completeness
            r0, rec0 = self.decoder(sp[:, 0], mk[:, 0])
            r1, rec1 = self.decoder(sp[:, 1], mk[:, 1])
            out["recon"] = r1; out["recon_raw"] = rec1; out["recon_prev"] = r0; out["recon_prev_raw"] = rec0
        else:
            r0, r1 = sp[:, 0], sp[:, 1]                # old way: zero-filled sparse message as-is
        if cfg.use_flow:                               # 2) temporal consistency
            r1, flow = self.align(r0, r1, batch["delay"]); out["flow"] = flow
        c, off, shifted = self.fusion(ego, r1, e, use_calib=cfg.use_calib)   # 3) spatial alignment + fusion
        out.update(offset=off, shifted=shifted, aligned=r1, logits=self.head(c))
        return out
