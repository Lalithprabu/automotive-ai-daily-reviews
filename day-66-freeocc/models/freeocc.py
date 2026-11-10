"""FreeOcc-style 4D occupancy forecaster (compact BEV re-implementation, PyTorch).

Reference idea: FreeOcc (arXiv 2611.08128, Zhang, Zhao, Metari, Laganiere) -- decouple ego-motion with an explicit
SE(2) warp driven by the planner's intended future pose, forecast only residual agent motion with attention-free
(ConvGRU) fusion, and blend a one-shot discriminative decoder with a flow-matching (ODE) branch by uncertainty.

What is from the paper's public description: SE(2) warp of BEV features with the future ego pose; timestamp-aware
ConvGRU history fusion; recurrent module + residual projector on warped features; discriminative decoder with a
per-voxel log-variance; conditional flow-matching 3-D U-Net branch with smoothness / mass-conservation regularisers;
confidence-weighted blend.  Everything else (sizes, 2-D BEV instead of 3-D voxels, loss weights, blend
parametrisation, Euler steps) is THIS repo's own choice.  Parameter count ~0.1M (paper: 28.3M).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.geometry import warp_bev, inv_pose, rel_pose


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell.  x:(N,Cin,h,w), h:(N,Ch,h,w) -> new h.
    z = sig(Wz[x,h]), r = sig(Wr[x,h]), h~ = tanh(Wh[x, r*h]), h' = (1-z)*h + z*h~."""
    def __init__(self, cin, ch):
        super().__init__()
        self.zr = nn.Conv2d(cin + ch, 2 * ch, 3, padding=1)
        self.hh = nn.Conv2d(cin + ch, ch, 3, padding=1)

    def forward(self, x, h):
        z, r = torch.sigmoid(self.zr(torch.cat([x, h], 1))).chunk(2, 1)
        hn = torch.tanh(self.hh(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * hn


class FlowNet(nn.Module):
    """Small conv U-Net velocity field v_theta(x_t, t | cond).  x_t:(N,1,G,G), cond:(N,C,G,G), t:(N,)."""
    def __init__(self, cc, d=24):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv2d(1 + 1 + cc, d, 3, padding=1), nn.GELU())
        self.down = nn.Sequential(nn.Conv2d(d, 2 * d, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(2 * d, 2 * d, 3, padding=1), nn.GELU())
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(2 * d, d, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(2 * d, 1, 3, padding=1)

    def forward(self, x, t, cond):
        tt = t.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        a = self.inp(torch.cat([x, tt, cond], 1))                   # (N,d,G,G)
        b = self.up(self.down(a))                                   # (N,d,G,G)
        return self.out(torch.cat([a, b], 1))                       # (N,1,G,G)  velocity


class FreeOcc(nn.Module):
    def __init__(self, d=24, use_warp=True, flow_steps=4):
        super().__init__()
        self.d, self.use_warp, self.flow_steps = d, use_warp, flow_steps
        # --- BEV encoder: 48x48 occupancy -> 24x24 latent (stride 2) -------------------------------------------------
        self.enc = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, d, 3, stride=2, padding=1), nn.GELU())
        # --- timestamp-aware ConvGRU: input = [feature, dt-since-newest plane] --------------------------------------
        self.hist_gru = ConvGRUCell(d + 1, d)
        # --- future recurrent module + residual projector (only RESIDUAL scene dynamics is learned) ------------------
        self.fut_gru = ConvGRUCell(1, d)                           # input: horizon-index plane
        self.res_proj = nn.Conv2d(d, d, 1)
        nn.init.zeros_(self.res_proj.weight); nn.init.zeros_(self.res_proj.bias)   # start as 'pure warp'
        # --- decoder trunk (shared by the discriminative head and the flow conditioning) ------------------------------
        self.trunk = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                   nn.Conv2d(d, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 12, 3, padding=1), nn.GELU())
        self.head = nn.Conv2d(12, 2, 1)                            # [occupancy logit, log-variance]
        self.flow = FlowNet(12, d=16)
        # --- uncertainty-aware blend: w = sigmoid(a * (logvar - b)) ---------------------------------------------------
        self.blend_a = nn.Parameter(torch.tensor(2.0))
        self.blend_b = nn.Parameter(torch.tensor(-1.0))

    # ------------------------------------------------------------------------------------------------------------
    def encode_state(self, hist, hist_rel, dt_hist=None):
        """hist (B,T,G,G) corrupted occupancy, each in ITS OWN ego frame; hist_rel (B,T,3) pose of frame i in the current
        frame (last index = identity).  Returns the fused latent state in the CURRENT frame, (B,d,24,24)."""
        B, T = hist.shape[:2]
        z = self.enc(hist.reshape(B * T, 1, *hist.shape[2:]))                       # (B*T,d,24,24)
        z = z.view(B, T, *z.shape[1:])
        h = torch.zeros_like(z[:, 0])
        for i in range(T):
            # move frame i into the current frame (history alignment uses recorded odometry)
            zi = warp_bev(z[:, i], inv_pose(hist_rel[:, i]))
            dt = (dt_hist[:, i] if dt_hist is not None else torch.full((B,), (T - 1 - i) * 0.5, device=hist.device))
            dtp = dt.view(B, 1, 1, 1).expand(-1, 1, *zi.shape[2:])
            h = self.hist_gru(torch.cat([zi, dtp], 1), h)
        return h

    def propagate(self, h, fut_rel):
        """Roll the state forward in FUTURE ego coordinates.  fut_rel (B,H,3): planned pose of every future frame in the
        current frame.  Returns (B,H,d,24,24).  Step tau: warp(h, ego step) then h += proj(GRU(plane_tau, h))."""
        B, Hn = fut_rel.shape[:2]
        prev = torch.zeros_like(fut_rel[:, 0])
        outs = []
        for k in range(Hn):
            if self.use_warp:
                step = rel_pose(prev, fut_rel[:, k])                                 # pose of frame k in frame k-1
                h = warp_bev(h, step)                                                # ego-motion REMOVED analytically
            plane = torch.full_like(h[:, :1], (k + 1) / Hn)
            h = h + self.res_proj(self.fut_gru(plane, h) - h)                        # residual scene dynamics
            outs.append(h)
            prev = fut_rel[:, k]
        return torch.stack(outs, 1)

    def decode(self, hs):
        B, Hn = hs.shape[:2]
        f = self.trunk(hs.flatten(0, 1))                                             # (B*H,12,48,48)
        lo, lv = self.head(f).chunk(2, 1)
        return f, lo.squeeze(1), lv.squeeze(1).clamp(-6, 3)

    # ---- flow matching ------------------------------------------------------------------------------------------
    def flow_loss(self, f, occ_gt, smooth=0.05, mass=0.5):
        """Conditional flow matching:  x_t = (1-t) x0 + t x1,  x1 = 2*occ-1,  x0 ~ N(0,I);  target v = x1 - x0."""
        x1 = (2 * occ_gt - 1).unsqueeze(1)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
        v = self.flow(xt, t, f)
        fm = F.mse_loss(v, x1 - x0)
        tv = (v[..., 1:, :] - v[..., :-1, :]).abs().mean() + (v[..., :, 1:] - v[..., :, :-1]).abs().mean()   # velocity smoothness
        x1_hat = xt + (1 - t.view(-1, 1, 1, 1)) * v                                                         # one-step estimate
        m = ((x1_hat.clamp(-1, 1) + 1) / 2).mean((1, 2, 3)) - occ_gt.mean((1, 2))                            # mass conservation
        return fm + smooth * tv + mass * (m ** 2).mean()

    @torch.no_grad()
    def sample_flow(self, f, steps=None, gen=None):
        """Euler integration of dx/dt = v_theta(x,t | cond) from noise to an occupancy sample in [0,1]."""
        steps = steps or self.flow_steps
        x = torch.randn(f.shape[0], 1, *f.shape[2:], generator=gen)
        for k in range(steps):
            t = torch.full((f.shape[0],), k / steps)
            x = x + self.flow(x, t, f) / steps
        return ((x.squeeze(1).clamp(-1, 1)) + 1) / 2

    def blend_weight(self, logvar):
        return torch.sigmoid(self.blend_a * (logvar - self.blend_b))

    # ------------------------------------------------------------------------------------------------------------
    def forward(self, hist, hist_rel, fut_rel, dt_hist=None, sample=False, gen=None):
        """hist (B,T,G,G), hist_rel (B,T,3), fut_rel (B,H,3) -> dict of (B,H,G,G) maps.
        sample=False -> decoder only (p_dec); sample=True additionally runs the ODE branch and the uncertainty blend."""
        B, Hn = fut_rel.shape[:2]
        h = self.encode_state(hist, hist_rel, dt_hist)
        hs = self.propagate(h, fut_rel)
        f, lo, lv = self.decode(hs)
        out = dict(f=f, logit=lo.view(B, Hn, *lo.shape[1:]), logvar=lv.view(B, Hn, *lv.shape[1:]))
        out["p_dec"] = torch.sigmoid(out["logit"])
        if sample:
            p_ode = self.sample_flow(f, gen=gen).view_as(out["p_dec"])
            w = self.blend_weight(out["logvar"])
            out.update(p_ode=p_ode, w=w, p=(1 - w) * out["p_dec"] + w * p_ode)
        return out
