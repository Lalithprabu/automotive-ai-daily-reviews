"""X-TRACK-style physics-aware trajectory predictor with uncertainty heads.

Pipeline: per-vehicle recurrent encoder -> graph attention over neighbours -> LSTM decoder that
emits mean + log-variance of the *motion variables* u=[a_x, yaw_rate] -> differentiable kinematic
bicycle layer rolls samples of u into positions (so every trajectory is dynamically feasible).
NOTE: paper uses an xLSTM encoder; this reconstruction uses a standard LSTM (disclosed in SOURCING.md).
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

class LockedDropout(nn.Module):
    """Variational ('locked') dropout: one mask per sequence, reused across time. Stays ON at inference
    when mc=True (Monte-Carlo dropout)."""
    def __init__(self, p): super().__init__(); self.p = p; self.mc = False
    def forward(self, x):                       # x: (B, T, H)
        if self.p == 0 or not (self.training or self.mc): return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p) / (1 - self.p)
        return x * mask

class GraphAttention(nn.Module):
    """Single-layer multi-head GAT over a dense (B, N, N) adjacency."""
    def __init__(self, d, heads=2, p=0.0):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.W = nn.Linear(d, d, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.dk) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, self.dk) * 0.1)
        self.drop = nn.Dropout(p)
    def forward(self, x, adj):                  # x: (B,N,D), adj: (B,N,N)
        B, N, D = x.shape
        z = self.W(x).view(B, N, self.h, self.dk)                       # (B,N,h,dk)
        es = (z * self.a_src).sum(-1); ed = (z * self.a_dst).sum(-1)     # (B,N,h)
        e = F.leaky_relu(ed.unsqueeze(2) + es.unsqueeze(1), 0.2)         # (B,N_i,N_j,h)
        e = e.masked_fill(adj.unsqueeze(-1) == 0, float("-inf"))
        a = self.drop(torch.softmax(e, dim=2))                            # attend over neighbours j
        out = torch.einsum("bijh,bjhd->bihd", a, z).reshape(B, N, D)
        return F.elu(out)

class KinematicBicycle(nn.Module):
    """s=[x,y,v,psi]; v+=a*dt, psi+=yaw_rate*dt, x+=v cos(psi) dt, y+=v sin(psi) dt. Fully differentiable.
    u: (..., T, 2) -> positions (..., T, 2)."""
    def __init__(self, dt): super().__init__(); self.dt = dt
    def forward(self, u, v0, psi0):
        T = u.shape[-2]; v = v0; psi = psi0
        x = torch.zeros_like(v); y = torch.zeros_like(v); out = []
        for t in range(T):
            v = torch.clamp(v + u[..., t, 0] * self.dt, min=0.0)
            psi = psi + u[..., t, 1] * self.dt
            x = x + v * torch.cos(psi) * self.dt; y = y + v * torch.sin(psi) * self.dt
            out.append(torch.stack([x, y], -1))
        return torch.stack(out, -2)

class XTrackUQ(nn.Module):
    def __init__(self, hidden=64, t_pred=25, dt=0.2, heads=2, p_drop=0.0, dec_layers=1):
        super().__init__()
        self.t_pred, self.hidden = t_pred, hidden
        self.embed = nn.Linear(2, hidden)
        self.drop_in = LockedDropout(p_drop)
        self.enc = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop_enc = LockedDropout(p_drop)
        self.gat = GraphAttention(hidden, heads)
        self.drop_gat = LockedDropout(p_drop)
        self.dec = nn.LSTM(hidden, hidden, num_layers=dec_layers, batch_first=True)
        self.drop_dec = LockedDropout(p_drop)
        self.head = nn.Linear(hidden, 4)        # mean(ax), mean(yr), logvar(ax), logvar(yr)
        self.kin = KinematicBicycle(dt)
        # stable init: start near zero-motion with moderate variance
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def set_mc(self, flag):
        for m in self.modules():
            if isinstance(m, LockedDropout): m.mc = flag
    def motion_dist(self, hist, adj):
        """hist (B,N,To,2), adj (B,N,N) -> mu (B,Tp,2), logvar (B,Tp,2) of target's future u."""
        B, N, To, _ = hist.shape
        h = self.drop_in(self.embed(hist.reshape(B * N, To, 2)))
        _, (hn, _) = self.enc(h)                                        # (1, B*N, H)
        e = self.drop_enc(hn[-1].view(B, N, -1))                        # (B,N,H)
        g = self.gat(e, adj)
        ctx = self.drop_gat(g[:, :1] + e[:, :1])                        # target node, residual (B,1,H)
        dec_in = ctx.expand(-1, self.t_pred, -1)
        d, _ = self.dec(dec_in)
        o = self.head(self.drop_dec(d))                                 # (B,Tp,4)
        mu = o[..., :2] * torch.tensor([3.0, 0.1], device=o.device)     # scale: accel m/s^2, yaw rate rad/s
        lv = o[..., 2:].clamp(-10, 4)
        return mu, lv
    def sample_traj(self, mu, lv, v0, psi0, n):
        """Reparameterised samples of u (independent per step, as in the paper) rolled through the bicycle layer.
        returns (B,n,Tp,2) positions."""
        std = torch.exp(0.5 * lv) * torch.tensor([1.0, 1.0], device=mu.device)
        eps = torch.randn(mu.shape[0], n, *mu.shape[1:], device=mu.device)
        u = mu.unsqueeze(1) + std.unsqueeze(1) * eps
        return self.kin(u, v0.view(-1, 1).expand(-1, n), psi0.view(-1, 1).expand(-1, n))
    def forward(self, batch, n_samples=16):
        mu, lv = self.motion_dist(batch["hist"], batch["adj"])
        traj = self.sample_traj(mu, lv, batch["v0"], batch["psi0"], n_samples)
        mean_traj = self.kin(mu, batch["v0"], batch["psi0"])
        return dict(mu=mu, logvar=lv, samples=traj, mean_traj=mean_traj)

def gaussian_nll_2d(err, cov, eps=1e-4):
    """err (...,2), cov (...,2,2): 0.5*(log|S| + e^T S^-1 e)."""
    cov = cov + eps * torch.eye(2, device=cov.device)
    det = cov[..., 0, 0] * cov[..., 1, 1] - cov[..., 0, 1] ** 2
    inv00, inv11, inv01 = cov[..., 1, 1] / det, cov[..., 0, 0] / det, -cov[..., 0, 1] / det
    maha = err[..., 0] ** 2 * inv00 + err[..., 1] ** 2 * inv11 + 2 * err[..., 0] * err[..., 1] * inv01
    return 0.5 * (torch.log(det.clamp_min(1e-12)) + maha)

def loss_fn(out, batch, alpha_dyn=0.1):
    """L = L_traj + alpha_dyn * L_dyn (both Gaussian NLL). Trajectory covariance from sample spread."""
    s = out["samples"]                                       # (B,n,Tp,2)
    m = s.mean(1); d = s - m.unsqueeze(1)
    cov = torch.einsum("bnti,bntj->btij", d, d) / (s.shape[1] - 1)
    l_traj = gaussian_nll_2d(batch["pos"] - m, cov).mean()
    l_dyn = 0.5 * (out["logvar"] + (batch["fut"] - out["mu"]) ** 2 / out["logvar"].exp()).mean()
    return l_traj + alpha_dyn * l_dyn, dict(traj=l_traj.item(), dyn=l_dyn.item())
