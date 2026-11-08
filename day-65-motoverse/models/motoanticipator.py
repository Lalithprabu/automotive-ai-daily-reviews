"""MotoAnticipator: causal ego-rider maneuver anticipation + ego-motion-compensated agent forecasting.

This is MY reference design motivated by the MotoVerse benchmark (arXiv 2611.06007) findings -- the
benchmark paper itself defines tasks (ERMA / TATP) and a challenge taxonomy, not this architecture.
"""
import torch
import torch.nn as nn

DT = 0.2


class EgoMotionCompensation(nn.Module):
    """Removes the rider's own motion from apparent agent velocity.

    In the ego frame, d p/dt = v_agent - v_ego - w x p   (w x p = w*[-p_y, p_x]).
    Hence v_agent ~= dp/dt + [v_ego, 0] + w*[-p_y, p_x]. `gain` (learnable, init 1) lets the net down-weight
    the correction when IMU speed / yaw-rate are noisy. use_comp=False gives the apparent velocity.
    """
    def __init__(self, use_comp=True):
        super().__init__()
        self.use_comp = use_comp
        self.gain = nn.Parameter(torch.ones(1))

    def forward(self, pos, ego):                       # pos (B,T,K,2) ; ego (B,T,2)=[v, w]
        dp = torch.zeros_like(pos)
        dp[:, 1:] = (pos[:, 1:] - pos[:, :-1]) / DT    # apparent velocity (B,T,K,2); frame 0 -> 0
        if not self.use_comp:
            return dp
        v, w = ego[..., 0:1, None], ego[..., 1:2, None]  # (B,T,1,1)
        rot = torch.stack([-pos[..., 1], pos[..., 0]], -1)          # (B,T,K,2) = [-p_y, p_x]
        fwd = torch.cat([v.expand_as(pos[..., :1]), torch.zeros_like(pos[..., :1])], -1)
        return dp + self.gain * (fwd + w * rot)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class MotoAnticipator(nn.Module):
    def __init__(self, d=64, n_man=4, horizon=10, use_comp=True):
        super().__init__()
        self.H, self.d = horizon, d
        self.comp = EgoMotionCompensation(use_comp)
        self.ego_enc = mlp(2, d, d)
        self.ag_enc = mlp(4, d, d)
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)          # ego-conditioned attention over agents
        self.gru = nn.GRU(2 * d, d, batch_first=True)              # strictly causal in time
        self.man_head = nn.Linear(d, n_man)                        # a maneuver prediction at EVERY frame
        self.traj_head = mlp(2 * d + 4, 2 * d, horizon * 2)        # per-agent residual over const. velocity
        nn.init.zeros_(self.traj_head[-1].weight); nn.init.zeros_(self.traj_head[-1].bias)  # start at CV
        self.register_buffer("tau", torch.arange(1, horizon + 1).float() * DT)

    def forward(self, ego, pos, L=None):
        """ego (B,T,2)=[speed m/s, yaw-rate rad/s]; pos (B,T,K,2) ego-frame agent positions (m).
        L (B,) = number of observed frames (<=T); forecasts are decoded from frame L-1.
        Returns logits (B,T,n_man), pred (B,K,H,2), cv (B,K,H,2) [compensated const-velocity baseline]."""
        B, T, K, _ = pos.shape
        v_ag = self.comp(pos, ego)                                              # (B,T,K,2)
        feat = torch.cat([pos / 30.0, v_ag / 10.0], -1)                         # (B,T,K,4)
        e = self.ego_enc(torch.stack([ego[..., 0] / 8.0, ego[..., 1]], -1))     # (B,T,d)
        a = self.ag_enc(feat)                                                   # (B,T,K,d)
        att = torch.softmax((self.q(e)[:, :, None] * self.k(a)).sum(-1) / self.d ** 0.5, -1)   # (B,T,K)
        ctx = (att[..., None] * a).sum(2)                                       # (B,T,d)
        h, _ = self.gru(torch.cat([e, ctx], -1))                                # (B,T,d) causal
        logits = self.man_head(h)                                               # (B,T,n_man)
        if L is None:
            L = torch.full((B,), T, dtype=torch.long, device=pos.device)
        ar = torch.arange(B, device=pos.device)
        hl, al, fl = h[ar, L - 1], a[ar, L - 1], feat[ar, L - 1]                # state at the last observed frame
        z = torch.cat([hl[:, None].expand(-1, K, -1), al, fl], -1)              # (B,K,2d+4)
        res = self.traj_head(z).view(B, K, self.H, 2) * 5.0                     # metres
        p, v = pos[ar, L - 1], v_ag[ar, L - 1]
        cv = p[:, :, None] + v[:, :, None] * self.tau[None, None, :, None]
        return logits, cv + res, cv
