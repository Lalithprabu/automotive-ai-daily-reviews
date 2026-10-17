"""Frozen, deliberately MAP-BLIND stand-in backbone (K proposals + logits).

It sees only the target agent's own kinematic history, so its proposals carry exactly the kind of
structured residual error (wrong junction distance, wrong turn radius, un-anticipated braking)
that a map-aware refiner can correct. It is NOT QCNet/DenseTNT/DONUT -- see SOURCING.md.
"""
import torch, torch.nn as nn
from .data import T_H, T_F, SCALE


class MapBlindBackbone(nn.Module):
    def __init__(self, K=6, hidden=128):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(nn.Linear(T_H * 4, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, K * (T_F * 2 + 1)))

    def forward(self, batch):
        x = batch["hist"][:, 0].flatten(1) / 10.0                 # (B, T_H*4) target only
        o = self.net(x).view(-1, self.K, T_F * 2 + 1)
        traj = o[..., :-1].view(-1, self.K, T_F, 2) * SCALE       # metres
        return traj, o[..., -1]                                   # (B,K,T,2), (B,K)


def backbone_loss(traj, logits, gt):
    err = (traj - gt[:, None]).norm(dim=-1).mean(-1)              # (B,K) ADE per mode
    best = err.argmin(1)
    w = torch.full_like(err, 0.05 / traj.shape[1]); w.scatter_(1, best[:, None], 0.95)  # relaxed WTA
    reg = (w * torch.nn.functional.smooth_l1_loss(
        traj, gt[:, None].expand_as(traj), reduction="none").mean((-1, -2))).sum(1).mean()
    cls = torch.nn.functional.cross_entropy(logits, best)
    return reg + 0.1 * cls
