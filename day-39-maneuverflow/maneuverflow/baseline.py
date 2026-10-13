"""'Old way': multi-query regression planner (K learned queries -> K trajectories + a classifier score).
Same SceneEncoder, no intent prior, no generator, no future evidence, no refinement."""
import torch
import torch.nn as nn
from .model import SceneEncoder, mlp, NORM
from .scoring import TF


class MultiQueryPlanner(nn.Module):
    def __init__(self, d=64, K=6):
        super().__init__()
        self.K = K
        self.enc = SceneEncoder(d)
        self.q = nn.Parameter(torch.randn(K, d) * 0.5)
        self.reg = mlp(2 * d, 2 * d, TF * 2)
        self.cls = mlp(2 * d, d, 1)

    def forward(self, hist, agents):
        cond, _, _ = self.enc(hist, agents)
        h = torch.cat([cond[:, None].expand(-1, self.K, -1), self.q[None].expand(cond.shape[0], -1, -1)], -1)
        traj = self.reg(h).view(-1, self.K, TF, 2) * NORM.to(h)
        return traj, self.cls(h).squeeze(-1)

    @torch.no_grad()
    def predict(self, hist, agents, **_):
        traj, lg = self(hist, agents)
        best = lg.argmax(-1)
        return dict(cands=traj, scores=lg.softmax(-1), best=traj[torch.arange(traj.shape[0]), best], best_idx=best)
