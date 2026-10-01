"""EMPlan-style planner (project reconstruction from an abstract-level description; see SOURCING.md).

Pipeline:  scene -> encoder -> [anchor scores (K logits)] + [offset head (K,T,2)]
           trajectory_k = anchor_k + offset_k ;  inference picks argmax-score candidate.
Stage 1: imitation (CE on nearest anchor + L1 on that anchor's offset).
Stage 2: reward-guided UNPAIRED preference fine-tuning (see train.py).
"""
import torch, torch.nn as nn
from .reward import SCENE_DIM


def _mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.GELU()]; d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class SceneEncoder(nn.Module):
    """(B, 9) -> (B, hidden). Inputs are scaled to ~unit range."""
    def __init__(self, hidden):
        super().__init__()
        self.register_buffer("scale", torch.tensor([10., 30., 10., 1., 40., 10., 1., 40., 10.]))
        self.net = _mlp(SCENE_DIM, hidden, hidden)

    def forward(self, s):
        return self.net(s / self.scale)


class EMPlan(nn.Module):
    def __init__(self, anchors, hidden=128):
        super().__init__()
        K, T, _ = anchors.shape
        self.K, self.T = K, T
        self.register_buffer("anchors", anchors.clone())              # (K,T,2) sparse trajectory vocabulary
        self.enc = SceneEncoder(hidden)
        self.score = nn.Linear(hidden, K)                              # anchor logits
        self.offset = _mlp(hidden, hidden, K * T * 2, n=1)             # per-anchor offset refinement

    def forward(self, scene):
        h = self.enc(scene)                                            # (B,H)
        logits = self.score(h)                                         # (B,K)
        off = self.offset(h).view(-1, self.K, self.T, 2)               # (B,K,T,2)
        trajs = self.anchors[None] + off                               # (B,K,T,2) all K candidates
        return logits, trajs

    @torch.no_grad()
    def plan(self, scene):
        logits, trajs = self.forward(scene)
        k = logits.argmax(-1)
        return trajs[torch.arange(len(k)), k], k


class RegressionPlanner(nn.Module):
    """Old way: single-mode waypoint regression with L1 loss (mode-averages multimodal demos)."""
    def __init__(self, T=8, hidden=128):
        super().__init__()
        self.T = T
        self.enc = SceneEncoder(hidden)
        self.head = nn.Linear(hidden, T * 2)

    def forward(self, scene):
        return self.head(self.enc(scene)).view(-1, self.T, 2)

    @torch.no_grad()
    def plan(self, scene):
        return self.forward(scene)


def kmeans_anchors(trajs, K, iters=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = trajs.flatten(1)
    c = X[torch.randperm(len(X), generator=g)[:K]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for k in range(K):
            m = a == k
            if m.any():
                c[k] = X[m].mean(0)
    return c.view(K, *trajs.shape[1:])
