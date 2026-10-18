"""Small Transformer pedestrian forecaster (stand-in for the paper's 'foundation model')."""
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)

    def forward(self, x):  # x: (B,T,d)
        B, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)               # each (B,T,d)
        sp = lambda t: t.view(B, T, self.h, d // self.h).transpose(1, 2)  # (B,h,T,dh)
        a = torch.softmax(sp(q) @ sp(k).transpose(-1, -2) / (d // self.h) ** 0.5, -1)  # (B,h,T,T)
        o = (a @ sp(v)).transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(o)
        return x + self.ff2(torch.relu(self.ff1(self.n2(x))))


class TrajTransformer(nn.Module):
    """obs (B,T_obs,2) relative-to-last-position displacements -> future offsets (B,T_pred,2)."""

    def __init__(self, obs_len=8, pred_len=12, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.pred_len = pred_len
        self.embed = nn.Linear(4, d_model)           # (pos, vel) per step
        self.pos = nn.Parameter(torch.zeros(obs_len, d_model))
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len * 2)

    def forward(self, obs):
        vel = torch.cat([torch.zeros_like(obs[:, :1]), obs[:, 1:] - obs[:, :-1]], 1)  # (B,T,2)
        x = self.embed(torch.cat([obs, vel], -1)) + self.pos                           # (B,T,d)
        for b in self.blocks:
            x = b(x)
        z = self.norm(x)[:, -1]                                                         # (B,d)
        return self.head(z).view(-1, self.pred_len, 2)                                 # (B,P,2)
