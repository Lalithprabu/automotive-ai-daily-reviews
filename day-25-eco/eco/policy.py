"""Small waypoint-regression policy: the stand-in for 'a behaviour-cloned end-to-end driving policy'.
Trained with per-waypoint L1 (the waypoint-supervision recipe the ECO paper argues is insufficient)."""
from __future__ import annotations
import torch
import torch.nn as nn


class WaypointPolicy(nn.Module):
    def __init__(self, n_route=8, history=4, horizon=8, hidden=128):
        super().__init__()
        self.in_dim = n_route * 2 + 1 + history * 2
        self.T, self.H, self.n_route = horizon, history, n_route
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, horizon * 2))          # each waypoint regressed independently (no smoothness prior)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B, in_dim) -> waypoints (B, T, 2) in the ego frame."""
        return self.net(obs).view(-1, self.T, 2)

    def history_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, self.n_route * 2 + 1:].view(-1, self.H, 2)
