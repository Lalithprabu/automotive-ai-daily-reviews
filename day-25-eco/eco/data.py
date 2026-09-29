"""Open-loop behaviour-cloning dataset on synthetic routes."""
from __future__ import annotations
import numpy as np
import torch
from .env import Route, expert_waypoints, observe_route, build_obs


def make_dataset(n, cfg, seed):
    rng = np.random.default_rng(seed)
    T, H, dt = cfg["horizon"], cfg["history"], cfg["wp_dt"]
    X, Y = [], []
    while len(X) < n:
        route = Route(rng)
        for _ in range(20):
            s0 = rng.uniform(5, route.length - 80)
            rx, ry, rth, _ = route.at(s0)
            off = rng.normal(0, 0.6)
            x, y = rx - off * np.sin(rth), ry + off * np.cos(rth)
            th = rth + rng.normal(0, 0.1)
            v = rng.uniform(3, 10.5)
            pts = observe_route(route, x, y, th, s0, cfg["route_points"], cfg["route_spacing"])
            hist = np.stack([[-dt * (H - j) * v, rng.normal(0, 0.05)] for j in range(H)])
            wp = expert_waypoints(route, x, y, th, v, T, dt, s0)
            X.append(build_obs(pts, v, hist))
            Y.append(wp.astype(np.float32))
    X, Y = np.stack(X[:n]), np.stack(Y[:n])
    return torch.from_numpy(X), torch.from_numpy(Y)
