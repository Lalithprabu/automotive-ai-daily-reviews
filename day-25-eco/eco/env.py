"""Synthetic closed-loop driving world: curved route, kinematic bicycle, waypoint-tracking controller,
expert (for open-loop BC data + ground-truth overlay). Pure numpy. 100% synthetic, this repo's own design."""
from __future__ import annotations
import numpy as np

L_WHEELBASE = 2.7
SIM_DT = 0.1


class Route:
    """Centreline built by integrating a sum-of-sinusoids curvature profile."""
    def __init__(self, rng: np.random.Generator, length: float = 400.0, ds: float = 0.5):
        n = int(length / ds)
        s = np.arange(n) * ds
        a1, a2 = rng.uniform(0.008, 0.022), rng.uniform(0.004, 0.012)
        p1, p2 = rng.uniform(0, 2 * np.pi, 2)
        k = a1 * np.sin(s / rng.uniform(25, 45) + p1) + a2 * np.sin(s / rng.uniform(10, 18) + p2)
        th = np.cumsum(k) * ds
        self.s, self.k, self.th = s, k, th
        self.x = np.cumsum(np.cos(th)) * ds
        self.y = np.cumsum(np.sin(th)) * ds
        self.ds, self.length = ds, s[-1]

    def at(self, s):
        s = np.clip(s, 0, self.length - 1e-6)
        i = np.clip((np.asarray(s) / self.ds).astype(int), 0, len(self.s) - 2)
        f = (np.asarray(s) / self.ds) - i
        lerp = lambda a: a[i] * (1 - f) + a[i + 1] * f
        return lerp(self.x), lerp(self.y), lerp(self.th), lerp(self.k)

    def project(self, x, y, s_hint=None, window=60.0):
        lo, hi = 0, len(self.s)
        if s_hint is not None:
            lo, hi = max(0, int((s_hint - 10) / self.ds)), min(len(self.s), int((s_hint + window) / self.ds))
        d = (self.x[lo:hi] - x) ** 2 + (self.y[lo:hi] - y) ** 2
        j = lo + int(np.argmin(d))
        th = self.th[j]
        lat = -(x - self.x[j]) * np.sin(th) + (y - self.y[j]) * np.cos(th)   # + = left of route
        return self.s[j], lat


def to_ego(px, py, x, y, th):
    dx, dy = px - x, py - y
    c, s = np.cos(th), np.sin(th)
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=-1)


def from_ego(pe, x, y, th):
    c, s = np.cos(th), np.sin(th)
    return np.stack([x + c * pe[..., 0] - s * pe[..., 1], y + s * pe[..., 0] + c * pe[..., 1]], axis=-1)


def v_desired(route: Route, s, vmax=10.0, a_lat=2.5):
    k = np.abs(route.at(s)[3]) + 1e-3
    return np.minimum(vmax, np.sqrt(a_lat / k))


def expert_waypoints(route, x, y, th, v, T, dt, s_hint=None):
    """Expert plan: follow the route, speed profile from curvature, lateral offset decays exponentially."""
    s0, lat = route.project(x, y, s_hint)
    heading_err = th - route.at(s0)[2]
    s, vv, out = s0, v, []
    for k in range(1, T + 1):
        vt = float(v_desired(route, s))
        vv += np.clip(vt - vv, -2.0 * dt, 2.0 * dt)
        s += max(vv, 0.5) * dt
        rx, ry, rth, _ = route.at(s)
        off = lat * np.exp(-0.5 * k * dt * 1.2)
        out.append([rx - off * np.sin(rth), ry + off * np.cos(rth)])
    return to_ego(np.array(out)[:, 0], np.array(out)[:, 1], x, y, th)   # (T,2)


def observe_route(route, x, y, th, s_hint, n, spacing, rng=None, noise=0.0):
    s0, _ = route.project(x, y, s_hint)
    ss = s0 + spacing * np.arange(1, n + 1)
    rx, ry, _, _ = route.at(ss)
    pts = to_ego(rx, ry, x, y, th)
    if noise > 0 and rng is not None:
        pts = pts + rng.normal(0, noise, pts.shape)
    return pts   # (n,2) ego frame


def build_obs(route_pts, v, hist_ego):
    """Flat policy input: route pts (n*2) + speed + history (H*2)."""
    return np.concatenate([route_pts.ravel(), [v], hist_ego.ravel()]).astype(np.float32)


class Tracker:
    """Waypoint-consuming controller (pure-pursuit steering + speed from first-segment length)."""
    def __init__(self, wp_dt=0.5, look_idx=2):
        self.wp_dt, self.look_idx = wp_dt, look_idx

    def __call__(self, wps, v):
        v_ref = np.linalg.norm(wps[0]) / self.wp_dt
        tgt = wps[min(self.look_idx, len(wps) - 1)]
        ld = max(np.linalg.norm(tgt), 2.0)
        alpha = np.arctan2(tgt[1], max(tgt[0], 1e-3))
        steer = float(np.clip(np.arctan2(2 * L_WHEELBASE * np.sin(alpha), ld), -0.5, 0.5))
        acc = float(np.clip(1.5 * (v_ref - v), -4.0, 3.0))
        return steer, acc


def step_bicycle(state, steer, acc):
    x, y, th, v = state
    x += v * np.cos(th) * SIM_DT
    y += v * np.sin(th) * SIM_DT
    th += v / L_WHEELBASE * np.tan(steer) * SIM_DT
    v = max(0.0, v + acc * SIM_DT)
    return np.array([x, y, th, v])
