"""Closed-loop rollout shared by evaluate.py and simulate.py. use_eco toggles the ONLY difference."""
from __future__ import annotations
import numpy as np
import torch
from .env import (Route, observe_route, build_obs, to_ego, expert_waypoints, Tracker, step_bicycle, SIM_DT)
from .metrics import episode_metrics

REPLAN_EVERY = 1   # replan every sim step (0.1 s) so waypoints are always in the current ego frame


def run_episode(policy, eco, cfg, seed, use_eco, record=False, max_steps=None):
    rng = np.random.default_rng(seed)
    route = Route(rng)
    T, H, dt = cfg["horizon"], cfg["history"], cfg["wp_dt"]
    noise = cfg["eval"]["perception_noise"]
    max_steps = max_steps or cfg["eval"]["max_steps"]
    rx, ry, rth, _ = route.at(2.0)
    off = rng.normal(0, 0.4)
    state = np.array([rx - off * np.sin(rth), ry + off * np.cos(rth), rth + rng.normal(0, 0.05), 6.0])
    trail = [state[:2].copy()]            # executed positions every sim step
    tracker, wps = Tracker(dt), None
    lat, speeds, steers, frames = [], [state[3]], [0.0], []
    s_hint, crashed, steer, acc = 0.0, False, 0.0, 0.0
    obs_rng = np.random.default_rng(seed + 999)
    for t in range(max_steps):
        x, y, th, v = state
        if t % REPLAN_EVERY == 0:
            s_hint, _ = route.project(x, y, s_hint)
            pts = observe_route(route, x, y, th, s_hint, cfg["route_points"], cfg["route_spacing"], obs_rng, noise)
            idx = [max(0, len(trail) - 1 - int(round((H - j) * dt / SIM_DT))) for j in range(H)]
            hist_w = np.array([trail[i] for i in idx])
            hist = to_ego(hist_w[:, 0], hist_w[:, 1], x, y, th)          # executed history, ego frame
            obs = torch.from_numpy(build_obs(pts, v, hist))[None]
            with torch.no_grad():
                raw = policy(obs)
                out = eco(raw, torch.from_numpy(hist.astype(np.float32))[None]) if use_eco else raw
            wps = out[0].numpy()
            if record:
                gt = expert_waypoints(route, x, y, th, v, T, dt, s_hint)
                frames.append(dict(t=t, state=state.copy(), pts=pts, raw=raw[0].numpy(), eco=out[0].numpy(),
                                   gt=gt, trail=np.array(trail)))
        steer, acc = tracker(wps, v)
        state = step_bicycle(state, steer, acc)
        trail.append(state[:2].copy())
        s_hint, l = route.project(state[0], state[1], s_hint)
        lat.append(l); speeds.append(state[3]); steers.append(steer)
        if abs(l) > 1.75:
            crashed = True
            break
        if s_hint > route.length - 5:
            break
    m = episode_metrics(lat, speeds, steers, s_hint / route.length, crashed)
    m["frames"] = frames
    m["speed_trace"], m["lat_trace"] = np.array(speeds), np.array(lat)
    return m
