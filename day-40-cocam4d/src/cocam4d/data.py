"""Synthetic cooperative-camera scenes. Depth-axis noise grows ~ range^2 (stereo/monocular depth
behaviour); lateral noise grows ~ range. Objects move with constant velocity + accel noise."""
from __future__ import annotations
import numpy as np

H_OBJ, H_AGENT = 0.8, 1.5


def simulate_scene(rng, n_obj, n_steps, dt, extent=60.0):
    pos0 = rng.uniform(-extent, extent, (n_obj, 3)); pos0[:, 2] = H_OBJ
    dyn = rng.random(n_obj) < 0.7
    speed = np.where(dyn, rng.uniform(2.0, 15.0, n_obj), 0.0)
    head = rng.uniform(0, 2 * np.pi, n_obj)
    vel = np.stack([speed * np.cos(head), speed * np.sin(head), np.zeros(n_obj)], 1)
    P = np.zeros((n_steps, n_obj, 3)); V = np.zeros_like(P)
    p = pos0.copy(); v = vel.copy()
    for t in range(n_steps):
        P[t], V[t] = p, v
        a = rng.normal(0, 0.5, (n_obj, 3)) * dyn[:, None]; a[:, 2] = 0
        v = v + a * dt; p = p + v * dt
    return P, V, dyn


def sample_agents(rng, n_agents, extent=40.0):
    A = rng.uniform(-extent, extent, (n_agents, 3)); A[:, 2] = H_AGENT
    return A


def observe(rng, agent_pos, traj, sensor_range=45.0):
    """traj [T,N,3] (true positions) -> meas [T,N,3], vis [N] (in range at last frame), feats [T,N,7]."""
    d = traj - agent_pos                                      # [T,N,3]
    D = np.linalg.norm(d, axis=-1, keepdims=True)             # [T,N,1]
    u = d / np.maximum(D, 1e-6)                               # line-of-sight unit vector
    sd = 0.0005 * D ** 2 + 0.05                               # depth-axis sigma
    sl = 0.01 * D + 0.03                                      # lateral sigma
    eps = rng.normal(size=traj.shape)
    eps_d = rng.normal(size=D.shape)
    # noise = depth along u + lateral isotropic projected onto plane perpendicular to u
    lat = eps - (eps * u).sum(-1, keepdims=True) * u
    meas = traj + eps_d * sd * u + lat * sl
    vis = D[-1, :, 0] < sensor_range
    rel = meas - meas[-1:]                                    # translation-invariant history
    feats = np.concatenate([rel, D / 50.0, u], axis=-1)       # [T,N,7]
    return meas, vis, feats.astype(np.float32)


def build_training_set(rng, n_scenes, cfg):
    X, V, DYN, PE = [], [], [], []
    T, dt = cfg["seq_len"], cfg["dt"]
    for _ in range(n_scenes):
        P, Vel, dyn = simulate_scene(rng, cfg["n_objects"], cfg["n_steps"], dt)
        A = sample_agents(rng, cfg["n_agents"])
        for a in A:
            end = rng.integers(T - 1, cfg["n_steps"])
            win = P[end - T + 1:end + 1]
            meas, vis, feats = observe(rng, a, win, cfg["sensor_range"])
            for i in np.where(vis)[0]:
                X.append(feats[:, i]); V.append(Vel[end, i]); DYN.append(float(dyn[i]))
                PE.append(meas[-1, i] - win[-1, i])
    f = lambda a: np.asarray(a, dtype=np.float32)
    return f(X), f(V), f(DYN), f(PE)
