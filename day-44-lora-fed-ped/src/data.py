"""Synthetic pedestrian data. Source domain: steady walkers. Client domain: curb stop-and-go crossers
with client-specific speed / wait time (non-IID). Positions in metres at 2.5 Hz (0.4 s steps)."""
import numpy as np
import torch

DT = 0.4


def _walker(rng, T):
    v = rng.uniform(0.9, 1.6); th = rng.uniform(0, 2 * np.pi); w = rng.normal(0, 0.05)
    p = np.zeros((T, 2)); pos = np.zeros(2)
    for t in range(T):
        th += w + rng.normal(0, 0.02)
        pos = pos + v * DT * np.array([np.cos(th), np.sin(th)]); p[t] = pos
    return p


def _crosser(rng, T, speed, wait):
    """Walks along curb, pauses `wait` steps at a random time, then turns ~90 deg and crosses."""
    th = rng.uniform(-0.3, 0.3); v = speed * rng.uniform(0.9, 1.1)
    t_stop = rng.integers(4, 10); pos = np.zeros(2); p = np.zeros((T, 2)); turn = rng.choice([-1, 1])
    for t in range(T):
        if t_stop <= t < t_stop + wait:
            vel = 0.05
        else:
            if t == t_stop + wait: th += turn * np.pi / 2
            vel = v * (1.15 if t >= t_stop + wait else 1.0)
        th += rng.normal(0, 0.02)
        pos = pos + vel * DT * np.array([np.cos(th), np.sin(th)]); p[t] = pos
    return p


def _pack(trajs, obs_len):
    arr = np.stack(trajs).astype(np.float32)                 # (N,T,2)
    ref = arr[:, obs_len - 1:obs_len]                         # last observed position
    rel = arr - ref
    return torch.from_numpy(rel[:, :obs_len]), torch.from_numpy(rel[:, obs_len:]), torch.from_numpy(arr)


def make_source(n, obs_len, pred_len, seed):
    rng = np.random.default_rng(seed)
    return _pack([_walker(rng, obs_len + pred_len) for _ in range(n)], obs_len)


def client_profile(cid, n_clients):
    """Deterministic per-client (speed, wait) so clients are heterogeneous."""
    f = cid / max(1, n_clients - 1)
    return 1.0 + 0.5 * f, int(2 + 6 * (1 - f))


def make_client(cid, n_clients, n, obs_len, pred_len, seed):
    rng = np.random.default_rng(seed * 1000 + cid)
    sp, wt = client_profile(cid, n_clients)
    return _pack([_crosser(rng, obs_len + pred_len, sp, wt) for _ in range(n)], obs_len)


def make_target_val(n, obs_len, pred_len, n_clients, seed):
    rng = np.random.default_rng(seed + 777); trajs = []
    for i in range(n):
        sp, wt = client_profile(i % n_clients, n_clients)
        trajs.append(_crosser(rng, obs_len + pred_len, sp, wt))
    return _pack(trajs, obs_len)
