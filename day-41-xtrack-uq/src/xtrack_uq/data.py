"""Synthetic highway scenes (NOT highD). Each sample: target + K neighbours, history of
motion variables u=[a_x, yaw_rate], future of the same, plus ground-truth positions.
Heteroscedastic noise grows with horizon; the target's accel depends on the lead neighbour,
so interaction (GAT) carries real signal."""
import numpy as np

def _rollout(v0, psi0, ax, yr, dt):
    x = np.zeros(len(ax)); y = np.zeros(len(ax)); v = v0; psi = psi0; px = py = 0.0
    for t in range(len(ax)):
        v = max(v + ax[t] * dt, 0.0); psi = psi + yr[t] * dt
        px += v * np.cos(psi) * dt; py += v * np.sin(psi) * dt
        x[t], y[t] = px, py
    return np.stack([x, y], -1)

def make_dataset(n, cfg, seed, noise_scale=1.0):
    rng = np.random.default_rng(seed)
    To, Tp, K, dt = cfg["t_obs"], cfg["t_pred"], cfg["n_neighbors"], cfg["dt"]
    T = To + Tp
    hist = np.zeros((n, K + 1, To, 2), np.float32)   # index 0 = target
    fut = np.zeros((n, Tp, 2), np.float32)
    pos = np.zeros((n, Tp, 2), np.float32)
    v0s = np.zeros((n, K + 1), np.float32)
    adj = np.zeros((n, K + 1, K + 1), np.float32)
    for i in range(n):
        v0 = rng.uniform(18, 32, K + 1)
        # neighbour motion: smooth random accel + occasional lane change pulse
        u = np.zeros((K + 1, T, 2))
        for k in range(1, K + 1):
            ax = np.convolve(rng.normal(0, 1.2, T + 8), np.ones(5) / 5, "same")[4:4 + T]
            yr = np.zeros(T)
            if rng.random() < 0.3:
                c = rng.integers(5, T - 10); s = rng.choice([-1, 1]) * rng.uniform(0.03, 0.08)
                yr[c:c + 5] = s; yr[c + 5:c + 10] = -s
            u[k, :, 0], u[k, :, 1] = ax, yr
        # target: follows lead neighbour (k=1 is lead) + own manoeuvre
        lead_ax = u[1, :, 0]
        gain = rng.uniform(0.4, 0.9)
        manv = rng.choice(["keep", "keep", "brake", "lc"], p=[.35, .25, .2, .2])
        ax = gain * lead_ax + rng.normal(0, 0.15, T)
        yr = np.zeros(T)
        if manv == "brake":
            c = rng.integers(To - 3, To + 8); ax[c:c + 8] -= rng.uniform(2, 3.5)
        if manv == "lc":
            c = rng.integers(To - 5, To + 8); s = rng.choice([-1, 1]) * rng.uniform(0.04, 0.09)
            yr[c:c + 5] = s; yr[c + 5:c + 10] = -s
        # heteroscedastic aleatoric noise on the FUTURE only (unobservable in history)
        h = np.linspace(0, 1, Tp)
        ax[To:] += noise_scale * rng.normal(0, 1, Tp) * (0.1 + 0.5 * h) * (1.5 if manv != "keep" else 0.6)
        yr[To:] += noise_scale * rng.normal(0, 1, Tp) * (0.003 + 0.012 * h) * (1.5 if manv == "lc" else 0.5)
        u[0, :, 0], u[0, :, 1] = ax, yr
        hist[i] = u[:, :To].astype(np.float32)
        fut[i] = u[0, To:].astype(np.float32)
        v0s[i] = v0
        # target state at end of history (for the kinematic rollout start)
        v = v0[0]; psi = 0.0
        for t in range(To):
            v = max(v + ax[t] * dt, 0.0); psi += yr[t] * dt
        pos[i] = _rollout(v, psi, ax[To:], yr[To:], dt).astype(np.float32)
        v0s[i, 0] = v; v0s[i, 1:] = v0[1:]
        psis = np.zeros(K + 1)
        # distance-based adjacency: lead & lane neighbours strongest
        d = rng.uniform(10, 60, (K + 1,)); d[1] = rng.uniform(10, 30)
        adj[i, 0, 1:] = (d[1:] < 50).astype(np.float32); adj[i, 0, 0] = 1
        adj[i, 1:, 0] = 1; adj[i, np.arange(K + 1), np.arange(K + 1)] = 1
    # initial heading at prediction start for the target (psi at end of history)
    psi0 = np.zeros(n, np.float32)
    for i in range(n):
        psi0[i] = hist[i, 0, :, 1].sum() * dt
    return dict(hist=hist, fut=fut, pos=pos, v0=v0s[:, 0], psi0=psi0, adj=adj)

def to_tensors(d, device="cpu"):
    import torch
    return {k: torch.as_tensor(v, device=device) for k, v in d.items()}
