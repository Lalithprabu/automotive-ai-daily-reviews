"""Synthetic VRU (pedestrian) world with three 'weather' environments for leave-one-environment-out shift tests.

NOT IMPTC / Waymo / nuScenes / inD data.  A window is L=32 steps (dt=0.1 s) x F=5 channels:
  0 speed (m/s)  1 body turn-rate (rad/s)  2 head-minus-body yaw (rad)  3 acceleration (m/s^2)  4 distance-to-curb (m)
Intent (what happens right after the window): 0 straight, 1 left, 2 right, 3 stop.  The manoeuvre starts at step s0 = L - U(6,12),
so the window only contains the first 6-12 steps of it plus leading cues (head turns before the body, speed decays for 'stop').
Environments differ in sensor noise, dropped steps and cue reliability; 'fog' is held out as the OOD test environment.
"""
import numpy as np, torch

L, F, DT = 32, 5, 0.1
CLASSES = ("straight", "left", "right", "stop")
ENVS = ("clear", "rain", "fog")
ENV_CFG = {  # noise scale, dropped-step prob, head-yaw cue reliability, mean speed factor
    "clear": dict(noise=2.0, drop=0.02, cue=1.0, spd=1.0),
    "rain": dict(noise=3.0, drop=0.08, cue=0.7, spd=0.9),
    "fog": dict(noise=5.0, drop=0.22, cue=0.35, spd=0.8),
}
SIGMA = np.array([0.05, 0.04, 0.05, 0.12, 0.10], np.float32)         # base per-channel observation noise
SCALE = np.array([1.5, 0.8, 0.5, 1.0, 3.0], np.float32)              # feature normalisation


def track(cls, T, s0, env, rng):
    """Clean features [T,F] and positions [T,2] for one pedestrian walking along +y. psi>0 = turning left (towards -x)."""
    cfg = ENV_CFG[env]
    t = np.arange(T) - s0                                             # steps since manoeuvre onset
    on = np.clip(t, 0, None) * DT
    v0 = rng.normal(1.4 * cfg["spd"], 0.15)
    w = rng.uniform(0.3, 0.9) * (1 if cls == 1 else -1 if cls == 2 else 0)          # turn rate
    tau = rng.uniform(0.5, 1.0)
    speed = v0 * (np.exp(-on / tau) if cls == 3 else np.ones(T))
    speed = speed * (1 + 0.03 * np.sin(np.arange(T) * 0.7 + rng.uniform(0, 6)))
    psi = rng.normal(0, 0.05) + w * on
    # head leads the body by ~0.5 s for turns; for 'stop' pedestrians glance at traffic (oscillation); weaker cue in bad weather
    lead = np.clip(t + 5, 0, None) * DT
    head = np.where(cls in (1, 2), 0.7 * np.sign(w) * np.minimum(lead, 0.6), 0.0) * cfg["cue"] * np.ones(T)
    if cls == 3: head = head + 0.35 * cfg["cue"] * np.sin(np.arange(T) * 0.55) * (t > -6)
    if cls == 0 and rng.random() < 0.35:                                # distractor: a straight walker glances sideways
        k = int(rng.integers(8, T - 3)); head[k:k + 4] += rng.choice([-1, 1]) * 0.5 * cfg["cue"]
    head = head + rng.normal(0, 0.03, T)
    accel = np.gradient(speed, DT)
    curb0 = rng.uniform(3.0, 6.0)
    curb = np.clip(curb0 - np.cumsum(speed * np.cos(psi)) * DT * (0.5 if cls != 3 else 1.0), 0.3, None)
    if cls == 3: curb = np.minimum(curb, 0.6 + 0.3 * np.exp(-np.arange(T) * 0.05))
    vel = np.stack([-speed * np.sin(psi), speed * np.cos(psi)], 1)
    pos = np.cumsum(vel, 0) * DT
    feats = np.stack([speed, np.gradient(psi, DT), head, accel, curb], 1).astype(np.float32)
    return feats, pos.astype(np.float32)


def observe(feats, env, rng):
    """Add env-dependent noise and dropped steps.  -> X [T,F] normalised, valid [T] bool."""
    cfg = ENV_CFG[env]
    x = feats + rng.normal(0, 1, feats.shape).astype(np.float32) * SIGMA * cfg["noise"]
    valid = rng.random(len(x)) >= cfg["drop"]
    x = x / SCALE
    x[~valid] = 0
    return x.astype(np.float32), valid


def make_dataset(n, envs, seed, min_len=18):
    rng = np.random.default_rng(seed)
    X, V, Y, E = [], [], [], []
    for i in range(n):
        env = envs[i % len(envs)] if len(envs) > 1 else envs[0]
        cls = int(rng.integers(0, 4))
        f, _ = track(cls, L, L - int(rng.integers(3, 9)), env, rng)
        x, v = observe(f, env, rng)
        n_pad = int(rng.integers(0, L - min_len + 1)) if rng.random() < 0.3 else 0   # some windows are shorter (padded at the start)
        if n_pad: x[:n_pad] = 0; v[:n_pad] = False
        X.append(x); V.append(v); Y.append(cls); E.append(ENVS.index(env))
    return dict(X=torch.tensor(np.stack(X)), valid=torch.tensor(np.stack(V)), y=torch.tensor(Y), env=torch.tensor(E))
