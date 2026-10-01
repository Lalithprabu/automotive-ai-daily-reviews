"""Synthetic 3-lane driving scenes with a rule-searching 'expert' whose choices are genuinely multi-modal
(brake behind the lead car, or overtake left/right depending on which lane is free)."""
import numpy as np, torch
from .reward import rule_reward, SCENE_DIM


def sample_scenes(n, rng):
    v0 = rng.uniform(5, 12, n)
    d1 = rng.uniform(14, 34, n); v1 = v0 * rng.uniform(0.0, 0.6, n)
    s = np.zeros((n, SCENE_DIM), dtype=np.float32)
    s[:, 0], s[:, 1], s[:, 2] = v0, d1, v1
    for col, p in ((3, 0.5), (6, 0.5)):
        pres = rng.random(n) < p
        s[:, col] = pres
        s[:, col + 1] = np.where(pres, rng.uniform(-8, 40, n), 100.0)   # in an adjacent lane, near ego or ahead/behind
        s[:, col + 2] = np.where(pres, v0 * rng.uniform(0.5, 1.0, n), 0.0)
    return torch.from_numpy(s)


def parametric_trajs(scene, T=8, dt=0.5):
    """Enumerate 3 target lanes x 7 target speeds x 2 lane-change durations = 42 smooth candidates per scene."""
    B = scene.shape[0]
    t = (torch.arange(T) + 1) * dt
    lanes = torch.tensor([-3.5, 0.0, 3.5]); vf = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.15, 1.3])
    lc = torch.tensor([2.5, 3.5])
    out = []
    for yl in lanes:
        for f in vf:
            for tl in lc:
                v0 = scene[:, 0:1]
                vt = v0 * f
                a = (vt - v0) / (T * dt)
                x = (v0 * t + 0.5 * a * t * t).clamp(min=0)
                x = torch.where(a < 0, torch.minimum(x, v0 * v0 / (-2 * a - 1e-6)).clamp(min=0), x)
                u = (t / tl).clamp(0, 1)
                y = yl * (3 * u ** 2 - 2 * u ** 3)
                out.append(torch.stack([x, y.expand_as(x)], -1))
    return torch.stack(out, 1)   # (B, 42, T, 2)


def make_dataset(n, seed, T=8, dt=0.5):
    rng = np.random.default_rng(seed)
    scene = sample_scenes(n, rng)
    cand = parametric_trajs(scene, T, dt)
    r = rule_reward(scene, cand, dt)
    g = torch.from_numpy(rng.gumbel(size=r.shape).astype(np.float32)) * 0.35   # expert noise -> multimodality among near-ties
    idx = (r + g).argmax(1)
    expert = cand[torch.arange(n), idx]
    return scene, expert
