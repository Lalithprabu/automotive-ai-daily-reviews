"""Synthetic 'kerbside' world for motion-ONSET prediction (my own toy, NOT the paper's IsaacSim warehouse).

N road users are first static (parked car, delivery van, pedestrian waiting, ...). Each static user starts to move at a random
frame. The hazard of starting depends on (a) a class base rate, (b) an EPISODE-specific class mobility m_c and a global tempo g that are
hidden (so the model must infer them in context from what it has watched), and (c) a relational cue: a same-class neighbour that just
started moving makes you more likely to move ("queue release").  Class semantics are given as fixed-prototype embeddings with
per-episode jitter (augmentation), so the net cannot memorise class -> behaviour.
"""
import numpy as np

CLASSES = ["pedestrian", "cyclist", "parked_car", "delivery_van", "e_scooter", "traffic_cone"]
NOVEL = ["skateboarder", "tow_truck", "food_cart"]          # unseen classes (fresh prototypes + base rates) for the novel-object test
D, N, T, H, DT = 32, 14, 48, 6, 0.5                         # embed dim, objects, frames, horizon (3 s), seconds/frame
AREA = (40.0, 14.0)                                         # metres (x, y)


def _protos(n, seed):
    p = np.random.default_rng(seed).normal(size=(n, D))
    return p / np.linalg.norm(p, axis=1, keepdims=True)


PROTO, NPROTO = _protos(len(CLASSES), 0), _protos(len(NOVEL), 1)
BASE = np.array([.06, .05, .010, .018, .035, .0])           # per-frame onset hazard (traffic cone never moves)
NBASE = np.array([.05, .012, .03])
SPEED = np.array([1.3, 4.0, 3.0, 2.5, 3.0, 0.0])
NSPEED = np.array([3.5, 2.0, 1.0])


def episode(rng, novel=False):
    """One episode. Returns dict of numpy arrays (T=frames, N=objects)."""
    P, B, S, names = (NPROTO, NBASE, NSPEED, NOVEL) if novel else (PROTO, BASE, SPEED, CLASSES)
    K = len(names)
    cls = rng.integers(0, K, N)
    mob = np.exp(rng.normal(0, .7, K))                      # hidden per-episode class mobility
    g = np.exp(rng.normal(0, .5))                           # hidden global tempo
    jit = P + rng.normal(0, .15, P.shape); jit /= np.linalg.norm(jit, axis=1, keepdims=True)   # per-episode class-consistent jitter
    sem = jit[cls] + rng.normal(0, .05, (N, D)); sem /= np.linalg.norm(sem, axis=1, keepdims=True)
    pos0 = np.stack([rng.uniform(2, AREA[0] - 2, N), rng.uniform(1, AREA[1] - 1, N)], 1)
    ang = rng.uniform(0, 2 * np.pi, N)
    dirv = np.stack([np.cos(ang), np.sin(ang)], 1) * S[cls][:, None]
    appear = rng.integers(0, 10, N)                         # frame at which each object first becomes visible
    pos, vel = np.zeros((T, N, 2)), np.zeros((T, N, 2))
    onset = np.full(N, 10 ** 6); moving = np.zeros(N, bool); since = np.full(N, 99)
    p = pos0.copy()
    for t in range(T):
        d = np.linalg.norm(p[:, None] - p[None], axis=-1)
        same = (cls[:, None] == cls[None]) & (d < 10) & ~np.eye(N, dtype=bool)
        recent = (same & (since[None] <= 4) & moving[None]).sum(1)          # same-class neighbours that began moving in the last 2 s
        h = np.clip(B[cls] * mob[cls] * g * (1 + 4 * recent), 0, .6)
        start = (~moving) & (rng.random(N) < h) & (t >= appear)
        onset[start] = t; moving |= start; since[start] = 0
        since[moving & ~start] += 1
        v = np.where(moving[:, None], dirv, 0.0)
        p = p + v * DT * moving[:, None]
        pos[t], vel[t] = p, v + rng.normal(0, .03, (N, 2))                 # noisy observed velocity
    vis = np.arange(T)[:, None] >= appear[None]
    t_idx = np.arange(T)[:, None]
    static = onset[None] > t_idx                                            # still static at frame t
    y = static & (onset[None] <= t_idx + H)                                 # will start moving within the next H frames
    q = vis & static & (t_idx > appear[None])                               # query: visible, static, velocity observed >= 1 frame
    return dict(pos=pos, vel=vel, vis=vis, sem=sem, y=y, q=q, cls=cls, onset=onset, names=names, appear=appear)


def batch(rng, n, novel=False):
    eps = [episode(rng, novel) for _ in range(n)]
    f = lambda k, dt: np.stack([e[k] for e in eps]).astype(dt)
    return dict(pos=f("pos", np.float32), vel=f("vel", np.float32), vis=f("vis", bool), sem=f("sem", np.float32),
                y=f("y", np.float32), q=f("q", bool), cls=f("cls", np.int64), onset=f("onset", np.int64))
