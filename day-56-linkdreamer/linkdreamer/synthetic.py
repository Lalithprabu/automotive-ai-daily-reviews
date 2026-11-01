"""Synthetic intermittent-V2X world (this repo's own toy, NOT V2X-Real).

Ego-centred BEV window: x in [XMIN, XMIN+G*CELL) forward, y in [-G*CELL/2, G*CELL/2) lateral.
Ego drives at EGO_SPEED m/frame (= EGO_CELLS cells).  A collaborator vehicle drives ahead and sees what
the ego cannot (trucks cast angular shadows).  Each agent renders a 2-channel BEV feature:
    ch0 = object evidence (Gaussian blobs: vehicles ~1.0, static poles ~0.55 = false positives)
    ch1 = visibility mask (1 = cell inside sensor range and not in a truck's shadow)
Half of the vehicles are parked (static in the world), half move: that is what makes ego-motion-compensated
reuse good for some objects and wrong for others.
"""
import numpy as np
import torch

G, CELL = 24, 2.0                      # BEV size [cells], metres per cell
XMIN, YMIN = -8.0, -24.0               # window origin in the ego frame [m]
EGO_SPEED = 4.0                        # m / frame
EGO_CELLS = int(EGO_SPEED / CELL)      # ego displacement per frame in cells (=2, integer -> exact warp)
R_SENS = 34.0                          # sensor range [m]
T_EP = 24                              # frames per episode
SIG = 2.0                              # blob sigma [m]

_xs = XMIN + (np.arange(G) + .5) * CELL
_ys = YMIN + (np.arange(G) + .5) * CELL
GX, GY = np.meshgrid(_xs, _ys, indexing="ij")          # [G,G]  row i = forward, col j = lateral


def visible(px, py, ax, ay, occ, skip=-1):
    """Boolean visibility of points (px,py) from agent (ax,ay); occ = [K,3] (x, y, half-width) truck shadows."""
    r = np.hypot(px - ax, py - ay)
    ok = r < R_SENS
    ang = np.arctan2(py - ay, px - ax)
    for k, (ox, oy, hw) in enumerate(occ):
        if k == skip:
            continue
        ro = np.hypot(ox - ax, oy - ay)
        d = np.abs((ang - np.arctan2(oy - ay, ox - ax) + np.pi) % (2 * np.pi) - np.pi)
        ok &= ~((r > ro + 2.5) & (d < np.arctan2(hw, max(ro, 1.0))))
    return ok


def make_episode(rng, T=T_EP):
    """Returns dict: ego [T,2,G,G], collab [T,2,G,G], tgt [T,G,G], objs list of [n,2] (ego-frame x,y) per frame."""
    n = 24
    x0 = rng.uniform(-8, 140, n); y0 = rng.uniform(-20, 20, n)
    moving = rng.random(n) < .5
    vx = np.where(moving, rng.uniform(2.0, 6.0, n), 0.0) - EGO_SPEED     # ego-frame velocity [m/frame]
    vy = np.where(moving, rng.uniform(-.3, .3, n), 0.0)
    px0 = rng.uniform(-8, 140, 12); py0 = rng.uniform(-22, 22, 12); pamp = rng.uniform(.45, .65, 12)
    trucks_e = np.stack([rng.uniform(6, 14, 2), np.array([rng.uniform(-9, -4), rng.uniform(4, 9)])], 1)   # ego-frame
    c0 = np.array([rng.uniform(16, 28), rng.uniform(-8, 8)]); crv = rng.uniform(-.8, .8)
    truck_c = np.array([rng.uniform(5, 9), rng.uniform(-6, 6)])                                             # rel. to collab
    ego = np.zeros((T, 2, G, G), np.float32); col = np.zeros_like(ego); tgt = np.zeros((T, G, G), np.float32); objs = []
    for t in range(T):
        cars = np.stack([x0 + vx * t, y0 + vy * t], 1)
        poles = np.stack([px0 - EGO_SPEED * t, py0], 1)
        c = c0 + np.array([crv * t, 0.0])
        trucks = np.concatenate([trucks_e, (c + truck_c)[None]], 0)
        occ = np.concatenate([trucks, np.full((3, 1), 2.0)], 1)                       # shadow casters (half-width 2 m)
        allv = np.concatenate([cars, trucks], 0)                                      # every GT vehicle
        inwin = (allv[:, 0] >= XMIN) & (allv[:, 0] < XMIN + G * CELL) & (np.abs(allv[:, 1]) < G * CELL / 2)
        objs.append(allv[inwin].astype(np.float32))
        for ai, (ax, ay) in enumerate(((0.0, 0.0), tuple(c))):
            vis_map = visible(GX, GY, ax, ay, occ).astype(np.float32)
            ev = np.zeros((G, G), np.float32)
            for k, (ox, oy) in enumerate(allv):
                if visible(np.array(ox), np.array(oy), ax, ay, occ, skip=(k - len(cars)) if k >= len(cars) else -1) \
                        and np.hypot(ox - ax, oy - ay) > 1.0:
                    ev += rng.uniform(.9, 1.1) * np.exp(-((GX - ox) ** 2 + (GY - oy) ** 2) / (2 * SIG ** 2))
            for (ox, oy), a in zip(poles, pamp):
                if visible(np.array(ox), np.array(oy), ax, ay, occ):
                    ev += a * np.exp(-((GX - ox) ** 2 + (GY - oy) ** 2) / (2 * SIG ** 2))
            ev = np.clip((ev + rng.normal(0, .04, ev.shape)) * vis_map, 0, 1.3)
            (ego if ai == 0 else col)[t] = np.stack([ev, vis_map])
        for ox, oy in objs[-1]:
            tgt[t] = np.maximum(tgt[t], np.exp(-((GX - ox) ** 2 + (GY - oy) ** 2) / (2 * (CELL * .8) ** 2)))
    return dict(ego=ego, collab=col, tgt=tgt, objs=objs)


def make_dataset(n, seed, T=T_EP):
    rng = np.random.default_rng(seed)
    eps = [make_episode(rng, T) for _ in range(n)]
    return dict(ego=torch.tensor(np.stack([e["ego"] for e in eps])), collab=torch.tensor(np.stack([e["collab"] for e in eps])),
                tgt=torch.tensor(np.stack([e["tgt"] for e in eps])), objs=[e["objs"] for e in eps])


def shift_rows(x, k):
    """Ego-motion compensation: express a BEV tensor [B,C,G,G] recorded k[B] cells ago in the *current* ego frame.
    out[:, :, i] = x[:, :, i + k]  (static world content slides backwards when the ego drives forward); exposed rows = 0."""
    B, C = x.shape[:2]
    idx = torch.arange(G, device=x.device)[None, :] + k[:, None].long()          # [B,G]
    valid = (idx < G).float()[:, None, :, None]
    out = torch.gather(x, 2, idx.clamp(max=G - 1)[:, None, :, None].expand(B, C, G, G))
    return out * valid


def link_sequence(rng, E, T=T_EP, warm=2):
    """Markov link quality q in {1.0 (up), 0.5 (degraded), 0.0 (down)}; first `warm` frames forced up."""
    q = np.ones((E, T), np.float32)
    for e in range(E):
        s = 0
        for t in range(T):
            if t >= warm:
                u = rng.random()
                if s == 0: s = 2 if u < .12 else (1 if u < .20 else 0)
                elif s == 2: s = 0 if u < .25 else (1 if u < .35 else 2)
                else: s = 0 if u < .30 else (2 if u < .45 else 1)
            q[e, t] = (1.0, 0.5, 0.0)[s]
    return q
