"""Open-loop selection quality + closed-loop driving for any scorer."""
import numpy as np, torch
from .world import World, make_batch, ANCHORS, N_ANCH


def to_t(B): return {k: torch.as_tensor(v) for k, v in B.items()}


@torch.no_grad()
def select_quality(model, n=400, seed=999, bs=50):
    """Held-out scenes: true PDMS of the argmax-predicted anchor vs oracle / random / straight; score correlation."""
    rng = np.random.default_rng(seed); A = torch.as_tensor(ANCHORS)[None]
    sel, orc, rnd, stt, cors = [], [], [], [], []
    straight = int(np.argmin(np.abs(np.arange(N_ANCH) // 16 - 7.5) + 1e-3 * np.abs(np.arange(N_ANCH) % 16 - 9)))
    for _ in range(n // bs):
        B = to_t(make_batch(rng, bs)); out = model(B, A.expand(bs, -1, -1, -1)); p, t = out["pdms"], B["t_pdms"]
        i = p.argmax(1)
        sel.append(t.gather(1, i[:, None]).squeeze(1)); orc.append(t.max(1).values); rnd.append(t.mean(1)); stt.append(t[:, straight])
        pc, tc = p - p.mean(1, keepdim=True), t - t.mean(1, keepdim=True)
        cors.append((pc * tc).sum(1) / (pc.norm(dim=1) * tc.norm(dim=1) + 1e-8))
    f = lambda L: float(torch.cat(L).mean())
    return dict(selected=f(sel), oracle=f(orc), random=f(rnd), straight_mid=f(stt), score_corr=f(cors))


@torch.no_grad()
def closed_loop(model, n=100, steps=14, seed=5):
    """Drive `steps` x 0.5 s: pick the best anchor each step, execute its first waypoint. Counts collisions/off-road."""
    rng = np.random.default_rng(seed); A = torch.as_tensor(ANCHORS)[None]
    col = off = 0; prog = []
    for _ in range(n):
        w = World(rng); x0 = w.ego[0]; hit = out_of_road = False
        for _ in range(steps):
            o = w.observe(); B = to_t({k: v[None] for k, v in o.items()})
            i = int(model(B, A)["pdms"].argmax(1)); hit |= w.step(ANCHORS[i]); out_of_road |= not bool(w.on_road(w.ego))
        col += hit; off += out_of_road; prog.append(w.ego[0] - x0)
    return dict(collision_rate=col / n, offroad_rate=off / n, progress_m=float(np.mean(prog)))
