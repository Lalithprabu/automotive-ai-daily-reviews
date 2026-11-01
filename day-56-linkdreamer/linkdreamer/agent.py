"""Training-batch sampling, cost model and the closed-loop episode runner shared by train.py / simulate.py / tests."""
import numpy as np
import torch

from .synthetic import G, T_EP
from .models import COMM, LAT, build_modes, decision_state, mode_mask
from .metrics import decode, match, ap_f1

LAMBDA_C, LAMBDA_L, LAMBDA_U = 0.10, 0.015, 0.20      # communication / latency / uncertainty weights (toy-tuned)
RISK_REF = {"v": 1.0}                                  # mean ego-only detection loss, set in train.py


def sample_batch(ds, rng, B, ages=None, K=2):
    """Random (episode, time) samples with a K=2 memory: frames at tau1-g and tau1, current frame t = tau1 + age."""
    N, T = ds["ego"].shape[:2]
    ep = rng.integers(0, N, B)
    age = rng.integers(1, 9, B) if ages is None else np.full(B, ages)
    g = rng.integers(1, 4, B)
    tau1 = np.array([rng.integers(gi, T - ai) for gi, ai in zip(g, age)])
    t = tau1 + age
    hist = torch.stack([ds["collab"][ep, tau1 - g], ds["collab"][ep, tau1]], 1)            # [B,2,2,G,G]
    return dict(ep=ep, t=t, hist=hist, gaps=torch.tensor(g[:, None]), ages=torch.tensor(age), ego=ds["ego"][ep, t],
                col=ds["collab"][ep, t], tgt=ds["tgt"][ep, t], has_mem=torch.ones(B, dtype=torch.bool))


def cost_table(risk, state, ages=None):
    """risk [N,5] (detector BCE under each mode) -> total cost [N,5]; state [N,5] = (u, comp, fresh, q, budget).
    cost = risk/ref + lc*(1+2*(1-budget))*comm + ll*lat + lu*u*[imagine]"""
    u, b = state[:, 0:1], state[:, 4:5]
    imag = torch.tensor([0., 0., 1., 0., 0.])[None]
    return risk / RISK_REF["v"] + LAMBDA_C * (1 + 2 * (1 - b)) * COMM[None] + LAMBDA_L * LAT[None] + LAMBDA_U * u * imag


@torch.no_grad()
def run(ds, idx, q, policy, wm, det, ctrl=None, budget_frac=0.5, hysteresis=0.3, trace=False):
    """Closed-loop evaluation on episodes `idx`.  q [E,T] link quality.  policy in
    {'ego','reuse','imagine','ctrl','upper'}.  Returns dict with modes, comm, detections and pooled AP/F1 per link state."""
    E, T = len(idx), T_EP
    ego, col, objs = ds["ego"][idx], ds["collab"][idx], [ds["objs"][i] for i in idx]
    qT = torch.tensor(q)
    mem = torch.zeros(E, 2, 2, G, G); mem_t = torch.full((E, 2), -10, dtype=torch.long); n_mem = torch.zeros(E, dtype=torch.long)
    budget0 = budget_frac * T; budget = torch.full((E,), budget0); prev = torch.zeros(E, dtype=torch.long)
    modes = torch.zeros(E, T, dtype=torch.long); comm = torch.zeros(E, T); dets = [[None] * T for _ in range(E)]
    tr = []
    ar = torch.arange(E)
    for t in range(T):
        has_mem = n_mem > 0
        ages = torch.where(has_mem, t - mem_t[:, 1], torch.zeros(E, dtype=torch.long)).clamp(min=0)
        gaps = torch.where(n_mem >= 2, mem_t[:, 1] - mem_t[:, 0], torch.ones(E, dtype=torch.long)).clamp(1, 8)[:, None]
        qt = torch.ones(E) if policy == "upper" else qT[:, t]
        coll, sig, wsig, reuse = build_modes(wm, col[:, t], mem, gaps, ages, has_mem)
        mask = mode_mask(qt, has_mem)
        state = decision_state(ego[:, t], reuse, wsig, ages, has_mem, qt, (budget / budget0).clamp(0, 1))
        if policy == "ctrl":
            sc = ctrl(state) + hysteresis * torch.nn.functional.one_hot(prev, 5).float()
            m2 = mask & (COMM[None] <= budget[:, None] + 1e-6)
            m2[:, 0] = True
            mode = sc.masked_fill(~m2, -1e9).argmax(1)
        elif policy == "ego": mode = torch.zeros(E, dtype=torch.long)
        elif policy == "reuse": mode = torch.where(qt >= 1, 4, torch.where(has_mem, 1, 0))
        elif policy == "imagine": mode = torch.where(qt >= 1, 4, torch.where(has_mem, 2, 0))
        else: mode = torch.full((E,), 4)
        if t < 2: mode = torch.full((E,), 4)                                          # warm-up: fill the memory
        logits = det(ego[:, t], coll[ar, mode], sig[ar, mode], mode)
        for e, d in enumerate(decode(logits, 0.3)):
            dets[e][t] = d
        modes[:, t], comm[:, t] = mode, COMM[mode]
        if trace: tr.append(dict(state=state.clone(), coll=coll[ar, mode].clone(), logits=logits.clone(), wsig=wsig.clone()))
        upd = mode == 4
        newest = col[:, t]
        mem = torch.where(upd[:, None, None, None, None], torch.stack([mem[:, 1], newest], 1), mem)
        mem_t = torch.where(upd[:, None], torch.stack([mem_t[:, 1], torch.full((E,), t)], 1), mem_t)
        n_mem = torch.where(upd, (n_mem + 1).clamp(max=2), n_mem)
        budget = budget - COMM[mode]; prev = mode
    out = dict(modes=modes, comm=comm, dets=dets, q=q, trace=tr)
    out.update(score(dets, objs, q))
    out["comm_per_frame"] = float(comm[:, 2:].mean())
    return out


def score(dets, objs, q, t0=2):
    """Pooled AP/F1 over frames t>=t0, split by link state."""
    res = {}
    for name, sel in (("all", lambda v: True), ("outage", lambda v: v == 0.0), ("degraded", lambda v: v == 0.5), ("up", lambda v: v == 1.0)):
        rec, n = [], 0
        for e in range(len(dets)):
            for t in range(t0, len(dets[e])):
                if sel(q[e, t]):
                    r, miss = match(dets[e][t], [tuple(o) for o in objs[e][t]])
                    rec += r; n += len(objs[e][t])
        res[name] = ap_f1(rec, n)
    return res
