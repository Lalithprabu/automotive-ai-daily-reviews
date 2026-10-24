"""Tiny receding-horizon closed loop: ego follows the first 2 waypoints of each sampled plan (0.5 s), then replans.
Physical decel is clipped at A_MAX so a plan cannot teleport the car to a stop."""
import numpy as np, torch
from models.flow_model import sample
from src.scene import DT, EGO_FRONT, A_MAX, make_context, decode

def run_episode(net, v0, D, guide_lambda, seed, steps=16, margin=0.8, max_t=12.0, record=False):
    g = torch.Generator().manual_seed(seed)
    p, v, hist = 0.0, v0, [v0] * 4
    log, t = [], 0.0
    collided = False
    while t < max_t:
        d_rel = D - p
        c = make_context(torch.tensor([hist[-4:]], dtype=torch.float32), torch.tensor([d_rel], dtype=torch.float32),
                         torch.tensor([d_rel > 0.0]))
        tr = decode(sample(net, c, torch.tensor([d_rel], dtype=torch.float32), torch.tensor([True]), steps,
                           guide_lambda, margin, generator=g))[0].numpy()
        if record: log.append(dict(t=t, p=p, v=v, plan=tr.copy(), d_rel=d_rel))
        s0, s1, s2 = 0.0, tr[0, 0], tr[1, 0]
        for k, sk in enumerate([tr[0, 0], tr[1, 0]]):
            prev = 0.0 if k == 0 else tr[0, 0]
            vk = max(0.0, (sk - prev) / DT)
            vk = max(vk, v - A_MAX * DT)               # physical decel limit
            vk = min(vk, v + 3.0 * DT)                 # physical accel limit
            v = vk; p += v * DT; hist.append(v); t += DT
            if p + EGO_FRONT >= D: collided = True; break
        if collided or (v < 0.1 and t > 1.0): break
    return dict(collided=collided, final_gap=D - EGO_FRONT - p, log=log, v_final=v)
