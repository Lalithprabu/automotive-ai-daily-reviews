"""Train the INTERACT-style world model (stages 1-2) then the planners (stage 3), evaluate, write results.json.
    python train.py            # full run  (~25-35 min on 2 CPU cores)
    python train.py --quick    # smoke test (~2 min)
"""
import argparse, json, time
import numpy as np, torch, torch.nn.functional as F
from interact.synthetic import make_dataset, NA
from interact.model import WorldModel, SlotPlanner, DensePlanner, candidate_futures
from interact.evaluate import open_loop_eval, closed_loop_eval, ground_actions, encode_all

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=2500); ap.add_argument("--psteps", type=int, default=1500)
ap.add_argument("--ntrain", type=int, default=1200); ap.add_argument("--out", default="checkpoints")
a = ap.parse_args()
if a.quick: a.steps, a.psteps, a.ntrain = 60, 60, 60
torch.set_num_threads(2); torch.set_flush_denormal(True); torch.manual_seed(a.seed); np.random.seed(a.seed)
t0 = time.time(); log = lambda *x: print(f"[{time.time() - t0:6.0f}s]", *x, flush=True)

tr = make_dataset(a.ntrain, 1000 + a.seed); va = make_dataset(40 if a.quick else 150, 7777)
log("data", tr["frames"].shape)
fr = lambda d, idx: torch.as_tensor(d["frames"][idx]).float() / 255.0

wm = WorldModel(); opt = torch.optim.Adam(wm.parameters(), lr=4e-4)
Lc, B = 4, 12
for step in range(a.steps):
    lr = 6e-4 * min(1.0, (step + 1) / 100) * (0.5 * (1 + np.cos(np.pi * step / a.steps)) * 0.9 + 0.1)
    for g in opt.param_groups: g["lr"] = lr
    e = np.random.randint(0, len(tr["frames"]), B); s0 = np.random.randint(0, 11 - Lc + 1, B)
    x = torch.stack([fr(tr, e[i])[s0[i]:s0[i] + Lc] for i in range(B)])                      # [B,5,3,48,48]
    S = wm.enc(x)                                                                           # [B,5,K,d]
    sel = [0, 2]
    rec, _ = wm.dec(S[:, sel].flatten(0, 1))
    tgt = x[:, sel].flatten(0, 1); l_rec = (((rec - tgt) ** 2) * (1 + 5 * tgt)).mean() * 50      # foreground-weighted MSE: objects are ~4% of pixels
    w_pred = float(np.clip((step - 0.4 * a.steps) / (0.2 * a.steps), 0, 1))
    loss, info = l_rec, dict(rec=l_rec.item())
    if w_pred > 0:
        tau = max(0.5, 1.0 - step / a.steps)
        codes, probs = [], []
        for t in range(Lc - 1):
            c, p, _ = wm.lat(S[:, t], S[:, t + 1], tau); codes.append(c); probs.append(p)
        one = torch.stack([wm.pred(S[:, t], codes[t]) for t in range(Lc - 1)], 1)           # one-step predictions of S_{t+1}
        ln = lambda z: F.layer_norm(z, z.shape[-1:])
        l_slot = F.mse_loss(ln(one), ln(S[:, 1:].detach())) * 10                              # scale-free: cannot be zeroed by shrinking slots
        cur, fut = S[:, 0], []
        for t in range(2):                                                                  # 2-step free rollout from S_0
            cur = wm.pred(cur, codes[t]); fut.append(cur)
        pi, _ = wm.dec(torch.stack(fut, 1).flatten(0, 1))
        tp = x[:, 1:3].flatten(0, 1); l_pimg = (((pi - tp) ** 2) * (1 + 5 * tp)).mean() * 50
        pm = torch.stack(probs).mean((0, 1)); l_ent = (pm * (pm + 1e-8).log()).sum()        # -H(code usage): avoid dead codes
        loss = l_rec + w_pred * (l_slot + l_pimg + 0.3 * l_ent)
        info.update(slot=l_slot.item(), pimg=l_pimg.item(), ent=-l_ent.item())
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0); opt.step()
    if step % 100 == 0 or step == a.steps - 1: log(f"wm step {step} " + " ".join(f"{k}={v:.3f}" for k, v in info.items()))
wm.eval(); torch.save(wm.state_dict(), f"{a.out}/wm.pt")

# ---- stage 3: ground latent actions (small action-labelled set), precompute slots + candidate futures, distil the planner
aemb, purity = ground_actions(wm, tr)
log("latent-action purity vs true prototype", round(purity, 3))
Str, Fut_tr = encode_all(wm, tr, aemb)                                                       # [N,11,K,d], [N,10,A,2,K,d]
Sva, Fut_va = encode_all(wm, va, aemb)
Ctr = torch.as_tensor(tr["costs"]); Cva = torch.as_tensor(va["costs"])
N, T = tr["act"].shape
planners = {"slot_full": SlotPlanner(use_future=True), "slot_nofuture": SlotPlanner(use_future=False), "dense": DensePlanner()}
for name, pl in planners.items():
    o = torch.optim.Adam(pl.parameters(), lr=1e-3)
    for step in range(a.psteps):
        e = torch.randint(0, N, (128,)); t = torch.randint(0, T, (128,))
        if name == "dense":
            fprev = torch.as_tensor(tr["frames"][e.numpy(), np.maximum(t.numpy() - 1, 0)]).float() / 255.0
            fcur = torch.as_tensor(tr["frames"][e.numpy(), t.numpy()]).float() / 255.0
            pred = pl(fprev, fcur)
        else:
            pred = pl(Str[e, t], Fut_tr[e, t].float(), aemb)
        loss = F.smooth_l1_loss(pred, Ctr[e, t]); o.zero_grad(); loss.backward(); o.step()
    with torch.no_grad():                                                                    # val agreement with the oracle's choice
        e, t = torch.meshgrid(torch.arange(len(va["act"])), torch.arange(T), indexing="ij"); e, t = e.flatten(), t.flatten()
        if name == "dense":
            pv = pl(torch.as_tensor(va["frames"][e.numpy(), np.maximum(t.numpy() - 1, 0)]).float() / 255.0, torch.as_tensor(va["frames"][e.numpy(), t.numpy()]).float() / 255.0)
        else:
            pv = pl(Sva[e, t], Fut_va[e, t].float(), aemb)
        agree = (pv.argmin(-1) == Cva[e, t].argmin(-1)).float().mean().item()
        regret = (Cva[e, t].gather(1, pv.argmin(-1)[:, None]).squeeze(1) - Cva[e, t].min(-1).values).mean().item()
    log(f"planner {name}: val agreement with oracle {agree:.3f}, mean cost regret {regret:.3f}")
    planners[name].val = dict(agree=agree, regret=regret)
torch.save({k: v.state_dict() for k, v in planners.items()} | {"aemb": aemb}, f"{a.out}/planners.pt")

res = dict(args=vars(a), latent_action_purity=purity, planner_val={k: v.val for k, v in planners.items()})
res["open_loop"] = open_loop_eval(wm, va, aemb)
log("open loop", json.dumps(res["open_loop"]))
res["closed_loop"] = closed_loop_eval(wm, planners, aemb, n=40 if a.quick else 300)
log("closed loop", json.dumps(res["closed_loop"], indent=1))
json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)
log("done")
