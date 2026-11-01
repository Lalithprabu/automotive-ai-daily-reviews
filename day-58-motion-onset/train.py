"""Train full (recurrent) vs stateless (w/o recurrency) GNN on synthetic episodes, evaluate vs baselines.  python train.py [--quick]"""
import argparse, json, time
import numpy as np, torch
from motion_onset.synthetic import batch, CLASSES, H
from motion_onset.model import MotionOnsetGNN, masked_bce
from motion_onset.metrics import bss, pr_auc, ece

ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=500); ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--bs", type=int, default=24); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
if a.quick: a.steps, a.seeds = 40, 1
torch.set_num_threads(2)
T_ = lambda b: {k: torch.as_tensor(v) for k, v in b.items()}


def evaluate(model, b):
    model.eval()
    with torch.no_grad(): p = model.rollout(b["pos"], b["vel"], b["vis"], b["sem"]).numpy()
    q = b["q"].numpy(); return p[q], b["y"].numpy()[q]


# class-frequency baseline (analogue of the paper's class-prior comparator): per-class positive rate on train episodes
tb = batch(np.random.default_rng(99), 300)
rate = np.array([tb["y"][(tb["cls"][:, None, :].repeat(tb["y"].shape[1], 1) == c) & tb["q"]].mean() for c in range(len(CLASSES))])
clim = float(tb["y"][tb["q"]].mean())
val, nov = batch(np.random.default_rng(1000), 150), batch(np.random.default_rng(2000), 150, novel=True)
res = {"climatology_rate": clim, "class_rate": dict(zip(CLASSES, rate.round(4).tolist())), "runs": []}
for seed in range(a.seeds):
    for name, rec in (("full", True), ("stateless", False)):
        torch.manual_seed(seed); rng = np.random.default_rng(seed); m = MotionOnsetGNN(recurrent=rec)
        opt = torch.optim.Adam(m.parameters(), 2e-3); t0 = time.time()
        for it in range(a.steps):
            m.train(); b = T_(batch(rng, a.bs)); p = m.rollout(b["pos"], b["vel"], b["vis"], b["sem"])
            loss = masked_bce(p, b["y"], b["q"].float()); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if it % 50 == 0: print(f"{name} s{seed} it{it} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save(m.state_dict(), f"checkpoints/{name}_s{seed}.pt")
        if seed == 0: torch.save(m.state_dict(), f"checkpoints/{name}.pt")
        for sn, bb in (("val", val), ("novel", nov)):
            p, y = evaluate(m, T_(bb)); res["runs"].append(dict(model=name, seed=seed, set=sn, bss=bss(p, y, clim), pr_auc=pr_auc(p, y), ece=ece(p, y)))
    for sn, bb in (("val", val), ("novel", nov)):                         # class-frequency baseline (no learning)
        cls = np.broadcast_to(bb["cls"][:, None, :], bb["y"].shape)[bb["q"]]; y = bb["y"][bb["q"]]
        p = rate[cls] if sn == "val" else np.full_like(y, clim)             # unseen classes -> fall back to global rate
        res["runs"].append(dict(model="class_freq", seed=seed, set=sn, bss=bss(p, y, clim), pr_auc=pr_auc(p, y), ece=ece(p, y)))
summ = {}
for r in res["runs"]: summ.setdefault((r["model"], r["set"]), []).append(r)
res["summary"] = {f"{k[0]}/{k[1]}": {m: [round(float(np.mean([r[m] for r in v])), 3), round(float(np.std([r[m] for r in v])), 3)] for m in ("bss", "pr_auc", "ece")} for k, v in summ.items()}
json.dump(res, open("checkpoints/results.json", "w"), indent=1); print(json.dumps(res["summary"], indent=1))
