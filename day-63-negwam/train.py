"""Train + evaluate baseline WAM (flow matching only) vs NegWAM-style (adds near-expert negatives).
Usage:  python train.py [--steps 4000] [--seeds 0 1] [--out checkpoints]
Writes checkpoints/{base,neg}_seed{S}.pt and checkpoints/results.json
"""
from __future__ import annotations
import argparse, json, time, os
import numpy as np
import torch
from negwam.world import build_dataset, norm_scene, norm_traj, denorm_traj, safety, toy_score
from negwam.model import NegWAM
from negwam.losses import training_loss


def to_tensors(D):
    return dict(scene=torch.tensor(norm_scene(D["scene"])), expert=torch.tensor(norm_traj(D["expert"])),
                neg_long=torch.tensor(norm_traj(D["neg_long"])), neg_lat=torch.tensor(norm_traj(D["neg_lat"])),
                lead=torch.tensor(D["lead"] / 60.0))


@torch.no_grad()
def evaluate(model, D, noise_seed=123, steps=8):
    model.eval()
    g = torch.Generator().manual_seed(noise_seed)
    sc_n = torch.tensor(norm_scene(D["scene"]))
    pred = denorm_traj(model.sample(sc_n, steps=steps, generator=g).numpy()).astype(np.float64)
    sc = D["scene"].astype(np.float64)
    s = safety(sc, pred)
    ade = np.linalg.norm(pred - D["expert"], axis=-1).mean()
    return dict(collision=float(s["collision"].mean()), offlane=float(s["offlane"].mean()),
                headway_viol=float((~s["ttc_ok"]).mean()), ade_m=float(ade),
                toy_score=float(toy_score(sc, pred, D["expert"].astype(np.float64)).mean())), pred


def run(use_neg: bool, seed: int, steps: int, bs: int, data, out_dir: str, log):
    torch.manual_seed(seed); np.random.seed(seed)
    model = NegWAM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    tr = data["train_t"]; N = tr["scene"].shape[0]
    t0 = time.time(); hist = []
    for it in range(steps):
        model.train()
        idx = torch.randint(0, N, (bs,))
        batch = {k: v[idx] for k, v in tr.items()}
        loss, parts = training_loss(model, batch, use_neg)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if it % 250 == 0 or it == steps - 1:
            hist.append(dict(step=it, **{k: float(v) for k, v in parts.items()}))
            log(f"[{'neg ' if use_neg else 'base'} s{seed}] step {it:5d} " + " ".join(f"{k}={float(v):.4f}" for k, v in parts.items()) + f"  ({time.time()-t0:.0f}s)")
    tag = f"{'neg' if use_neg else 'base'}_seed{seed}"
    torch.save(model.state_dict(), os.path.join(out_dir, tag + ".pt"))
    res = {}
    for name in ("normal", "hard"):
        res[name], _ = evaluate(model, data[name])
    res["train_seconds"] = time.time() - t0
    res["params"] = sum(p.numel() for p in model.parameters())
    res["history"] = hist
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--ntrain", type=int, default=20000)
    ap.add_argument("--out", default="checkpoints")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(2)
    logf = open(os.path.join(a.out, "train_log.txt"), "w")
    def log(m): print(m, flush=True); logf.write(m + "\n"); logf.flush()
    # data: train from normal scenes + extra hard scenes (so thin-margin cases are represented), eval on held-out sets
    tr_n = build_dataset(a.ntrain, seed=1000); tr_h = build_dataset(a.ntrain // 4, seed=1001, hard=True)
    Dtr = {k: np.concatenate([tr_n[k], tr_h[k]]) for k in tr_n}
    data = dict(train_t=to_tensors(Dtr), normal=build_dataset(2000, seed=2000), hard=build_dataset(1000, seed=2001, hard=True))
    log(f"train scenes: {len(Dtr['scene'])}; eval normal {len(data['normal']['scene'])}, hard {len(data['hard']['scene'])}")
    results = {"base": [], "neg": []}
    for seed in a.seeds:
        for use_neg in (False, True):
            r = run(use_neg, seed, a.steps, a.bs, data, a.out, log)
            results["neg" if use_neg else "base"].append(r)
            log(json.dumps({k: r[k] for k in ("normal", "hard")}))
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    log("done")
