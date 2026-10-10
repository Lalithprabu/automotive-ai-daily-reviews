"""Train CAA-GMM and the raster-only baseline on the same synthetic data, same steps; then evaluate + robustness.

    python train.py --config config.yaml [--epochs N]
Writes: checkpoints/{caagmm,baseline}.pt, results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from caa_gmm.data import build_raster, make_dataset, perturb_neighbours, take  # noqa: E402
from caa_gmm.losses import gmm_loss, metrics  # noqa: E402
from caa_gmm.model import CAAConfig, CAAGMM, RasterBaseline, count_params  # noqa: E402


def forward(model, batch):
    return model(build_raster(batch), batch["ego"], batch["nb"], batch["mask"])


@torch.no_grad()
def evaluate(model, data, bs=500, **perturb):
    model.eval()
    gen = torch.Generator().manual_seed(123)
    outs, n = {}, len(data["fut"])
    for i in range(0, n, bs):
        b = take(data, torch.arange(i, min(i + bs, n)))
        if perturb:
            b = perturb_neighbours(b, gen=gen, **perturb)
        m = metrics(*forward(model, b), b["fut"])
        for k, v in m.items():
            outs[k] = outs.get(k, 0.0) + v * len(b["fut"]) / n
    return outs


@torch.no_grad()
def lead_attention_share(model, data, bs=500):
    """Mean attention mass on the true lead vehicle (among scenes that have one) vs. uniform over visible slots."""
    model.eval()
    hit, uni, cnt = 0.0, 0.0, 0
    for i in range(0, len(data["fut"]), bs):
        b = take(data, torch.arange(i, min(i + bs, len(data["fut"]))))
        *_, a = model(build_raster(b), b["ego"], b["nb"], b["mask"], return_attn=True)
        sel = b["lead_slot"] >= 0
        if sel.any():
            w = a["w_int"][sel]
            ls = b["lead_slot"][sel]
            hit += w[torch.arange(len(ls)), ls].sum().item()
            uni += (1.0 / (1 + b["mask"][sel].sum(1).float())).sum().item()   # uniform over null + visible
            cnt += int(sel.sum())
    return hit / max(cnt, 1), uni / max(cnt, 1)


def train_one(name, model, tr, va, cfg, seed, comm_aug=False):
    torch.manual_seed(seed)
    t = cfg["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    n = len(tr["fut"])
    steps = t["epochs"] * (n // t["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t["lr"], total_steps=steps)
    best, best_state, step, t0 = 1e9, None, 0, time.time()
    for ep in range(t["epochs"]):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n - t["batch_size"] + 1, t["batch_size"]):
            b = take(tr, perm[i:i + t["batch_size"]])
            if comm_aug:                                   # simulate V2X/perception outages during training
                b = perturb_neighbours(b, drop_p=float(torch.tensor([0.0, 0.0, 0.3, 0.6, 1.0])[torch.randint(0, 5, (1,))]))
            loss, _ = gmm_loss(*forward(model, b), b["fut"], t["relax_eps"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item()
            step += 1
        v = evaluate(model, va)
        if v["minADE"] < best:
            best, best_state = v["minADE"], {k: x.clone() for k, x in model.state_dict().items()}
        if ep % 4 == 0 or ep == t["epochs"] - 1:
            print(f"[{name}] ep {ep:02d} loss {tot / (n // t['batch_size']):.3f} val minADE {v['minADE']:.3f} "
                  f"minFDE {v['minFDE']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--comm-aug", action="store_true", help="train with random neighbour-message dropout")
    ap.add_argument("--tag", default="", help="suffix for checkpoints/results (e.g. _aug)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.epochs:
        cfg["train"]["epochs"] = a.epochs
    torch.set_num_threads(2)
    d = cfg["data"]
    tr, va, te = make_dataset(d["n_train"], 1), make_dataset(d["n_val"], 2), make_dataset(d["n_test"], 3)
    mc = CAAConfig(**cfg["model"])
    models = {"caagmm": CAAGMM(mc), "baseline": RasterBaseline(mc)}
    os.makedirs("checkpoints", exist_ok=True)
    res = {"comm_aug": a.comm_aug, "params": {k: count_params(m) for k, m in models.items()}, "config": cfg}
    for name, m in models.items():
        print(f"{name}: {count_params(m):,} params")
        train_one(name, m, tr, va, cfg, cfg["seed"], a.comm_aug)
        torch.save({"state": m.state_dict(), "cfg": cfg["model"]}, f"checkpoints/{name}{a.tag}.pt")
    r = cfg["robustness"]
    res["test"] = {k: evaluate(m, te) for k, m in models.items()}
    res["robustness"] = {}
    for p in r["drop_probs"]:
        res["robustness"][str(p)] = {k: evaluate(m, te, drop_p=p, pos_noise=0.0) for k, m in models.items()}
    res["robustness"]["noise"] = {k: evaluate(m, te, drop_p=0.0, pos_noise=r["pos_noise_m"])
                                  for k, m in models.items()}
    hit, uni = lead_attention_share(models["caagmm"], te)
    res["lead_attention"] = {"mean_mass_on_lead": hit, "uniform_reference": uni}
    # regime split: scenes with a lead vehicle vs. without
    for tag, sel in (("lead", te["has_lead"]), ("no_lead", ~te["has_lead"])):
        sub = take(te, torch.nonzero(sel).squeeze(1))
        res.setdefault("by_regime", {})[tag] = {k: evaluate(m, sub) for k, m in models.items()}
        res["by_regime"][tag]["n"] = int(sel.sum())
    json.dump(res, open(f"results{a.tag}.json", "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("params", "test", "lead_attention")}, indent=2))


if __name__ == "__main__":
    main()
