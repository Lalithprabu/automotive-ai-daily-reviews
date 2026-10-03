"""Train MomWorld (new way) and the single-latent-state baseline (old way) with identical data/budget.

Usage: python train.py --config config.yaml [--epochs N]
Outputs: outputs/{momworld,baseline}_seed{S}.pt, outputs/results.json
"""
import argparse, json, time, copy
import numpy as np, torch, yaml
from momworld import MomWorld, MomWorldConfig, make_dataset
from momworld.data import index_batch
from momworld.metrics import ade_fde, collision, near_far_ade, stale_momentum_extrapolation


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.hist)
        plan = out["plan"]
        ade, fde = ade_fde(plan, data.future); nr, fr = near_far_ade(plan, data.future)
        col = collision(plan, data.lead_future)
        ev = data.event
        res = dict(ade=ade.mean().item(), fde=fde.mean().item(), collision=col.float().mean().item(),
                   ade_event=ade[ev].mean().item(), ade_free=ade[~ev].mean().item(),
                   collision_event=col[ev].float().mean().item(), collision_free=col[~ev].float().mean().item(),
                   ade_near=nr.mean().item(), ade_far=fr.mean().item(),
                   keep_event=out["keep"][ev].mean().item() if model.cfg.use_momentum else None,
                   keep_free=out["keep"][~ev].mean().item() if model.cfg.use_momentum else None)
        if model.moflow is not None:
            res["fusion_w"] = model.moflow.fusion_weights().tolist()
    return res


def fit(cfg_m, tr, va, tcfg, seed, tag):
    torch.manual_seed(seed)
    model = MomWorld(cfg_m)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    n = tr.hist.shape[0]; bs = tcfg["batch_size"]; E = tcfg["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tcfg["lr"], total_steps=E * ((n + bs - 1) // bs))
    best, best_state, t0 = 1e9, None, time.time()
    for ep in range(E):
        model.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, bs):
            b = index_batch(tr, perm[i:i + bs])
            loss, parts = model.compute_loss(b.hist, b.future, b.lead_future)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item() * len(b.hist)
        v = evaluate(model, va)
        if v["ade"] < best: best, best_state = v["ade"], copy.deepcopy(model.state_dict())
        if ep % 5 == 0 or ep == E - 1:
            print(f"[{tag} s{seed}] ep {ep:02d} train_loss {tot/n:.4f} val_ade {v['ade']:.3f} val_col {v['collision']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.epochs: cfg["epochs"] = a.epochs
    tcfg = dict(lr=cfg["lr"], weight_decay=cfg["weight_decay"], batch_size=cfg["batch_size"], epochs=cfg["epochs"])
    tr = make_dataset(cfg["n_train"], 100); va = make_dataset(cfg["n_val"], 200); te = make_dataset(cfg["n_test"], 300)
    variants = {
        "momworld": MomWorldConfig(**cfg["model"]),
        "baseline": MomWorldConfig(**{**cfg["model"], "use_momentum": False}),
        "no_moflow": MomWorldConfig(**{**cfg["model"], "use_moflow": False}),
        "no_reset_gate": MomWorldConfig(**{**cfg["model"], "use_reset_gate": False}),
    }
    results = {k: [] for k in variants}
    for seed in cfg["seeds"]:
        for name, mc in variants.items():
            m = fit(mc, tr, va, tcfg, seed, name)
            torch.save({"state": m.state_dict(), "cfg": mc.__dict__}, f"outputs/{name}_seed{seed}.pt")
            r = evaluate(m, te); r["params"] = sum(p.numel() for p in m.parameters())
            results[name].append(r)
            print(f"  TEST {name} s{seed}: ADE {r['ade']:.3f} FDE {r['fde']:.3f} collision {r['collision']:.3f} "
                  f"(event {r['collision_event']:.3f}, free {r['collision_free']:.3f})", flush=True)
    summary = {}
    for name, rs in results.items():
        summary[name] = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs])))
                         for k in rs[0] if isinstance(rs[0][k], float)}
    # non-learned "physics old way": persist the observed trend, ignore the scene
    sp = stale_momentum_extrapolation(te.hist); sa, sf = ade_fde(sp, te.future); sc = collision(sp, te.lead_future)
    stale = dict(ade=sa.mean().item(), fde=sf.mean().item(), collision=sc.float().mean().item(),
                 collision_event=sc[te.event].float().mean().item(), ade_event=sa[te.event].mean().item(), ade_free=sa[~te.event].mean().item())
    print("stale_momentum", {k: round(v, 3) for k, v in stale.items()})
    json.dump({"stale_momentum": stale, "per_seed": results, "summary": summary, "config": cfg}, open("outputs/results.json", "w"), indent=1)
    for name, s in summary.items():
        print(f"{name:14s} ADE {s['ade'][0]:.3f}±{s['ade'][1]:.3f}  FDE {s['fde'][0]:.3f}  collision {s['collision'][0]:.3f}±{s['collision'][1]:.3f}"
              f"  col_event {s['collision_event'][0]:.3f}  ade_far {s['ade_far'][0]:.3f}")

if __name__ == "__main__":
    main()
