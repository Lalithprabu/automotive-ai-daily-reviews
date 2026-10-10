"""Train LighTROcc and the dense-BEV baseline on the synthetic scene set, then evaluate on held-out scenes.
Usage: python train.py --config config.yaml [--steps N]
"""
import argparse, json, os, sys, time
import numpy as np, torch, yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from lightrocc import LighTROcc, DenseBEVOccForecaster
from lightrocc.data import SceneSet, collate
from lightrocc.losses import lighTROcc_loss, occ_loss
from lightrocc.metrics import voxel_iou, instance_metrics, dense_instances, lighTROcc_instances
from torch.utils.data import DataLoader


def train_model(model, name, train_set, val_set, cfg, steps):
    t = cfg["train"]
    torch.manual_seed(t["seed"])
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t["lr"], total_steps=steps, pct_start=0.1)
    dl = DataLoader(train_set, batch_size=t["batch"], shuffle=True, collate_fn=collate, drop_last=True)
    it, step, log, t0 = iter(dl), 0, [], time.time()
    model.train()
    while step < steps:
        try: batch = next(it)
        except StopIteration: it = iter(dl); batch = next(it)
        if isinstance(model, LighTROcc):
            out = model(batch["img"], return_instances=True)
            loss, parts = lighTROcc_loss(out, batch, cfg)
        else:
            out = model(batch["img"])
            loss, parts = occ_loss(out["probs"], batch["occ"], t["w_occ_bce"], t["w_dice"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        log.append(loss.item())
        if step % 100 == 0 or step == 1:
            print(f"[{name}] step {step}/{steps} loss {np.mean(log[-100:]):.4f} {time.time()-t0:.0f}s", flush=True)
    return log


@torch.no_grad()
def evaluate(model, ds, cfg, kind):
    model.eval()
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)
    ious, pred_sets_all, cat = [], [], {"cen": [], "valid": []}
    for batch in dl:
        out = model(batch["img"], return_instances=(kind == "light"))
        ious.append(voxel_iou(out["probs"], batch["occ"]) * len(batch["img"]))
        if kind == "light":
            sets = lighTROcc_instances(out)
        else:
            sets = [[dense_instances(out["probs"][b, t], cfg) for t in range(out["probs"].shape[1])]
                    for b in range(out["probs"].shape[0])]
        pred_sets_all.append((sets, batch))
    iou = (sum(ious) / len(ds)).numpy()
    f1s, ades, ws = [], [], []
    for sets, batch in pred_sets_all:
        f1, ade = instance_metrics(sets, batch, cfg)
        f1s.append(f1 * len(sets)); ades.append(ade * len(sets)); ws.append(len(sets))
    n = sum(ws)
    return dict(iou=iou.tolist(), f1=(sum(f1s) / n).tolist(), ade=(sum(ades) / n).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml"); ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--dense-long", type=int, default=None,
                    help="fairness check: retrain ONLY the dense baseline for N steps; merge into results.json as 'dense_long'")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); steps = a.steps or cfg["train"]["steps"]
    d = cfg["data"]
    print("generating synthetic scenes ...", flush=True)
    tr, va, te = SceneSet(d["n_train"], d["seed"] * 1000, cfg), SceneSet(d["n_val"], 900000, cfg), SceneSet(d["n_test"], 950000, cfg)
    os.makedirs(a.out, exist_ok=True)
    if a.dense_long:
        res = json.load(open(f"{a.out}/results.json"))
        torch.manual_seed(cfg["train"]["seed"]); model = DenseBEVOccForecaster(cfg)
        log = train_model(model, "dense_long", tr, va, cfg, a.dense_long)
        torch.save(model.state_dict(), f"{a.out}/dense_long.pt")
        res["dense_long"] = evaluate(model, te, cfg, "dense"); res["dense_long_steps"] = a.dense_long
        res["dense_long_loss"] = [float(np.mean(log[:50])), float(np.mean(log[-50:]))]
        json.dump(res, open(f"{a.out}/results.json", "w"), indent=2); print("dense_long", json.dumps(res["dense_long"])); return
    res = {"steps": steps}
    for name, M, kind in (("lightrocc", LighTROcc, "light"), ("dense", DenseBEVOccForecaster, "dense")):
        torch.manual_seed(cfg["train"]["seed"]); model = M(cfg)
        res[f"{name}_params"] = sum(p.numel() for p in model.parameters())
        log = train_model(model, name, tr, va, cfg, steps)
        torch.save(model.state_dict(), f"{a.out}/{name}.pt")
        res[f"{name}_loss"] = [float(np.mean(log[:50])), float(np.mean(log[-50:]))]
        res[name] = evaluate(model, te, cfg, kind)
        if kind == "light":
            # inference-time ablation: freeze instances in place (no displacement) = "persistence"
            model.eval()
            dl = DataLoader(te, batch_size=32, collate_fn=collate); ious = []
            with torch.no_grad():
                for b in dl: ious.append(voxel_iou(model(b["img"], zero_disp=True)["probs"], b["occ"]) * len(b["img"]))
            res["lightrocc_zero_disp_iou"] = (sum(ious) / len(te)).tolist()
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=2)
    print("done ->", f"{a.out}/results.json")


if __name__ == "__main__":
    main()
