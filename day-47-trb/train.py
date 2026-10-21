"""Train + evaluate 4 variants on identical synthetic data streams:
stale (no compensation), warp (pose-warp only), trb_nogate, trb (full).  python train.py --config config.yaml"""
import argparse, json, math, os, sys, time
import numpy as np, torch, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from trb.model import AsyncCollabDetector, TRBConfig
from trb.data import make_batch, MAX_DELAY
from trb.losses import det_loss, objectness_loss, app_loss
from trb.metrics import detection_stats, summarize

VARIANTS = {"stale": ("stale", {}), "warp": ("warp", {}), "trb_nogate": ("trb", {"use_gate": False}), "trb": ("trb", {})}


def build(name, mcfg):
    kind, extra = VARIANTS[name]
    return AsyncCollabDetector(kind, TRBConfig(**{**mcfg, **extra}))


def step_loss(model, batch, cfg):
    out = model(batch)
    l_det = det_loss(out["logits"], batch["heat"])
    l_obj = objectness_loss(model.objectness(batch["target"]), batch["heat"]) * cfg["objectness_weight"] if model.variant == "trb" else torch.zeros(())
    l_app = app_loss(out["B"], out["dF"], batch["target"]) * cfg["lambda_app"] if model.variant == "trb" else torch.zeros(())
    return l_det + l_obj + l_app, dict(det=float(l_det), obj=float(l_obj), app=float(l_app))


@torch.no_grad()
def evaluate(model, batches, thr):
    tot = dict(tp=0, fn=0, fp=0, tp_b=0, fn_b=0, err=0.0); mse = []
    for b in batches:
        out = model(b)
        s = detection_stats(out["logits"], b["pos"], b["mask"], thr)
        for k in tot: tot[k] += s[k]
        mse.append(float(((out["F_hat"] - b["target"]) ** 2).mean()))
    r = summarize(tot); r["feat_mse"] = float(np.mean(mse)); return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--steps", type=int); ap.add_argument("--variants", default="stale,warp,trb_nogate,trb")
    ap.add_argument("--out", default="checkpoints"); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); steps = a.steps or cfg["steps"]
    os.makedirs(a.out, exist_ok=True); torch.set_num_threads(2)
    val = [make_batch(20, 900000 + i) for i in range(cfg["val_scenes"] // 20)]
    test = {d: [make_batch(25, 800000 + 10 * d + i, delays=[d] * 25) for i in range(4)] for d in range(MAX_DELAY + 1)}
    results = {}
    for name in a.variants.split(","):
        torch.manual_seed(cfg["seed"]); model = build(name, cfg["model"])
        npar = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], eps=1e-10)
        sched = lambda s: (s + 1) / cfg["warmup_steps"] if s < cfg["warmup_steps"] else 0.5 * (1 + math.cos(math.pi * (s - cfg["warmup_steps"]) / max(1, steps - cfg["warmup_steps"])))
        t0 = time.time(); hist = []
        for s in range(steps):
            for g in opt.param_groups: g["lr"] = cfg["lr"] * max(sched(s), 0.01)
            batch = make_batch(cfg["batch_size"], cfg["seed"] * 1000003 + s)
            loss, parts = step_loss(model, batch, cfg)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            hist.append(float(loss))
            if s % 50 == 0 or s == steps - 1:
                print(f"[{name}] step {s:4d} loss {float(loss):.4f} {parts} ({time.time()-t0:.0f}s)", flush=True)
        model.eval()
        best = max(np.arange(0.3, 0.91, 0.05), key=lambda t: evaluate(model, val, float(t))["f1"]); best = float(best)
        res = {"params": npar, "thr": best, "train_loss_first": hist[0], "train_loss_last": float(np.mean(hist[-20:])),
               "by_delay": {d: evaluate(model, test[d], best) for d in test}}
        allb = [b for d in test for b in test[d]]; res["overall"] = evaluate(model, allb, best)
        results[name] = res
        torch.save({"state": model.state_dict(), "variant": name, "model_cfg": cfg["model"], "thr": best}, f"{a.out}/{name}.pt")
        print(f"[{name}] overall {res['overall']}", flush=True)
        json.dump(results, open("results.json", "w"), indent=1)
    try: make_plot(results)
    except Exception as e: print("plot skipped:", e)


def make_plot(results):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = {"stale": "#9aa0a6", "warp": "#d9822b", "trb_nogate": "#6a9fd8", "trb": "#1a7f4b"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for n, r in results.items():
        ds = sorted(r["by_delay"], key=int)
        ax[0].plot([int(d) * 0.15 for d in ds], [r["by_delay"][d]["blind_recall"] for d in ds], "o-", c=cols[n], label=n)
        ax[1].plot([int(d) * 0.15 for d in ds], [r["by_delay"][d]["feat_mse"] for d in ds], "o-", c=cols[n], label=n)
    ax[0].set_title("Blind-zone recall vs. message delay"); ax[1].set_title("Feature MSE vs. fresh collaborator feature")
    for x in ax: x.set_xlabel("delay (s)"); x.grid(alpha=.3)
    ax[0].legend(); fig.suptitle("Synthetic reconstruction (this repo's data - not the paper's DAIR-V2X/OPV2V numbers)", fontsize=9)
    fig.tight_layout(); fig.savefig("assets/results.png", dpi=130)


if __name__ == "__main__":
    main()
