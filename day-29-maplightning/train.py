"""Train MapLightning-style mapper and the dense-BEV baseline on identical synthetic data, then sweep
extrinsic-perturbation severity. Usage: python train.py --config config.yaml"""
import argparse, json, sys, time, warnings
from pathlib import Path
import torch, yaml
sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")
from maplightning import MapLightning, BEVBaseline, make_batch
from maplightning.model import count_params
from maplightning.loss import set_loss
from maplightning.metrics import evaluate


def fit(model, cfg, name, out):
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 1)
    opt = torch.optim.AdamW(model.parameters(), cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["steps"], pct_start=0.1)
    t0, hist = time.time(), []
    for step in range(cfg["steps"]):
        model.train()
        b = make_batch(cfg["batch_size"], cfg["train_jitter_deg"], gen)
        logits, pts = model(b["img"])
        loss, parts = set_loss(logits, pts, b)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        hist.append(float(loss.detach()))
        if step % 100 == 0 or step == cfg["steps"] - 1:
            print(f"[{name}] step {step:4d} loss {hist[-1]:.4f} cls {parts['cls']:.3f} pts {parts['pts']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), out / f"{name}.pt")
    return hist


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="outputs"); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); out = Path(a.out); out.mkdir(exist_ok=True)
    models = {"maplightning": MapLightning(cfg["d_model"], cfg["n_map_tokens"]), "bev_baseline": BEVBaseline(cfg["d_model"])}
    res = {"config": cfg, "params": {k: count_params(m) for k, m in models.items()}, "loss_history": {}, "sweep": {}}
    for name, m in models.items():
        res["loss_history"][name] = fit(m, cfg, name, out)
    for s in cfg["eval_jitter_sweep"]:
        res["sweep"][str(s)] = {n: evaluate(m, cfg["eval_samples"], s, seed=123) for n, m in models.items()}
        print(f"jitter {s} deg:", {n: (round(v['err_m'], 3), round(v['f1_1m'], 3)) for n, v in res["sweep"][str(s)].items()}, flush=True)
    json.dump(res, open(out / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
