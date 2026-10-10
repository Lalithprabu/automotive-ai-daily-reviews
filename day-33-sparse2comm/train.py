"""Train ego-only / old-way / Sparse2comm jointly-configured detectors on synthetic cooperative scenes."""
import argparse, json, time, yaml, torch
from sparse2comm import ModelCfg, DataCfg, Sparse2comm, sample_scene, make_batch
from sparse2comm.losses import total_loss
from sparse2comm.metrics import evaluate

VARIANTS = {
    "ego_only": dict(cooperate=False),
    "old_way": dict(use_recon=False, use_flow=False, use_calib=False),   # zero-filled sparse msg, plain gated fusion
    "sparse2comm": dict(),
}


def train_variant(name, cfg, steps=None):
    t = cfg["train"]; steps = steps or t["steps"]
    torch.manual_seed(t["seed"]); g = torch.Generator().manual_seed(t["seed"] + 1)
    dcfg = DataCfg(**cfg["data"])
    model = Sparse2comm(ModelCfg(C=dcfg.C, hidden=cfg["model"]["hidden"], **VARIANTS[name]))
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t["lr"], total_steps=steps, pct_start=0.1)
    t0 = time.time(); hist = []
    for step in range(steps):
        model.train()
        b = make_batch(sample_scene(t["batch_size"], dcfg, g), dcfg, g)
        out = model(b); loss, parts = total_loss(out, b, t)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
        if step % 100 == 0 or step == steps - 1:
            hist.append((step, float(loss.detach()), {k: float(v.detach()) for k, v in parts.items()}))
            print(f"[{name}] step {step:5d} loss {hist[-1][1]:.4f} " + " ".join(f"{k}={v:.3f}" for k, v in hist[-1][2].items()), flush=True)
    print(f"[{name}] done in {time.time()-t0:.0f}s, params={sum(p.numel() for p in model.parameters()):,}")
    return model, dcfg, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS)); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); results = {}
    for name in a.variants:
        model, dcfg, hist = train_variant(name, cfg, a.steps)
        torch.save({"state": model.state_dict(), "cfg": model.cfg.__dict__, "data": dcfg.__dict__}, f"checkpoints/{name}.pt")
        results[name] = evaluate(model, dcfg, cfg["eval"]["n_scenes"], radii=cfg["eval"]["match_radii"])
        results[name]["train_hist"] = hist
        print(name, {k: v for k, v in results[name].items() if k != "train_hist"})
    json.dump(results, open("checkpoints/results.json", "w"), indent=1)
