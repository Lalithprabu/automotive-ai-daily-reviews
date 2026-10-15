"""Train: deterministic baseline, MC-dropout model, heterogeneous 5-member deep ensemble; then
conformal-calibrate on a held-out calibration split and report accuracy/coverage. Saves checkpoints + metrics.json."""
import sys, json, time, yaml, torch, numpy as np
sys.path.insert(0, "src")
from xtrack_uq.data import make_dataset, to_tensors
from xtrack_uq.model import XTrackUQ, loss_fn
from evaluate import evaluate

def fit(model, tr, va, cfg, tag):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(cfg["epochs"] * .5), int(cfg["epochs"] * .8)], 0.3)
    n = tr["hist"].shape[0]; bs = cfg["batch_size"]; best = (1e9, None)
    for ep in range(cfg["epochs"]):
        model.train(); perm = torch.randperm(n); tot = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; b = {k: v[idx] for k, v in tr.items()}
            out = model(b, cfg["n_samples"]); loss, _ = loss_fn(out, b, cfg["alpha_dyn"])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item() * len(idx)
        sched.step(); model.eval()
        with torch.no_grad(): vl = loss_fn(model(va, cfg["n_samples"]), va, cfg["alpha_dyn"])[0].item()
        if vl < best[0]: best = (vl, {k: v.clone() for k, v in model.state_dict().items()})
        if ep % 4 == 0 or ep == cfg["epochs"] - 1: print(f"[{tag}] ep {ep:2d} train {tot/n:.3f} val {vl:.3f}", flush=True)
    model.load_state_dict(best[1]); return model

def ade_fde(mean, pos):
    e = (mean - pos).norm(dim=-1); return e.mean().item(), e[:, -1].mean().item()

def main(cfg_path="config.yaml"):
    cfg = yaml.safe_load(open(cfg_path)); torch.manual_seed(cfg["seed"]); t0 = time.time()
    D = {k: to_tensors(make_dataset(cfg["n_" + k], cfg, cfg["seed"] + i)) for i, k in enumerate(["train", "val", "cal", "test"])}
    H = cfg["hidden"]; kw = dict(t_pred=cfg["t_pred"], dt=cfg["dt"])
    base = fit(XTrackUQ(H, **kw), D["train"], D["val"], cfg, "baseline")
    mcd = fit(XTrackUQ(H, p_drop=0.2, **kw), D["train"], D["val"], cfg, "mcdropout")
    # heterogeneous ensemble: different seeds + capacity (wider, more heads, deeper decoder)
    variants = [dict(hidden=H, heads=2, dec_layers=1), dict(hidden=H + 16, heads=2, dec_layers=1), dict(hidden=H, heads=4, dec_layers=1),
                dict(hidden=H, heads=2, dec_layers=2), dict(hidden=H + 16, heads=4, dec_layers=2)][:cfg["ensemble_k"]]
    ens = []
    for k, v in enumerate(variants):
        torch.manual_seed(cfg["seed"] + 100 + k)
        ens.append(fit(XTrackUQ(v["hidden"], heads=v["heads"], dec_layers=v["dec_layers"], **kw), D["train"], D["val"], cfg, f"ens{k}"))
    torch.save(dict(cfg=cfg, base=base.state_dict(), mcd=mcd.state_dict(), variants=variants,
                    ens=[m.state_dict() for m in ens]), "checkpoint.pt")
    res = evaluate(cfg, D, base, mcd, ens)
    res["params_single_model"] = sum(p.numel() for p in base.parameters()); res["train_seconds"] = time.time() - t0
    json.dump(res, open("metrics.json", "w"), indent=2); print("done", round(time.time() - t0), "s")
if __name__ == "__main__": main(*sys.argv[1:])
