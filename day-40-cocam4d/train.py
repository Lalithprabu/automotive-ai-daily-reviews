"""Train MotionUncertaintyNet on synthetic cooperative tracks, then evaluate cooperative fusion."""
import argparse, json, time
import numpy as np, torch, yaml
from src.cocam4d import MotionUncertaintyNet, ModelConfig
from src.cocam4d.model import compute_loss
from src.cocam4d.data import build_training_set
from src.cocam4d.evaluate import run_eval


def lsq_velocity(X, dt):
    """Classic baseline: least-squares slope of the noisy centroid history (no learning)."""
    rel = X[:, :, :3]; T = rel.shape[1]; t = (np.arange(T) - (T - 1)) * dt
    tc = t - t.mean()
    return ((rel - rel.mean(1, keepdims=True)) * tc[None, :, None]).sum(1) / (tc ** 2).sum()


def main(a):
    cfg = yaml.safe_load(open(a.config)); d, tr = cfg["data"], cfg["train"]
    torch.manual_seed(cfg["seed"]); rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    Xtr, Vtr, Dtr, Etr = build_training_set(rng, d["n_train_scenes"], d)
    Xva, Vva, Dva, Eva = build_training_set(rng, d["n_val_scenes"], d)
    print(f"data: {len(Xtr)} train / {len(Xva)} val tracks ({time.time()-t0:.1f}s)")
    model = MotionUncertaintyNet(ModelConfig(**cfg["model"]))
    print("params:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tr["lr"], epochs=tr["epochs"],
                                                steps_per_epoch=int(np.ceil(len(Xtr) / tr["batch_size"])))
    T_ = lambda *z: [torch.from_numpy(x) for x in z]
    Xt, Vt, Dt, Et = T_(Xtr, Vtr, Dtr, Etr)
    for ep in range(tr["epochs"]):
        model.train(); perm = torch.randperm(len(Xt)); tot = 0
        for i in range(0, len(perm), tr["batch_size"]):
            b = perm[i:i + tr["batch_size"]]
            out = model(Xt[b])
            loss, parts = compute_loss(out, {"vel": Vt[b], "dyn": Dt[b], "pos_err": Et[b]},
                                       tr["w_vel"], tr["w_nll"], tr["w_dyn"], tr["w_unc"])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item() * len(b)
        if ep % 5 == 0 or ep == tr["epochs"] - 1:
            print(f"epoch {ep:2d} loss {tot/len(Xt):.4f}  {parts}")
    model.eval()
    with torch.no_grad():
        o = model(torch.from_numpy(Xva))
    vel_rmse = float(((o["vel"].numpy() - Vva) ** 2).mean() ** 0.5)
    lsq = lsq_velocity(Xva, d["dt"]); lsq_rmse = float(((lsq - Vva) ** 2).mean() ** 0.5)
    dyn_acc = float(((o["dyn_logit"].numpy() > 0) == (Dva > 0.5)).mean())
    sig = o["pos_sigma"].numpy() * 1.25; z = np.abs(Eva) / sig
    corr = float(np.corrcoef(sig.ravel(), np.abs(Eva).ravel())[0, 1])
    cov95 = float((z < 1.96).mean())
    summary = {"params": sum(p.numel() for p in model.parameters()),
               "val_vel_rmse_model": vel_rmse, "val_vel_rmse_lsq_baseline": lsq_rmse,
               "val_dyn_acc": dyn_acc, "pos_sigma_vs_abs_err_corr": corr, "pos_sigma_1.96_coverage": cov95}
    print(json.dumps(summary, indent=1))
    torch.save({"model": model.state_dict(), "cfg": cfg}, "checkpoints/cocam4d.pt")
    fus = run_eval(model, d, n_scenes=a.eval_scenes)
    summary["fusion"] = fus
    json.dump(summary, open("results.json", "w"), indent=1)
    for m, r in fus.items():
        print(f"{m:28s}", " | ".join(f"{L}ms err {v['mean_pos_err_m']:.2f}m rec {v['recall_2m']:.2f}" for L, v in r.items()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config.yaml")
    p.add_argument("--eval_scenes", type=int, default=60)
    main(p.parse_args())
