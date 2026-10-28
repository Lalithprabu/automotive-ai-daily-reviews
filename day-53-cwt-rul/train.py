"""Train CWT-RUL-Mini on synthetic cells. python train.py [--epochs N]"""
import argparse, json, time, yaml, numpy as np, torch
from src.synthetic import make_cells, split_idx, pick_cycles
from src.model import CWTRULNet
from src.metrics import life_metrics


def batch_for(X, idx, n_window, max_in):
    cyc = pick_cycles(n_window, max_in)
    return torch.from_numpy(X[idx][:, cyc])                                    # [B,N,3,T]


@torch.no_grad()
def evaluate(model, X, y, idx, n_window, max_in, mu, sd):
    model.eval()
    p = np.concatenate([model(batch_for(X, idx[i:i + 256], n_window, max_in)).numpy() for i in range(0, len(idx), 256)])
    return life_metrics(np.exp(p * sd + mu), y[idx])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); d, c, m, t = cfg["data"], cfg["cwt"], cfg["model"], cfg["train"]
    torch.manual_seed(d["seed"]); rng = np.random.default_rng(d["seed"])
    X, life, _ = make_cells(d["n_cells"], d["n_cycles"], d["T"], d["seed"])
    tr, va, te = split_idx(len(X), d["val_frac"], d["test_frac"], d["seed"])
    ll = np.log(life); mu, sd = float(ll[tr].mean()), float(ll[tr].std())          # train-only target stats
    model = CWTRULNet(d["T"], 3, c["n_scales"], c["compress"], m["width"], m["emb"], m["dropout"], c["w0"], c["s_min"], c["s_max"])
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["wd"])
    E = a.epochs or t["epochs"]; sched = torch.optim.lr_scheduler.OneCycleLR(opt, t["lr"], total_steps=E * ((len(tr) + t["batch"] - 1) // t["batch"]))
    print(f"params={sum(p.numel() for p in model.parameters()):,}"); best, t0 = 1e9, time.time()
    for ep in range(E):
        model.train(); perm = rng.permutation(tr); tot = 0
        for i in range(0, len(perm), t["batch"]):
            idx = perm[i:i + t["batch"]]; nw = int(rng.choice(t["n_choices"]))     # random observation window per batch
            x = batch_for(X, idx, nw, t["max_in_cycles"] if "max_in_cycles" in t else t["max_cycles_in"])
            y = torch.from_numpy(((ll[idx] - mu) / sd).astype(np.float32))
            loss = torch.nn.functional.smooth_l1_loss(model(x), y, beta=0.5)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(idx)
        v = evaluate(model, X, life, va, 25, t["max_cycles_in"], mu, sd)
        if v["mape"] < best: best = v["mape"]; torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "cfg": cfg}, "checkpoint.pt")
        print(f"ep {ep+1:02d} loss {tot/len(tr):.4f} val@25 mape {v['mape']:.1f}%")
    ck = torch.load("checkpoint.pt"); model.load_state_dict(ck["model"])
    base = np.exp(mu)                                                              # mean-life baseline
    out = {"baseline_mape": life_metrics(np.full(len(te), base), life[te])["mape"], "by_cycles": {}}
    for n in t["n_choices"]:
        out["by_cycles"][n] = evaluate(model, X, life, te, n, t["max_cycles_in"], mu, sd)
        print(f"test n={n:>2}: {out['by_cycles'][n]}")
    print("mean-baseline MAPE", out["baseline_mape"]); json.dump(out, open("results.json", "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
