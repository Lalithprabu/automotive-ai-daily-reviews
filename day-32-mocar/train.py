"""Train MoCAR, MoCAR-without-coordinate-awareness (ablation) and the one-shot baseline; evaluate on held-out test."""
import argparse, json, time, yaml, torch, numpy as np
from torch.utils.data import DataLoader
from mocar.data import SyntheticDrivingDataset, KINDS
from mocar.model import MoCAR, OneShotBaseline


def evaluate(model, ds, kind_names=KINDS, miss_thr=2.0):
    model.eval(); res = {"ade": [], "fde": [], "top1": [], "kind": []}
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=256):
            out = model(b, teacher_forcing=False) if isinstance(model, MoCAR) else model(b)
            gt = b["fut_pts"][:, 1:]
            err = (out["pts"] - gt[:, None]).norm(dim=-1)                 # (B,M,T)
            ade, fde = err.mean(-1), err[..., -1]
            res["ade"].append(ade.min(1).values); res["fde"].append(fde.min(1).values)
            res["top1"].append(ade[torch.arange(len(ade)), out["scores"].argmax(1)]); res["kind"].append(b["kind"])
    r = {k: torch.cat(v) for k, v in res.items()}
    def agg(m):
        return dict(n=int(m.sum()), minADE3=r["ade"][m].mean().item(), minFDE3=r["fde"][m].mean().item(),
                    top1ADE=r["top1"][m].mean().item(), miss=(r["fde"][m] > miss_thr).float().mean().item())
    out = {"all": agg(torch.ones_like(r["kind"], dtype=torch.bool))}
    for i, k in enumerate(kind_names): out[k] = agg(r["kind"] == i)
    out["turns"] = agg((r["kind"] == 1) | (r["kind"] == 2))
    return out


def const_vel(ds):
    gt = ds.d["fut_pts"][:, 1:]; v = (ds.d["hist_pts"][:, -1] - ds.d["hist_pts"][:, -2])
    pred = v[:, None] * torch.arange(1, 31)[None, :, None]
    e = (pred - gt).norm(dim=-1)
    return dict(ADE=e.mean().item(), FDE=e[:, -1].mean().item())


def fit(model, train, val, cfg, name):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    E = cfg["epochs"]; sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=E * ((len(train) + cfg["batch_size"] - 1) // cfg["batch_size"]))
    t0, hist = time.time(), []
    for ep in range(E):
        model.train(); tot = n = 0
        for b in DataLoader(train, batch_size=cfg["batch_size"], shuffle=True):
            loss, st = model.loss(b); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item() * len(b["kind"]); n += len(b["kind"])
        v = evaluate(model, val)["all"]["minADE3"]; hist.append((tot / n, v))
        if ep % 5 == 0 or ep == E - 1: print(f"[{name}] ep {ep:02d} loss {tot/n:.4f} val minADE3 {v:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--epochs", type=int)
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config))
    if a.epochs: cfg["train"]["epochs"] = a.epochs
    torch.manual_seed(cfg["seed"])
    tr, va, te = (SyntheticDrivingDataset(cfg["data"][k], seed=s) for k, s in (("n_train", 1), ("n_val", 2), ("n_test", 3)))
    mc = cfg["model"]; models = {
        "oneshot": OneShotBaseline(mc["d_model"], mc["n_heads"], mc["n_layers"], mc["n_modes"]),
        "mocar": MoCAR(mc["code_dim"], mc["d_model"], mc["n_heads"], mc["n_layers"], mc["n_modes"], coord_aware=True),
        "mocar_nocoord": MoCAR(mc["code_dim"], mc["d_model"], mc["n_heads"], mc["n_layers"], mc["n_modes"], coord_aware=False)}
    results, curves = {"const_velocity": const_vel(te)}, {}
    for name, m in models.items():
        torch.manual_seed(cfg["seed"]); curves[name] = fit(m, tr, va, cfg["train"], name)
        results[name] = evaluate(m, te); results[name]["params"] = sum(p.numel() for p in m.parameters())
        torch.save(m.state_dict(), f"checkpoint_{name}.pt")
    json.dump(dict(results=results, curves=curves, cfg=cfg), open("results.json", "w"), indent=1)
    print(json.dumps({k: (v if k == "const_velocity" else {kk: v[kk] for kk in ("all", "turns", "params")}) for k, v in results.items()}, indent=1))
