"""Run Mean-FLIP vs baselines (opacity, supervised Taylor, random) on the trained dense bank."""
import json, time, yaml, copy, torch, torch.nn.functional as F
from decision_flip.model import GaussianOccModel
from decision_flip.data import make_split
from decision_flip.metrics import evaluate
from decision_flip.flip import mean_flip_scores, guarded_otsu, compile_pruned, topk_mask


def taylor_scores(model, cf, co, ncls):
    """Supervised first-order Taylor: |g * dL/dg| per row with a unit gate g, summed over calibration batches."""
    model.eval(); g = torch.ones(model.bank.n, requires_grad=True); tot = torch.zeros(model.bank.n)
    for i in range(0, len(cf), 8):
        loss = F.cross_entropy(model(cf[i:i + 8], g).reshape(-1, ncls), co[i:i + 8].reshape(-1))
        gr, = torch.autograd.grad(loss, g); tot += (gr * g).abs().detach()
    return tot


def finetune(model, tf, to, steps, lr, ncls):
    opt = torch.optim.Adam(model.parameters(), lr=lr); w = torch.tensor([0.5, 1.0, 3.0, 1.5, 2.0])
    for _ in range(steps):
        model.train(); idx = torch.randint(0, len(tf), (8,))
        loss = F.cross_entropy(model(tf[idx]).reshape(-1, ncls), to[idx].reshape(-1), weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def latency_ms(model, feat, reps=30):
    model.eval()
    with torch.no_grad():
        for _ in range(3): model(feat)
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); model(feat); ts.append((time.perf_counter() - t) * 1e3)
    return sorted(ts)[len(ts) // 2]


def main():
    ck = torch.load("checkpoints/dense.pt"); cfg = ck["cfg"]; torch.manual_seed(7)
    grid = (cfg["grid"]["x"], cfg["grid"]["y"], cfg["grid"]["z"]); C = cfg["num_classes"]
    model = GaussianOccModel(cfg); model.load_state_dict(ck["state"]); model.eval()
    tf, to = make_split(0, cfg["data"]["train_scenes"], grid)
    cf, co = make_split(5000, cfg["data"]["calib_scenes"], grid)       # calibration (labels used ONLY by Taylor baseline)
    vf, vo = make_split(10000, cfg["data"]["val_scenes"], grid)
    N = model.bank.n; base = evaluate(model, vf, vo)
    flip = mean_flip_scores(model, cf); keep, info = guarded_otsu(flip, cfg["prune"]["otsu_min_sep"]); K = int(keep.sum())
    opa = torch.sigmoid(model.bank.opa.detach()); tay = taylor_scores(model, cf, co, C)
    rnd = torch.rand(N, generator=torch.Generator().manual_seed(3))
    out = {"dense_rows": N, "dense_miou": base, "flip_info": info, "kept_rows": K, "removed_pct": 100 * (1 - K / N),
           "dead_rows_zero_flip": int((flip == 0).sum()), "scores": flip.tolist(), "methods": {}, "sweep": {}}
    methods = {"mean_flip": flip, "opacity": opa, "taylor_supervised": tay, "random": rnd}
    for name, s in methods.items():
        mk = keep if name == "mean_flip" else topk_mask(s, K)
        pm = compile_pruned(model, mk); no_rec = evaluate(pm, vf, vo)
        rec = evaluate(finetune(copy.deepcopy(pm), tf, to, cfg["prune"]["recovery_steps"], cfg["prune"]["recovery_lr"], C), vf, vo)
        out["methods"][name] = {"miou_no_recovery": no_rec, "miou_recovered": rec, "rows": int(mk.sum())}
        print(f"{name:18s} rows {int(mk.sum()):3d}  mIoU no-recovery {no_rec:.4f}  recovered {rec:.4f}", flush=True)
    # keep-ratio sweep (no recovery) for the simulation/plots
    for name, s in methods.items():
        out["sweep"][name] = []
        for frac in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            k = max(int(N * frac), 1); pm = compile_pruned(model, topk_mask(s, k))
            out["sweep"][name].append([frac, k, evaluate(pm, vf, vo)])
    fb = vf[:8]; pm = compile_pruned(model, keep)
    out["latency_ms"] = {"dense": latency_ms(model, fb), "pruned": latency_ms(pm, fb)}
    out["latency_ms"]["reduction_pct"] = 100 * (1 - out["latency_ms"]["pruned"] / out["latency_ms"]["dense"])
    out["params"] = {"dense": sum(p.numel() for p in model.parameters()), "pruned": sum(p.numel() for p in pm.parameters())}
    torch.save({"cfg": cfg, "state": pm.state_dict(), "keep": keep, "flip": flip, "bank_n": int(keep.sum())}, "checkpoints/pruned.pt")
    json.dump(out, open("checkpoints/prune_results.json", "w"), indent=1)
    print("latency", out["latency_ms"], "params", out["params"], "flip info", info)

if __name__ == "__main__":
    main()
