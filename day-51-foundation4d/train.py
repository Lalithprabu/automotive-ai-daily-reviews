"""Trains (a) Foundation4D-Mini jointly on 4 tasks, (b) a single-task detection-only 'old way' model,
(c) a joint model with dense (non-factorized) attention. Same data, same steps, same seed schedule."""
import argparse, json, time, yaml, torch
from src.synthetic import build_dataset
from src.foundation4d import Foundation4DMini, compute_loss, attention_pairs
from src.metrics import abs_rel, det_f1, best_threshold, occ_iou


def batches(d, bs, g):
    n = d["rgb"].shape[0]
    idx = torch.randperm(n, generator=g)
    for i in range(0, n - bs + 1, bs):
        j = idx[i:i + bs]
        yield {k: d[k][j] for k in ("rgb", "depth", "heat", "occ")}


def fit(name, cfg, tr, va, tasks, factorized=True, log=print):
    torch.manual_seed(0)
    m = cfg["model"]
    net = Foundation4DMini(m["dim"], m["heads"], m["depth"], m["patch"], m["n_views"], m["n_frames"],
                           (cfg["data"]["img_h"], cfg["data"]["img_w"]), factorized, tasks)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    E = cfg["train"]["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["train"]["lr"], total_steps=E * (tr["rgb"].shape[0] // cfg["train"]["batch_size"]))
    g = torch.Generator().manual_seed(1)
    t0 = time.time(); hist = []
    for ep in range(E):
        net.train(); tot = 0; nb = 0
        for b in batches(tr, cfg["train"]["batch_size"], g):
            loss, parts = compute_loss(net(b["rgb"]), b)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item(); nb += 1
        hist.append(tot / nb)
        log(f"[{name}] epoch {ep+1}/{E} loss {tot/nb:.4f}  ({time.time()-t0:.0f}s)")
    return net, hist, time.time() - t0


@torch.no_grad()
def evaluate(net, va):
    net.eval()
    out = net(va["rgb"])
    res = {}
    if "depth" in out: res["depth_absrel"] = abs_rel(out["depth"], va["depth"])
    if "occ" in out: res["occ_iou"] = occ_iou(out["occ"], va["occ"])
    if "det" in out:
        thr, (p, r, f1) = best_threshold(out["det"], va["heat"])
        res.update(det_thr=thr, det_precision=p, det_recall=r, det_f1=f1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); d = cfg["data"]
    tr = build_dataset(d["n_train"], d["seed"], d["img_h"], d["img_w"])
    va = build_dataset(d["n_val"], d["seed"] + 1, d["img_h"], d["img_w"])
    results = {}
    runs = {"joint_factorized": (("depth", "det", "occ", "gs"), True),
            "det_only_old_way": (("det",), True),
            "joint_dense_attention": (("depth", "det", "occ", "gs"), False)}
    for name, (tasks, fac) in runs.items():
        net, hist, secs = fit(name, cfg, tr, va, tasks, fac)
        r = evaluate(net, va); r.update(params=sum(p.numel() for p in net.parameters()), train_seconds=round(secs, 1),
                                        loss_first=hist[0], loss_last=hist[-1])
        results[name] = r; print(name, json.dumps(r))
        if name == "joint_factorized":
            torch.save({"state": net.state_dict(), "cfg": cfg, "det_thr": r["det_thr"]}, "checkpoints/joint.pt")
    m = cfg["model"]; N = (d["img_h"] // m["patch"]) * (d["img_w"] // m["patch"])
    results["attention_pairs_per_layer"] = dict(factorized=attention_pairs(m["n_views"], m["n_frames"], N, True),
                                                dense=attention_pairs(m["n_views"], m["n_frames"], N, False))
    json.dump(results, open("results.json", "w"), indent=2)
