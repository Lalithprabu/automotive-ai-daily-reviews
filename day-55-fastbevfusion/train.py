"""Train + evaluate the fusion ablation on the synthetic world.  python train.py [--n_train 3000 --seed 0]
Stages (mirrors the paper's 3-stage recipe, my toy scale):  1 LiDAR branch alone -> 2 image branch + fusion (LiDAR encoder frozen)
-> 3 fusion module only, with Gaussian noise on the camera extrinsics (yaw)."""
import argparse, json, math, time, copy
import numpy as np, torch
from fastbev.synthetic import make_dataset
from fastbev.models import FastBEVFusion
from fastbev.metrics import evaluate

YAW_STD = math.radians(2.5)


def focal(logits, tgt):
    """CenterNet penalty-reduced focal loss on the Gaussian centre heatmap."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4); pos = (tgt > 0.99).float()
    l_pos = -((1 - p) ** 2) * torch.log(p) * pos
    l_neg = -((1 - tgt) ** 4) * (p ** 2) * torch.log(1 - p) * (1 - pos)
    return (l_pos.sum() + l_neg.sum()) / pos.sum().clamp(min=1)


def fit(model, data, epochs, lr=2e-3, params=None, yaw_std=0.0, bs=32, seed=0):
    L, I, H = data; g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(params if params is not None else model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(L) / bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(L), generator=g)
        for s in range(0, len(L), bs):
            b = perm[s:s + bs]
            yaw = torch.randn(len(b), generator=g) * yaw_std if yaw_std else None
            loss = focal(model(L[b], I[b], yaw), H[b]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return model


def build(fusion, lidar_init=None):
    m = FastBEVFusion(fusion)
    if lidar_init is not None:
        m.lidar_enc.load_state_dict(lidar_init.lidar_enc.state_dict()); m.head.load_state_dict(lidar_init.head.state_dict())
    return m


def main(a):
    torch.manual_seed(a.seed); torch.set_num_threads(2)
    L, I, H, _ = make_dataset(a.n_train, a.seed + 10); Lt, It, Ht, Ot = make_dataset(a.n_test, a.seed + 99)
    data = (L, I, H); rng = np.random.default_rng(5)
    yaw_bad = torch.tensor(rng.choice([-1, 1], a.n_test) * math.radians(2.5), dtype=torch.float32)   # distorted calibration
    t0 = time.time(); models, res = {}, {}
    print("stage 1: LiDAR branch alone"); lid = fit(FastBEVFusion("lidar_only"), data, a.e1); models["lidar_only"] = lid
    print("camera-only (from scratch)"); models["camera_only"] = fit(FastBEVFusion("camera_only"), data, a.e1 + a.e2)
    for f in ("add", "concat", "xattn"):
        print(f"stage 2: {f}"); m = build(f, lid)
        for p in m.lidar_enc.parameters(): p.requires_grad = False
        models[f] = fit(m, data, a.e2, lr=1e-3 if f == "xattn" else 2e-3, params=[p for p in m.parameters() if p.requires_grad])
    models["xattn_no_stage3"] = copy.deepcopy(models["xattn"])
    print("stage 3: fusion-only fine-tune with extrinsic noise"); m = models["xattn"]
    fit(m, data, a.e3, lr=7e-4, params=list(m.fuse.parameters()), yaw_std=YAW_STD)
    for k, m in models.items():
        res[k] = dict(params=sum(p.numel() for p in m.parameters()), clean=evaluate(m, Lt, It, Ot), distorted=evaluate(m, Lt, It, Ot, yaw=yaw_bad))
        torch.save(m.state_dict(), f"checkpoints/{k}.pt")
    # latency of the fusion block alone (CPU, batch 1, 200 runs)
    import fastbev.models as M
    x = torch.randn(1, 32, 32, 32); fu = models["xattn"].fuse.eval()
    with torch.no_grad():
        for _ in range(20): fu(x, x)
        t = time.perf_counter(); [fu(x, x) for _ in range(200)]; res["fusion_block_ms_cpu"] = (time.perf_counter() - t) / 200 * 1e3
    json.dump(res, open("checkpoints/results.json", "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s")
    print(f"{'model':18s}{'params':>8s}  AP(clean) F1  nearR farR | AP(dist) F1")
    for k, r in res.items():
        if k.startswith("fusion"): print(k, round(r, 2)); continue
        c, d = r["clean"], r["distorted"]
        print(f"{k:18s}{r['params']:8d}  {c['AP']:.3f}  {c['F1']:.3f} {c['near_R']:.3f} {c['far_R']:.3f} | {d['AP']:.3f} {d['F1']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=3000); ap.add_argument("--n_test", type=int, default=800)
    ap.add_argument("--e1", type=int, default=12); ap.add_argument("--e2", type=int, default=20); ap.add_argument("--e3", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
