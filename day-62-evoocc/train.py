"""Train EvoOcc (and the time-only ablation) on the synthetic streaming-occupancy task, then evaluate.
Usage: python train.py --steps 1500 --out checkpoints
"""
import argparse, json, os, time, torch
import torch.nn.functional as F
from src.evoocc import EvoOcc
from src.warp import warp_bev
from src import data

T_MAX = 6


def train(use_control, steps, bs, seed, log):
    torch.manual_seed(seed)
    m = EvoOcc(hidden=32, use_control=use_control)
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    t0 = time.time()
    for it in range(steps):
        T = int(torch.randint(2, T_MAX + 1, (1,)))                    # random delay window
        b = data.make_batch(bs, T=T)
        out = m(b["slow0"], b["fast"][:, :T + 1], b["ego"][:, :T], b["dts"][:, :T])
        loss = F.cross_entropy(out.flatten(0, 1), b["gt"][:, :T + 1].flatten(0, 1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if it % 100 == 0 or it == steps - 1:
            log(f"[{'control' if use_control else 'time-only'}] step {it:5d} loss {loss.item():.4f} ({time.time()-t0:.0f}s)")
    return m


@torch.no_grad()
def evaluate(m, n=300, seed=1234):
    """Per-delay-step metrics on held-out scenes. Returns dict of lists indexed by step 0..T_MAX."""
    m.eval(); res = {k: [[] for _ in range(T_MAX + 1)] for k in ("fast", "warped_slow", "evoocc", "evoocc_dyn", "fast_dyn", "warped_dyn")}
    for bi in range(n // 50):
        b = data.make_batch(50, T=T_MAX, seed=seed + bi)
        out = m(b["slow0"], b["fast"], b["ego"], b["dts"]).argmax(2)
        ws = b["slow0"]
        for k in range(T_MAX + 1):
            if k > 0: ws = warp_bev(ws, b["ego"][:, k - 1], data.EXTENT)
            gt = b["gt"][:, k]
            res["fast"][k].append(data.miou(b["fast"][:, k].argmax(1), gt).item())
            res["warped_slow"][k].append(data.miou(ws.argmax(1), gt).item())
            res["evoocc"][k].append(data.miou(out[:, k], gt).item())
            res["fast_dyn"][k].append(data.class_iou(b["fast"][:, k].argmax(1), gt, 2))
            res["warped_dyn"][k].append(data.class_iou(ws.argmax(1), gt, 2))
            res["evoocc_dyn"][k].append(data.class_iou(out[:, k], gt, 2))
    return {k: [sum(v) / len(v) for v in vs] for k, vs in res.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=12); ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); lines = []
    def log(s): print(s, flush=True); lines.append(s)
    results = {}
    for name, uc in (("evoocc", True), ("time_only", False)):
        m = train(uc, a.steps, a.bs, a.seed, log)
        torch.save(m.state_dict(), f"{a.out}/{name}.pt")
        results[name] = evaluate(m)
        results["n_params"] = sum(p.numel() for p in m.parameters())
    json.dump(results, open(f"{a.out}/results.json", "w"), indent=1)
    log(json.dumps({k: v for k, v in results.items() if k == "n_params"}))
    open(f"{a.out}/train_log.txt", "w").write("\n".join(lines))
