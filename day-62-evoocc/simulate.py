"""Render the 'AI in action' streaming-occupancy simulation from the REAL trained checkpoint.

Panels (all in the ego-centred BEV frame, ego = white arrow, forward = right):
  1 Ground truth            2 FAST system (live, coarse/noisy)     3 SLOW system, delayed + ego-warped (stale)
  4 EvoOcc (fast + evolved slow anchor)   5 EvoOcc error map (green hit / red miss / orange false-pos)
  6 Telemetry: rolling mIoU vs GT for fast / warped-slow / EvoOcc + slow-update age

Usage:  python simulate.py --ckpt checkpoints/evoocc.pt --out assets/evoocc_sim --frames 18 [--live]
--live opens a matplotlib window instead of writing files (needs a display).
"""
import argparse, os, numpy as np, torch
import matplotlib
from src.evoocc import EvoOcc
from src.warp import warp_bev
from src import data

COL = np.array([[0.08, 0.09, 0.12], [0.55, 0.60, 0.70], [0.98, 0.78, 0.15]])   # free / static / dynamic


def run(ckpt, seed, frames, sync_every):
    """Stream `frames` steps; slow output for frame s arrives `sync_every` frames later (the 'delay')."""
    m = EvoOcc(hidden=32); m.load_state_dict(torch.load(ckpt)); m.eval()
    b = data.make_batch(1, T=frames, seed=seed)
    recs = []
    with torch.no_grad():
        for start in range(0, frames, sync_every):
            T = min(sync_every, frames - start)
            # slow logits for frame `start` are produced from the (hidden) world, arriving at start+T
            g = torch.Generator(); g.manual_seed(seed * 1000 + start)
            slow = data.slow_logits(b["gt"][:, start], g)
            out = m(slow, b["fast"][:, start:start + T + 1], b["ego"][:, start:start + T], b["dts"][:, start:start + T])
            ws = slow
            for k in range(T + 1):
                if k > 0: ws = warp_bev(ws, b["ego"][:, start + k - 1], data.EXTENT)
                f = start + k
                if f in [r["f"] for r in recs]: continue
                age = float(b["dts"][0, start:start + k].sum())
                recs.append(dict(f=f, gt=b["gt"][0, f].numpy(), fast=b["fast"][0, f].argmax(0).numpy(),
                                 slow=ws[0].argmax(0).numpy(), evo=out[0, k].argmax(0).numpy(), age=age))
    return recs


def iou(p, g):
    return float(np.mean([((p == c) & (g == c)).sum() / max(((p == c) | (g == c)).sum(), 1) for c in (1, 2)]))


def draw(recs, i, hist, fig, axes):
    import matplotlib.pyplot as plt
    r = recs[i]; ext = [-24, 24, 24, -24]
    def show(ax, img, title):
        ax.clear(); ax.imshow(img, extent=ext, interpolation="nearest"); ax.set_title(title, fontsize=10, color="w")
        ax.set_xticks([]); ax.set_yticks([]); ax.annotate("", (3.5, 0), (0, 0), arrowprops=dict(arrowstyle="-|>", color="w", lw=2))
    show(axes[0], COL[r["gt"]], "Ground truth")
    show(axes[1], COL[r["fast"]], f"FAST (live)  mIoU {iou(r['fast'], r['gt']):.2f}")
    show(axes[2], COL[r["slow"]], f"SLOW, {r['age']:.2f}s stale, ego-warped  {iou(r['slow'], r['gt']):.2f}")
    show(axes[3], COL[r["evo"]], f"EvoOcc  mIoU {iou(r['evo'], r['gt']):.2f}")
    err = np.zeros(r["gt"].shape + (3,)); occ_p, occ_g = r["evo"] > 0, r["gt"] > 0
    err[occ_p & occ_g & (r["evo"] == r["gt"])] = [0.2, 0.85, 0.35]; err[occ_g & ~(r["evo"] == r["gt"])] = [0.9, 0.2, 0.2]
    err[occ_p & ~occ_g] = [1.0, 0.55, 0.1]; err[~occ_p & ~occ_g] = [0.08, 0.09, 0.12]
    show(axes[4], err, "EvoOcc errors: green hit / red miss / orange FP")
    ax = axes[5]; ax.clear(); ax.set_facecolor("#14161c")
    for key, c, lab in (("fast", "#e0524a", "fast"), ("slow", "#8aa0ff", "warped slow"), ("evo", "#35d96a", "EvoOcc")):
        ax.plot(hist[key][:i + 1], color=c, lw=2, label=lab)
    ax.set_xlim(0, len(recs) - 1); ax.set_ylim(0, 1); ax.legend(loc="lower left", fontsize=8, facecolor="#14161c", labelcolor="w")
    ax.set_title(f"mIoU vs GT - frame {r['f']}  slow age {r['age']:.2f}s", fontsize=10, color="w")
    ax.tick_params(colors="w", labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/evoocc.pt"); ap.add_argument("--out", default="assets/evoocc_sim")
    ap.add_argument("--frames", type=int, default=18); ap.add_argument("--sync-every", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    if not a.live: matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = run(a.ckpt, a.seed, a.frames, a.sync_every)
    hist = {k: [iou(r[k], r["gt"]) for r in recs] for k in ("fast", "slow", "evo")}
    fig, axs = plt.subplots(2, 3, figsize=(12, 8.2), facecolor="#0b0c10"); axes = axs.ravel()
    fig.suptitle("EvoOcc streaming occupancy - synthetic world, real trained model", color="w", fontsize=13)
    if a.live:
        plt.ion()
        for i in range(len(recs)): draw(recs, i, hist, fig, axes); plt.pause(0.4)
        plt.show(block=True); return
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); imgs = []
    for i in range(len(recs)):
        draw(recs, i, hist, fig, axes); fig.tight_layout(rect=(0, 0, 1, .95)); fig.canvas.draw()
        imgs.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    imageio.mimsave(a.out + ".gif", imgs, duration=0.5, loop=0)
    imageio.mimsave(a.out + ".mp4", imgs, fps=2, macro_block_size=1)
    plt.imsave(a.out + "_still.png", imgs[min(8, len(imgs) - 1)])
    print("wrote", a.out + ".gif/.mp4", "| mean mIoU fast %.3f slow %.3f evo %.3f" % tuple(np.mean(hist[k]) for k in ("fast", "slow", "evo")))


if __name__ == "__main__":
    main()
