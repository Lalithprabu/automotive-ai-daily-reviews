"""Synthetic 'AI in action' simulation: camera strip + LiDAR BEV in, ground truth vs detections out.

python simulate.py --out assets/fastbev_demo      (needs checkpoints/ from train.py)
Panels: camera strip (INPUT, no depth) | BEV scene: LiDAR cells (INPUT), GT boxes (green), LiDAR-only (orange x),
FastBEVFusion (blue diamonds), cyan = cross-attention of the farthest object's LiDAR token | running F1 + telemetry.
Frames 20-31 inject a camera-extrinsics error (3 deg yaw): the 'add' fusion baseline degrades, cross-attention + stage-3 holds.
"""
import argparse, json, math
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Rectangle
from fastbev.synthetic import G, CELL, W_IMG, FOV, render, sample_objects
from fastbev.models import FastBEVFusion
from fastbev.metrics import decode, match

COL = dict(gt="#1a9850", lidar="#f28e2b", add="#8e44ad", fuse="#1f5fd1")
NF, BAD = 36, range(20, 32)


def load(k):
    m = FastBEVFusion("xattn" if k.startswith("xattn") else k); m.load_state_dict(torch.load(f"checkpoints/{k}.pt")); return m.eval()


def main(a):
    models = {k: load(k) for k in ("lidar_only", "add", "xattn")}
    rng = np.random.default_rng(a.seed)
    objs = sample_objects(rng, 8); vel = np.stack([rng.uniform(-.5, .1, 8), rng.uniform(-.15, .15, 8)], 1)   # m/frame, ego drives forward
    frames, stats = [], {k: np.zeros(3) for k in models}; hist = {k: [] for k in models}
    for t in range(NF):
        o = objs.copy(); o[:, :2] += vel * t
        o = o[(o[:, 0] > 3) & (o[:, 0] < 46) & (np.abs(o[:, 1]) < 22)]
        L, I = render(o, np.random.default_rng(1000 + t))
        yaw = math.radians(3.0) * (1 if t % 2 else -1) if t in BAD else 0.0
        Lt, It, yt = torch.tensor(L)[None], torch.tensor(I)[None], torch.tensor([yaw], dtype=torch.float32)
        dets, lat = {}, {}
        with torch.no_grad():
            for k, m in models.items():
                lg = m(Lt, It, yt, keep_attn=(k == "xattn")); dets[k] = decode(lg, 0.3)[0]
                r, miss = match(dets[k], o); stats[k] += [sum(x for _, x in r), sum(not x for _, x in r), len(miss)]
                tp, fp, fn = stats[k]; hist[k].append(2 * tp / max(2 * tp + fp + fn, 1))
        far = int(np.argmax(o[:, 0])); qi, qj = int(o[far, 0] / CELL) // 2, int((o[far, 1] / CELL + G / 2)) // 2
        att = models["xattn"].fuse.last_attn[0, qi * (G // 2) + min(qj, G // 2 - 1)].reshape(G // 2, G // 2).numpy()
        frames.append(dict(o=o, L=L, I=I, dets=dets, att=att, far=far, yaw=yaw, t=t))
    fig = plt.figure(figsize=(12.4, 6.2), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[.2, 1], hspace=.28, wspace=.14, left=.05, right=.98, top=.83, bottom=.08)
    axc, axb, axf = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    head = fig.text(.05, .94, "AI in action: LiDAR + camera fusion detects vehicles in BEV (no depth estimation)", fontsize=12, weight="bold")
    sub = fig.text(.05, .885, "", fontsize=9, family="monospace")
    fig.text(.98, .002, "Synthetic demo - FastBEVFusion-style re-implementation (arXiv 2610.37153), not the paper's data", ha="right", fontsize=8, color="#555")

    def draw(f):
        d = frames[min(f, NF - 1)]; t = d["t"]
        axc.cla(); img = np.clip(d["I"].T[None] * .9, 0, 1) * np.ones((10, 1, 1)); axc.imshow(img, aspect="auto", extent=[-70, 70, 0, 1])
        for ofw, ol, _ in d["o"]: axc.plot(math.degrees(math.atan2(-ol, ofw)), 1.12, "v", color=COL["gt"], ms=6, clip_on=False)
        axc.set_yticks([]); axc.set_xlabel("azimuth (deg)", fontsize=7, labelpad=1); axc.tick_params(labelsize=7)
        axc.set_title("INPUT camera strip (green v = true vehicles; posters are camera-only decoys)", fontsize=8.5)
        axb.cla(); X = (np.arange(G) + .5 - G / 2) * CELL
        axb.imshow(np.kron(d["att"], np.ones((2, 2))), origin="lower", extent=[-G * CELL / 2, G * CELL / 2, 0, G * CELL], cmap="Blues", alpha=.55, vmin=0, vmax=max(d["att"].max(), 1e-6))
        ii, jj = np.nonzero(d["L"][0] > 0)
        tall = d["L"][1][ii, jj] > .7
        axb.scatter(X[jj][~tall], (ii[~tall] + .5) * CELL, s=7, c="#bbbbbb", marker="s", label="LiDAR cell (low)")
        axb.scatter(X[jj][tall], (ii[tall] + .5) * CELL, s=14, c="#222222", marker="s", label="LiDAR cell (tall)")
        for ofw, ol, _ in d["o"]: axb.add_patch(Rectangle((-ol - 1.1 if False else ol - 1.1, ofw - 2.3), 2.2, 4.6, fill=False, ec=COL["gt"], lw=2.4))
        for k, mk, ms in (("lidar", "x", 8), ("fuse", "D", 6)):
            kk = "lidar_only" if k == "lidar" else "xattn"; p = d["dets"][kk]
            if p: axb.scatter([q[1] for q in p], [q[0] for q in p], marker=mk, s=ms ** 2, c=COL[k], edgecolors="w" if k == "fuse" else None, linewidths=.8, zorder=6, label={"lidar": "LiDAR-only", "fuse": "FastBEVFusion"}[k])
        axb.plot([], [], color=COL["gt"], lw=2.4, label="GROUND TRUTH"); axb.set_xlim(-24, 24); axb.set_ylim(0, 48); axb.set_aspect("equal")
        axb.set_title("Top-down BEV  (blue shading = where the farthest vehicle's LiDAR token looks in the image BEV)", fontsize=8.5)
        axb.legend(loc="upper left", fontsize=7, ncol=2, framealpha=.9); axb.tick_params(labelsize=7); axb.set_xlabel("lateral (m)", fontsize=7); axb.set_ylabel("forward (m)", fontsize=7)
        axf.cla(); tt = np.arange(t + 1)
        for k, nm, c in (("lidar_only", "LiDAR-only", COL["lidar"]), ("add", "add-fusion", COL["add"]), ("xattn", "FastBEVFusion", COL["fuse"])):
            axf.plot(tt, hist[k][:t + 1], color=c, lw=2.2, label=nm)
        axf.axvspan(BAD[0], BAD[-1], color="#e74c3c", alpha=.12); axf.text((BAD[0] + BAD[-1]) / 2, .02, "camera extrinsics\nerror 3 deg", ha="center", fontsize=7.5, color="#c0392b")
        axf.set_xlim(0, NF); axf.set_ylim(0, 1.02); axf.grid(alpha=.3); axf.legend(fontsize=7.5, loc="lower left"); axf.set_xlabel("frame", fontsize=8)
        axf.set_title("Running F1 over the drive (cumulative)", fontsize=9.5)
        s = {k: hist[k][t] for k in hist}
        sub.set_text(f"frame {t:02d}   yaw err {math.degrees(d['yaw']):+.1f} deg   F1  LiDAR-only {s['lidar_only']:.2f} | add {s['add']:.2f} | FastBEVFusion {s['xattn']:.2f}")
        return []
    anim = FuncAnimation(fig, draw, frames=NF + 6, interval=170, blit=False)
    anim.save(a.out + ".gif", writer=PillowWriter(fps=6)); anim.save(a.out + ".mp4", writer=FFMpegWriter(fps=6, bitrate=2400))
    fin = {k: round(v[-1], 3) for k, v in hist.items()}; json.dump(dict(final_running_f1=fin, stats={k: v.tolist() for k, v in stats.items()}), open(a.out + ".json", "w"))
    print(fin)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="assets/fastbev_demo"); ap.add_argument("--seed", type=int, default=7)
    main(ap.parse_args())
