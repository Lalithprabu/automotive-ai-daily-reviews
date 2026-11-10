"""Real-time synthetic V2X simulation: INPUT vs GROUND TRUTH vs uncorrected vs CoMET overlay.

    python simulate.py --seed 5000 --delay 200 --out assets/comet_simulation.gif [--mp4 assets/comet_simulation.mp4]

Everything on screen is computed live by the CoMETTracker in this repo on a SYNTHETIC world
(not V2X-Seq, not the authors' code).
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Rectangle

from comet import CoMETConfig, CoMETTracker
from comet.metrics import fuse, match_greedy, prf
from comet.world import WorldConfig, generate

BG, FG, GRID = "#0e1117", "#e6edf3", "#30363d"
C_EGO, C_RSU, C_HIT, C_FP, C_MISS, C_GT, C_COMET = "#58a6ff", "#f0883e", "#3fb950", "#f85149", "#58a6ff", "#7ee787", "#d2a8ff"
THR = 1.0


def run(seed, delay, stride):
    wc = WorldConfig(delay_ms=delay)
    frames = generate(seed, wc)
    tr = CoMETTracker(CoMETConfig())
    tr1 = CoMETTracker(CoMETConfig(use_filter=False))
    log = []
    for f in frames:
        e = torch.tensor(f.ego_xy, dtype=torch.float32)[None]
        r = torch.tensor(f.road_xy, dtype=torch.float32)[None]
        em = torch.ones(1, e.shape[1], dtype=torch.bool)
        rm = torch.ones(1, r.shape[1], dtype=torch.bool)
        o = tr.step(e, r, em, rm, dt=wc.dt)
        o1 = tr1.step(e, r, em, rm, dt=wc.dt)
        gt_all_mask = f.gt_vis_ego | f.gt_vis_rsu
        gt = f.gt_xy[gt_all_mask]
        row = dict(f=f, comet=o["corrected_road_xy"][0].numpy(), off=o["offset"][0].numpy(),
                   off1=o1["offset"][0].numpy(), var=float(o["variance"][0]), npairs=int(o["n_pairs"][0]))
        for k, rd in [("raw", f.road_xy), ("single", o1["corrected_road_xy"][0].numpy()), ("comet", row["comet"])]:
            pred = fuse(f.ego_xy, rd)
            hp, hg = match_greedy(pred, gt, THR)
            row[k + "_pred"], row[k + "_hp"] = pred, hp
            row[k + "_cnt"] = np.array([hg.sum(), (~hp).sum(), (~hg).sum()])
        row["gt_all"] = gt
        log.append(row)
    return wc, log


def style(ax, title, lim=((-20, 52), (-22, 30))):
    ax.set_facecolor(BG)
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal")
    ax.set_title(title, color=FG, fontsize=11, loc="left")
    ax.tick_params(colors=FG, labelsize=7)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.4)


def boxes(ax, pts, color, hp=None, lw=1.6):
    for i, (x, y) in enumerate(pts):
        c = color if hp is None else (C_HIT if hp[i] else C_FP)
        ax.add_patch(Rectangle((x - 1.1, y - 0.6), 2.2, 1.2, fill=False, ec=c, lw=lw))


def coverage(ax, wc):
    ax.add_patch(Circle((0, 0), wc.ego_range, fill=False, ec=C_EGO, ls="--", lw=0.8, alpha=0.6))
    ax.add_patch(Circle(wc.rsu_pos, wc.rsu_range, fill=False, ec=C_RSU, ls="--", lw=0.8, alpha=0.6))
    ax.plot(0, 0, marker="s", color=C_EGO, ms=8)
    ax.plot(*wc.rsu_pos, marker="^", color=C_RSU, ms=9)


def render(wc, log, stride, delay):
    fig = plt.figure(figsize=(16, 9), dpi=80, facecolor=BG)
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.14, left=0.04, right=0.985, top=0.9, bottom=0.07)
    axs = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    imgs = []
    tt = np.arange(len(log)) * wc.dt
    true_err = np.array([r["f"].true_err for r in log])
    est = np.array([r["off"] for r in log]); est1 = np.array([r["off1"] for r in log])
    cum = {k: np.cumsum([r[k + "_cnt"] for r in log], 0) for k in ["raw", "single", "comet"]}
    for t in range(0, len(log), stride):
        r, f = log[t], log[t]["f"]
        for a in axs:
            a.clear()
        # 1 INPUT
        a = axs[0]; style(a, f"INPUT  (roadside msg age {delay:.0f} ms, SHARED pose error applied)")
        coverage(a, wc)
        boxes(a, f.ego_xy, C_EGO); boxes(a, f.road_xy, C_RSU)
        a.legend(handles=[Rectangle((0, 0), 1, 1, fill=False, ec=C_EGO, label="ego detections"),
                          Rectangle((0, 0), 1, 1, fill=False, ec=C_RSU, label="roadside msg (shifted)")],
                 loc="lower left", fontsize=7, facecolor=BG, labelcolor=FG, edgecolor=GRID)
        # 2 GROUND TRUTH
        a = axs[1]; style(a, "GROUND TRUTH  (every object either sensor can see)")
        coverage(a, wc)
        for k in range(len(f.gt_xy)):
            if not (f.gt_vis_ego[k] or f.gt_vis_rsu[k]):
                continue
            tr_ = np.array([log[max(0, t - j)]["f"].gt_xy[k] for j in range(10)])
            a.plot(tr_[:, 0], tr_[:, 1], color=C_GT, lw=1, alpha=0.5)
            boxes(a, [f.gt_xy[k]], C_GT)
            if f.gt_vis_rsu[k] and not f.gt_vis_ego[k]:
                a.plot(*f.gt_xy[k], marker="o", ms=3, color=C_RSU)
        a.text(-19, 26, "dots = roadside-only objects", color=C_RSU, fontsize=7)
        # 3 UNCORRECTED / 4 COMET
        for ai, key, name in [(2, "raw", "UNCORRECTED online pose"), (3, "comet", "CoMET (Kalman-tracked shared offset)")]:
            a = axs[ai]; c = r[key + "_cnt"]; p_, r_, f1 = prf(*c)
            style(a, f"{name}   F1@{THR:.0f}m now: {prf(*c)[2]:.2f}")
            coverage(a, wc)
            boxes(a, r[key + "_pred"], None, r[key + "_hp"])
            for x, y in r["gt_all"]:
                a.plot(x, y, marker="+", color=C_GT, ms=6, alpha=0.7)
            if key == "comet":
                o = r["off"]
                a.annotate("", xy=tuple(np.array(wc.rsu_pos) - o), xytext=wc.rsu_pos,
                           arrowprops=dict(arrowstyle="->", color=C_COMET, lw=2))
                a.text(-19, 26, f"est. offset ({o[0]:+.2f},{o[1]:+.2f}) m  pairs={r['npairs']}", color=C_COMET, fontsize=8)
            a.text(-19, -20, "green=hit  red=false alarm  '+' = ground truth", color=FG, fontsize=7)
        # 5 offset telemetry
        a = axs[4]; a.set_facecolor(BG); a.set_title("TELEMETRY: shared pose error (x component)", color=FG, fontsize=11, loc="left")
        a.plot(tt[: t + 1], true_err[: t + 1, 0], color=C_FP, lw=2, label="true error")
        a.plot(tt[: t + 1], est1[: t + 1, 0], color=C_RSU, lw=1, alpha=0.8, label="current msg only")
        a.plot(tt[: t + 1], est[: t + 1, 0], color=C_COMET, lw=2, label="CoMET estimate")
        for j in range(t + 1):
            if log[j]["npairs"] == 0:
                a.axvspan(tt[j] - 0.05, tt[j] + 0.05, color=GRID, alpha=0.6)
        a.set_xlim(0, tt[-1]); lo, hi = true_err[:, 0].min() - 1, true_err[:, 0].max() + 1; a.set_ylim(lo, hi)
        a.tick_params(colors=FG, labelsize=7); a.grid(color=GRID, lw=0.4)
        for s in a.spines.values(): s.set_color(GRID)
        a.legend(fontsize=7, facecolor=BG, labelcolor=FG, edgecolor=GRID, loc="upper left")
        a.set_xlabel("time [s]  (grey bands: no matched pair, estimate carried forward)", color=FG, fontsize=7)
        # 6 F1 telemetry
        a = axs[5]; a.set_facecolor(BG); a.set_title("TELEMETRY: cumulative F1 @ 1 m", color=FG, fontsize=11, loc="left")
        for k, col, lab in [("raw", C_FP, "uncorrected"), ("single", C_RSU, "current msg only"), ("comet", C_COMET, "CoMET")]:
            f1s = [prf(*cum[k][j])[2] for j in range(t + 1)]
            a.plot(tt[: t + 1], f1s, color=col, lw=2, label=f"{lab}  {f1s[-1]:.2f}")
        a.set_xlim(0, tt[-1]); a.set_ylim(0, 1.02)
        a.tick_params(colors=FG, labelsize=7); a.grid(color=GRID, lw=0.4)
        for s in a.spines.values(): s.set_color(GRID)
        a.legend(fontsize=8, facecolor=BG, labelcolor=FG, edgecolor=GRID, loc="lower right")
        a.set_xlabel("time [s]", color=FG, fontsize=7)
        fig.suptitle("Day 67 - CoMET re-implementation: tracking the shared roadside pose error   |   SYNTHETIC toy world - not V2X-Seq, not the authors' code",
                     color=FG, fontsize=12, x=0.04, ha="left", y=0.965)
        fig.canvas.draw()
        imgs.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return imgs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--delay", type=float, default=200.0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="assets/comet_simulation.gif")
    ap.add_argument("--mp4", default=None)
    a = ap.parse_args()
    wc, log = run(a.seed, a.delay, a.stride)
    imgs = render(wc, log, a.stride, a.delay)
    import imageio.v2 as iio
    iio.mimsave(a.out, imgs, duration=0.125, loop=0)
    if a.mp4:
        iio.mimsave(a.mp4, imgs, fps=8, macro_block_size=1)
    print("frames:", len(imgs), "size:", imgs[0].shape, "->", a.out)
    for k in ["raw", "single", "comet"]:
        c = sum(r[k + "_cnt"] for r in log)
        print(f"{k:7s} F1@{THR:.0f}m = {prf(*c)[2]:.3f}")
