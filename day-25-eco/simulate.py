"""Renders the closed-loop simulation GIF: same scene, same weights, same perception noise, raw waypoints vs ECO.
Layers per panel: perceived route points (INPUT), expert waypoints (GROUND TRUTH), policy waypoints (PREDICTION),
executed trail, ego box; bottom strip = live speed + jerk telemetry for both runs."""
import argparse, yaml, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import io
from evaluate import load
from eco.rollout import run_episode
from eco.env import to_ego

BG, GRID, TXT = "#0e1117", "#232a36", "#e6edf3"
C_IN, C_GT, C_RAW, C_ECO = "#8b949e", "#3fb950", "#f85149", "#58a6ff"


def car(ax, x, y, th, color):
    l, w = 4.2, 1.9
    c = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    ax.add_patch(Polygon(c @ R.T + [x, y], closed=True, fc=color, ec="white", lw=0.8, alpha=0.9, zorder=6))


def _w(pts_ego, x, y, th):
    from eco.env import from_ego
    return from_ego(pts_ego, x, y, th)


def jerk_trace(speeds, dt=0.1):
    a = np.diff(speeds) / dt
    return np.abs(np.diff(a) / dt)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--ckpt", default="policy.pt")
    ap.add_argument("--seed", type=int, default=1003); ap.add_argument("--out", default="assets/eco_simulation.gif")
    ap.add_argument("--stride", type=int, default=5); ap.add_argument("--max_steps", type=int, default=220)
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); pol, eco = load(cfg, a.ckpt)
    runs = {k: run_episode(pol, eco, cfg, a.seed, u, record=True, max_steps=a.max_steps) for k, u in (("raw", False), ("eco", True))}
    n = min(len(runs["raw"]["frames"]), len(runs["eco"]["frames"]))
    jr, je = jerk_trace(runs["raw"]["speed_trace"]), jerk_trace(runs["eco"]["speed_trace"])
    imgs = []
    for i in range(0, n, a.stride):
        fig = plt.figure(figsize=(13, 6.6), facecolor=BG)
        gs = fig.add_gridspec(2, 2, height_ratios=[2.6, 1], hspace=0.28, wspace=0.08, left=0.04, right=0.98, top=0.9, bottom=0.08)
        for col, (key, title, cpred) in enumerate((("raw", "OLD WAY  |  raw waypoints", C_RAW), ("eco", "NEW WAY  |  + ECO layer", C_ECO))):
            ax = fig.add_subplot(gs[0, col]); ax.set_facecolor(BG)
            f = runs[key]["frames"][i]; x, y, th, v = f["state"]
            R = lambda p: to_ego(p[..., 0], p[..., 1], x, y, th)   # ego-centred view, car always heading +x
            tr = R(f["trail"]); ax.plot(tr[:, 0], tr[:, 1], color="#6e7681", lw=1.2, zorder=2)
            _pt = R(_w(f["pts"], x, y, th)); ax.scatter(_pt[:, 0], _pt[:, 1], c=C_IN, s=22, marker="x", zorder=3, label="input: perceived route (noisy)")
            g = f["gt"]; ax.plot(g[:, 0], g[:, 1], "-o", c=C_GT, ms=4, lw=1.6, zorder=4, label="ground truth (expert)")
            p = f[key]; ax.plot(p[:, 0], p[:, 1], "-o", c=cpred, ms=5, lw=2, zorder=5, label="model prediction")
            ax.plot(*p[-1], "*", c="white", ms=11, zorder=7)
            car(ax, 0.0, 0.0, 0.0, cpred)
            ax.set_xlim(-10, 34); ax.set_ylim(-11, 11); ax.set_aspect("equal")
            ax.set_title(title, color=cpred, fontsize=13, fontweight="bold"); ax.tick_params(colors=GRID, labelsize=7)
            for s_ in ax.spines.values(): s_.set_color(GRID)
            ax.text(0.02, 0.97, f"v = {v:4.1f} m/s   wp-jerk = {np.abs(np.diff(f[key], 3, axis=0)).mean():.2f}", transform=ax.transAxes, va="top", color=TXT, fontsize=9, family="monospace")
            if col == 0: ax.legend(loc="lower left", fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=TXT)
        step = min(f["t"] for f in [runs["raw"]["frames"][i]]) 
        for col, (key, arr, c) in enumerate((("raw", jr, C_RAW), ("eco", je, C_ECO))):
            ax = fig.add_subplot(gs[1, col]); ax.set_facecolor(BG)
            t = np.arange(len(arr)) * 0.1
            ax.plot(t, jr, color=C_RAW, lw=0.6, alpha=0.25); ax.plot(t, je, color=C_ECO, lw=0.6, alpha=0.25)
            k = min(step, len(arr)); ax.plot(t[:k], arr[:k], color=c, lw=1.4)
            ax.axvline(step * 0.1, color=TXT, lw=0.6); ax.set_ylim(0, max(jr.max(), je.max()) * 1.05); ax.set_xlim(0, t[-1])
            ax.set_title("live |jerk| (m/s³)  —  rms so far: %.2f" % float(np.sqrt(np.mean(arr[:max(k, 1)] ** 2))), color=TXT, fontsize=9)
            ax.tick_params(colors=TXT, labelsize=7)
            for s_ in ax.spines.values(): s_.set_color(GRID)
        fig.suptitle("Day 25 · ECO: Endpoint-Constrained Optimization — synthetic closed-loop sim (this repo's reconstruction)", color=TXT, fontsize=12)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=70, facecolor=BG); plt.close(fig); buf.seek(0)
        imgs.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    imgs[0].save(a.out, save_all=True, append_images=imgs[1:], duration=120, loop=0, optimize=True)
    m = {k: runs[k] for k in runs}
    print(f"wrote {a.out}: {len(imgs)} frames | rms jerk raw {np.sqrt(np.mean(jr**2)):.2f} vs eco {np.sqrt(np.mean(je**2)):.2f}")


if __name__ == "__main__":
    main()
