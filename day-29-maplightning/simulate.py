"""Synthetic 'AI in action' simulation: a drive through a curving road while the camera extrinsics
drift (speed bump / hard braking). Overlays INPUT (camera frame) vs GROUND TRUTH (green) vs the
dense-BEV 'old way' (red) vs MapLightning-style 1D map tokens (blue), with live telemetry.
Usage: python simulate.py --out assets/maplightning_simulation.gif"""
import argparse, math, sys, warnings
from pathlib import Path
import numpy as np, torch, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")
from maplightning import MapLightning, BEVBaseline
from maplightning.data import (X_MIN, X_MAX, Y_HALF, N_PTS, IMG_W, IMG_H, NOM, render, project,
                               CLS_BOUNDARY, CLS_DIVIDER, CLS_NONE)

BG, GT, OLD, NEW = "#0f1419", "#34d399", "#f87171", "#60a5fa"


def scene_at(t, T):
    """Deterministic evolving road: curvature sweeps left -> right, lane offset wobbles."""
    ph = t / (T - 1)
    xs = torch.linspace(X_MIN, X_MAX, N_PTS); dx = xs - X_MIN
    a, b = 0.06 * math.sin(2 * math.pi * ph), 0.0012 * math.cos(2 * math.pi * ph)
    off, w = 0.6 * math.sin(4 * math.pi * ph), 6.0 + 0.6 * math.sin(2 * math.pi * ph)
    yc = off + a * dx + b * dx ** 2
    offs = torch.tensor([w / 2, -w / 2, -0.6, 1.4])
    polys = torch.stack([xs.expand(4, -1), yc[None] + offs[:, None]], -1)[None]
    cls = torch.tensor([[CLS_BOUNDARY, CLS_BOUNDARY, CLS_DIVIDER, CLS_DIVIDER]])
    return polys, cls, torch.tensor([[True, True, True, ph > 0.25]])


def cam_at(t, T, peak_deg):
    """Calibration drift: a smooth bump in pitch (+yaw, height) peaking mid-drive."""
    k = math.exp(-((t / (T - 1) - 0.55) / 0.18) ** 2)
    return dict(h=torch.tensor([NOM["h"] - 0.03 * peak_deg * k]), pitch=torch.tensor([NOM["pitch"] + math.radians(peak_deg) * k]),
                yaw=torch.tensor([NOM["yaw"] + math.radians(0.6 * peak_deg) * k])), peak_deg * k


def match_err(pts, logits, gt, cls, present):
    keep = (logits.argmax(-1) != CLS_NONE).nonzero().squeeze(-1)
    g = present.nonzero().squeeze(-1)
    if len(keep) == 0:
        return float("nan"), []
    from scipy.optimize import linear_sum_assignment
    d = (pts[keep][:, None] - gt[g][None]).norm(dim=-1).mean(-1)
    bad = logits.argmax(-1)[keep][:, None] != cls[g][None]
    qi, gi = linear_sum_assignment((d + 1e3 * bad.float()).numpy())
    ok = [(int(keep[q]), int(g[k])) for q, k in zip(qi, gi) if not bad[q, k]]
    return float(np.mean([d[q, k] for q, k in zip(qi, gi) if not bad[q, k]] or [float("nan")])), ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="assets/maplightning_simulation.gif")
    ap.add_argument("--frames", type=int, default=36); ap.add_argument("--peak", type=float, default=6.0)
    ap.add_argument("--ckpt", default="outputs"); a = ap.parse_args()
    cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml"))
    ml, bev = MapLightning(cfg["d_model"], cfg["n_map_tokens"]), BEVBaseline(cfg["d_model"])
    ml.load_state_dict(torch.load(f"{a.ckpt}/maplightning.pt")); bev.load_state_dict(torch.load(f"{a.ckpt}/bev_baseline.pt"))
    ml.eval(), bev.eval()
    gen = torch.Generator().manual_seed(7)
    T, frames, e_new, e_old, jit = a.frames, [], [], [], []
    for t in range(T):
        polys, cls, present = scene_at(t, T)
        cam, j = cam_at(t, T, a.peak)
        img = render(polys, cls, present, cam, gen=gen)
        with torch.no_grad():
            ln, pn = ml(img); lo, po = bev(img)             # bev uses NOMINAL calibration internally
        en, okn = match_err(pn[0], ln[0], polys[0], cls[0], present[0]); eo, oko = match_err(po[0], lo[0], polys[0], cls[0], present[0])
        e_new.append(en); e_old.append(eo); jit.append(j)
        fig = plt.figure(figsize=(15, 5.2), dpi=72, facecolor=BG)
        gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1.1], height_ratios=[1, 1], wspace=0.22, hspace=0.45)
        ax = fig.add_subplot(gs[:, 0]); ax.imshow(img[0].permute(1, 2, 0).numpy(), interpolation="nearest")
        for P, col, ok in ((pn[0], NEW, okn), (po[0], OLD, oko)):
            for q, _ in ok:
                p3 = torch.cat([P[q], torch.zeros(N_PTS, 1)], -1)[None]
                u, v, zc = project(p3, cam["h"], cam["pitch"], cam["yaw"]); m = (zc[0] > .5).numpy()
                ax.plot(u[0].numpy()[m], v[0].numpy()[m], color=col, lw=1.6, alpha=.9)
        ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("INPUT: camera frame (preds reprojected w/ true cam)", color="w", fontsize=10)
        bx = fig.add_subplot(gs[:, 1]); bx.set_facecolor("#161c24")
        for g in present[0].nonzero().squeeze(-1):
            bx.plot(polys[0, g, :, 1], polys[0, g, :, 0], color=GT, lw=3, alpha=.55, label="ground truth" if g == 0 else None)
        for P, col, ok, lab in ((po[0], OLD, oko, "BEV baseline (nominal calib)"), (pn[0], NEW, okn, "MapLightning-style")):
            for i, (q, _) in enumerate(ok):
                bx.plot(P[q, :, 1], P[q, :, 0], color=col, lw=1.6, label=lab if i == 0 else None)
        bx.add_patch(plt.Rectangle((-.9, 0), 1.8, 4.2, color="w", alpha=.8))
        bx.set_xlim(Y_HALF, -Y_HALF); bx.set_ylim(0, X_MAX + 2); bx.tick_params(colors="w", labelsize=8)
        bx.set_xlabel("y (m)  [left ←]", color="w", fontsize=8); bx.set_ylabel("x forward (m)", color="w", fontsize=8)
        bx.set_title("BEV: GT vs old way vs new way", color="w", fontsize=10); bx.legend(fontsize=7, loc="lower right", facecolor=BG, labelcolor="w")
        tx = fig.add_subplot(gs[0, 2]); tx.set_facecolor("#161c24"); ts = np.arange(T)
        tx.plot(ts[:t + 1], jit, color="#fbbf24"); tx.set_ylim(0, a.peak * 1.1); tx.axvline(t, color="w", lw=.8)
        tx.set_title("camera pitch perturbation (deg)", color="w", fontsize=9); tx.tick_params(colors="w", labelsize=7); tx.set_xlim(0, T - 1)
        ex = fig.add_subplot(gs[1, 2]); ex.set_facecolor("#161c24")
        ex.plot(ts[:t + 1], e_old[:t + 1], color=OLD, label=f"BEV baseline {e_old[-1]:.2f} m")
        ex.plot(ts[:t + 1], e_new[:t + 1], color=NEW, label=f"MapLightning-style {e_new[-1]:.2f} m")
        ex.set_xlim(0, T - 1); ex.set_ylim(0, 1.2); ex.axvline(t, color="w", lw=.5)
        ex.set_title("live polyline error (m, matched, lower = better)", color="w", fontsize=9)
        ex.tick_params(colors="w", labelsize=7); ex.legend(fontsize=7, facecolor=BG, labelcolor="w", loc="upper left")
        fig.suptitle("Synthetic simulation — this repo's own toy data, not the paper's benchmarks", color="#9ca3af", fontsize=9, y=.99)
        fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())); plt.close(fig)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=220, loop=0, optimize=True)
    pk = int(np.argmax(jit))
    print(f"wrote {a.out} ({T} frames). mean err old {np.nanmean(e_old):.3f} m vs new {np.nanmean(e_new):.3f} m; "
          f"at peak perturbation (frame {pk}, {jit[pk]:.1f} deg) old {e_old[pk]:.3f} vs new {e_new[pk]:.3f}")
    Image.fromarray(np.asarray(frames[pk])).save(Path(a.out).with_name("peak_frame.png"))


if __name__ == "__main__":
    main()
