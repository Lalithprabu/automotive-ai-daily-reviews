"""Real-time-style synthetic simulation: radar input vs ground truth vs old-way vs RadarGSOcc prediction, with live telemetry.
Usage: python simulate.py --frames 24 --out assets/simulation.gif   (needs checkpoints/ from train.py)"""
import argparse, sys, yaml, numpy as np, torch
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from models.radargsocc import RadarGSOcc, RadarOccBaseline
from src.data import make_scene, render_radar, CLASS_NAMES
from src.metrics import occ_iou

COL = ["#0e1117", "#4b5563", "#d97706", "#22c55e", "#ef4444"]; CM = ListedColormap(COL)

def topdown(occ):              # class at highest occupied z for each (x,y)
    occ = np.asarray(occ); z = (occ > 0) * np.arange(occ.shape[2])[None, None]; top = z.argmax(2)
    return np.take_along_axis(occ, top[..., None], 2)[..., 0].T       # (Y,X) rows=lateral

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frames", type=int, default=24); ap.add_argument("--out", default="assets/simulation.gif")
    ap.add_argument("--config", default="config.yaml"); ap.add_argument("--ckpt", default="checkpoints"); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    new = RadarGSOcc(cfg["feat_dim"], 5, cfg["attn_points"]); new.load_state_dict(torch.load(f"{a.ckpt}/new.pt")); new.eval()
    old = RadarOccBaseline(cfg["baseline_dim"], 5); old.load_state_dict(torch.load(f"{a.ckpt}/old.pt")); old.eval()
    rng = np.random.default_rng(777); hist = {"old": [], "new": [], "snow": []}; frames = []
    for f in range(a.frames):
        snow = 0.0 if f < a.frames // 2 else 0.8                     # second half: heavy snow
        occ, vel = make_scene(rng); r = render_radar(occ, vel, rng, snow=snow)
        with torch.no_grad():
            po = old(torch.from_numpy(r)[None]).argmax(1)[0]; pn = new(torch.from_numpy(r)[None]).argmax(1)[0]
        g = torch.from_numpy(occ); io, i_n = occ_iou(po, g), occ_iou(pn, g)
        hist["old"].append(io); hist["new"].append(i_n); hist["snow"].append(snow)
        fig = plt.figure(figsize=(15, 6.2), facecolor="#0e1117"); gs = fig.add_gridspec(2, 4, height_ratios=[3, 1.6])
        panels = [("Radar input (peak power, top-down)", None), ("Ground truth", occ), ("Old way: early-fusion conv", po.numpy()), ("RadarGSOcc: geo+sem+deformable", pn.numpy())]
        for j, (title, o) in enumerate(panels):
            ax = fig.add_subplot(gs[0, j]); ax.set_facecolor("#0e1117")
            if o is None: ax.imshow(r[0].max(2).T, cmap="magma", origin="lower", vmin=0, vmax=1.2)
            else: ax.imshow(topdown(o), cmap=CM, vmin=0, vmax=4, origin="lower")
            ax.set_title(title, color="w", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
            if j >= 2: ax.text(1, 1, f"occ-IoU {io if j==2 else i_n:.2f}", color="w", fontsize=11, bbox=dict(fc="#000a", ec="none"))
        ax = fig.add_subplot(gs[1, :3]); ax.set_facecolor("#161b22"); xs = range(len(hist["new"]))
        ax.plot(xs, hist["old"], color="#f59e0b", label="old way"); ax.plot(xs, hist["new"], color="#38bdf8", label="RadarGSOcc")
        ax.axvspan(a.frames // 2 - 0.5, a.frames - 0.5, color="#94a3b8", alpha=0.15); ax.text(a.frames // 2, 0.02, "heavy snow", color="#cbd5e1", fontsize=9)
        ax.set_xlim(0, a.frames - 1); ax.set_ylim(0, 1); ax.set_ylabel("occ-IoU", color="w"); ax.tick_params(colors="w"); ax.legend(loc="upper right", facecolor="#161b22", labelcolor="w", fontsize=8)
        ax2 = fig.add_subplot(gs[1, 3]); ax2.axis("off")
        ax2.text(0, 0.9, f"frame {f+1}/{a.frames}   snow {snow:.1f}", color="w", fontsize=11, va="top")
        ax2.text(0, 0.55, "legend: " + " | ".join(CLASS_NAMES[1:]), color="#cbd5e1", fontsize=8, va="top")
        for k, (n, c) in enumerate(zip(CLASS_NAMES[1:], COL[1:])): ax2.add_patch(plt.Rectangle((0.02 + 0.24 * k, 0.1), 0.2, 0.15, color=c))
        fig.suptitle("RadarGSOcc reconstruction — synthetic 4D-radar tensor -> semantic occupancy (NOT the paper's K-Radar data)", color="w", fontsize=11)
        fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())); plt.close(fig)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=450, loop=0)
    print(f"wrote {a.out}: {len(frames)} frames; mean IoU old {np.mean(hist['old'][:a.frames//2]):.3f}/{np.mean(hist['old'][a.frames//2:]):.3f} (clear/snow), new {np.mean(hist['new'][:a.frames//2]):.3f}/{np.mean(hist['new'][a.frames//2:]):.3f}")

if __name__ == "__main__": main()
