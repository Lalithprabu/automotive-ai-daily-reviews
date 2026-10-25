"""Real-time-style synthetic simulation: 4 camera feeds with predicted-depth overlay + BEV with ground truth
vs. model-predicted detections/occupancy + live telemetry, rendered to a GIF from the TRAINED checkpoint.
  python simulate.py --ckpt checkpoints/joint.pt --out assets/foundation4d_simulation.gif
"""
import argparse, math, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from src.synthetic import make_scene, render_frame, bev_targets, BEV_N, BEV_R, MAX_D
from src.foundation4d import Foundation4DMini
from src.metrics import peaks

CAMS = ["FRONT", "RIGHT", "REAR", "LEFT"]


def load(ckpt):
    c = torch.load(ckpt, map_location="cpu"); m, d = c["cfg"]["model"], c["cfg"]["data"]
    net = Foundation4DMini(m["dim"], m["heads"], m["depth"], m["patch"], m["n_views"], m["n_frames"], (d["img_h"], d["img_w"]))
    net.load_state_dict(c["state"]); return net.eval(), c["det_thr"], d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="checkpoints/joint.pt")
    ap.add_argument("--out", default="assets/foundation4d_simulation.gif"); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--frames", type=int, default=10); a = ap.parse_args()
    net, thr, d = load(a.ckpt); H, W = d["img_h"], d["img_w"]
    g = torch.Generator().manual_seed(a.seed)
    sc = make_scene(g, 1); sc["vel"] = sc["vel"] * 1.0
    frames, tp_hist, fp_hist, ar_hist = [], [], [], []
    for t in range(1, a.frames + 1):
        r0, d0, _ = render_frame(sc, float(t - 1), H, W, g=g); r1, d1, pos = render_frame(sc, float(t), H, W, g=g)
        rgb = torch.stack([r0, r1], 2)
        with torch.no_grad():
            out = net(rgb)
        heat, occ = bev_targets(pos, sc["half"], sc["mask"])
        pk = peaks(torch.sigmoid(out["det"]), thr)[0]
        gt = (heat[0, 0] == 1.0).nonzero()[:, [1, 0]]
        tp = sum(1 for p in pk.float() if len(gt) and (gt.float() - p).norm(dim=1).min() <= 1.5)
        fp_hist.append(len(pk) - tp); tp_hist.append(tp)
        m = (d1[0] < MAX_D - 1e-3) & (d1[0] > 1.0)   # ignore degenerate camera-inside-box pixels
        ar_hist.append(((out["depth"][0, :, -1] - d1[0]).abs() / d1[0])[m].mean().item())
        fig = plt.figure(figsize=(14, 6.2), dpi=80, facecolor="#0e1117")
        gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1.25, 1.25, 0.01], hspace=0.28, wspace=0.3)
        for i in range(4):
            ax = fig.add_subplot(gs[i // 2, i % 2]); ax.imshow(rgb[0, i, -1].permute(1, 2, 0).numpy(), interpolation="nearest")
            dm = out["depth"][0, i, -1].numpy(); ax.imshow(dm, cmap="turbo", alpha=0.35, vmin=2, vmax=30, interpolation="nearest")
            ax.set_title(f"{CAMS[i]}  (RGB + predicted depth)", color="w", fontsize=9); ax.axis("off")
        ax = fig.add_subplot(gs[:, 2]); ax.set_facecolor("#10151d")
        ax.imshow(torch.sigmoid(out["det"])[0, 0].numpy(), extent=[-BEV_R, BEV_R, -BEV_R, BEV_R], origin="lower", cmap="magma", alpha=0.75, vmin=0, vmax=1)
        for k in range(sc["mask"].shape[1]):
            if sc["mask"][0, k]:
                cx, cy = pos[0, k].tolist(); hx, hy = sc["half"][0, k].tolist()
                ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, fill=False, ec="#39ff14", lw=1.8))
        cell = 2 * BEV_R / BEV_N
        for p in pk.float():
            ax.plot(-BEV_R + (p[0] + .5) * cell, -BEV_R + (p[1] + .5) * cell, "c+", ms=12, mew=2)
        ax.plot(0, 0, "w^", ms=9); ax.set_xlim(-BEV_R, BEV_R); ax.set_ylim(-BEV_R, BEV_R)
        ax.set_title("BEV: green=GT, cyan +=pred", color="w", fontsize=9); ax.tick_params(colors="gray")
        ax = fig.add_subplot(gs[:, 3]); ax.set_facecolor("#10151d")
        ax.imshow(torch.sigmoid(out["occ"])[0, :3].mean(0).numpy(), extent=[-BEV_R, BEV_R, -BEV_R, BEV_R], origin="lower", cmap="Blues", vmin=0, vmax=1)
        ax.contour(np.linspace(-BEV_R, BEV_R, BEV_N), np.linspace(-BEV_R, BEV_R, BEV_N), occ[0, 0].numpy(), levels=[0.5], colors="#39ff14", linewidths=1.5)
        ax.set_title("Occupancy: green=GT", color="w", fontsize=9); ax.tick_params(colors="gray")
        ax.text(-15.5, -15, f"t={t:02d}  det TP {tp}/{len(gt)}  FP {len(pk)-tp}\nrunning depth AbsRel {np.mean(ar_hist):.3f}", color="w", fontsize=8,
                bbox=dict(fc="#000000aa", ec="none"))
        fig.suptitle("Foundation4D-Mini  |  one model: depth + detection + occupancy + Gaussians  |  SYNTHETIC data", color="w", fontsize=11)
        fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])); plt.close(fig)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=450, loop=0)
    print(f"saved {a.out} ({len(frames)} frames) | mean TP/frame {np.mean(tp_hist):.2f} mean FP/frame {np.mean(fp_hist):.2f} depth AbsRel {np.mean(ar_hist):.3f}")


if __name__ == "__main__":
    main()
