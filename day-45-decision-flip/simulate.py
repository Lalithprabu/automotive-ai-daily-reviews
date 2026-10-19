"""Real-time-style synthetic simulation: progressive Mean-FLIP pruning vs random pruning on a live scene.

Panels: [camera-feature feed] [ground truth BEV] [Mean-FLIP pruned prediction + Gaussian centers] [random-pruned prediction]
        + telemetry (rows kept, mIoU, latency) + live mIoU-vs-rows-kept curve. Prediction is overlaid in red where it
        disagrees with ground truth. Writes assets/decision_flip_simulation.gif from the REAL trained checkpoint.
"""
import argparse, json, torch, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import io
from decision_flip.model import GaussianOccModel
from decision_flip.data import make_scene
from decision_flip.flip import compile_pruned, topk_mask
from decision_flip.metrics import confusion_miou

CMAP = ListedColormap(["#101820", "#6b7280", "#f59e0b", "#60a5fa", "#34d399"])  # empty, road, vehicle, building, vegetation


def bev(occ):  # top-most non-empty class per (x,y) column
    z = occ.shape[2]; out = np.zeros(occ.shape[:2], dtype=int)
    for k in range(z): out = np.where(occ[:, :, k] > 0, occ[:, :, k], out)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scene_seed", type=int, default=20000)
    ap.add_argument("--out", default="assets/decision_flip_simulation.gif"); a = ap.parse_args()
    ck = torch.load("checkpoints/dense.pt"); cfg = ck["cfg"]; res = json.load(open("checkpoints/prune_results.json"))
    model = GaussianOccModel(cfg); model.load_state_dict(ck["state"]); model.eval()
    flip = torch.tensor(res["scores"]); rnd = torch.rand(model.bank.n, generator=torch.Generator().manual_seed(3))
    N = model.bank.n; grid = (16, 16, 4)
    feat, occ = make_scene(a.scene_seed, grid); gt = bev(occ.numpy())
    fracs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]; kept_final = res["kept_rows"] / N
    fracs = sorted(set(fracs + [kept_final]), reverse=True)
    frames, hist = [], {"flip": [], "rand": [], "x": []}
    for f in fracs:
        k = max(int(round(N * f)), 1); row = {}
        for nm, s in (("flip", flip), ("rand", rnd)):
            m = topk_mask(s, k); pm = compile_pruned(model, m)
            with torch.no_grad(): pr = pm(feat[None]).argmax(-1).reshape(grid)
            row[nm] = (pr, m, confusion_miou(pr, occ, 5))
        hist["flip"].append(row["flip"][2]); hist["rand"].append(row["rand"][2]); hist["x"].append(k)
        fig = plt.figure(figsize=(15.5, 7.2), facecolor="#0b0f14"); gs = fig.add_gridspec(2, 4, height_ratios=[3, 2])
        axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
        axs[0].imshow(feat[:4].max(0).values.T.numpy() * 0 + feat[4].T.numpy(), cmap="magma", origin="lower"); axs[0].set_title("Sensor feed (BEV height feats, noisy, occluded)", color="w", fontsize=9)
        axs[1].imshow(gt.T, cmap=CMAP, vmin=0, vmax=4, origin="lower"); axs[1].set_title("Ground truth occupancy (BEV)", color="w", fontsize=9)
        for ax, nm, ttl in ((axs[2], "flip", "Mean-FLIP pruned"), (axs[3], "rand", "Random pruned (same rows)")):
            pr, m, mi = row[nm]; pb = bev(pr.numpy()); ax.imshow(pb.T, cmap=CMAP, vmin=0, vmax=4, origin="lower")
            bad = (pb != gt); ov = np.zeros(gt.shape + (4,)); ov[bad] = [1, 0, 0, 0.55]; ax.imshow(ov.transpose(1, 0, 2), origin="lower")
            mu = model.bank.mean.detach()
            ax.scatter(mu[m, 0] / 16 * 16 - 0.5, mu[m, 1] - 0.5, s=5, c="#a7f3d0", alpha=.8)
            ax.scatter(mu[~m, 0] - 0.5, mu[~m, 1] - 0.5, s=12, c="#ef4444", marker="x", alpha=.5, linewidths=.7)
            ax.set_title(f"{ttl}: {int(m.sum())}/{N} rows | mIoU {mi:.3f}", color="w", fontsize=9)
        for ax in axs: ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.5, 15.5); ax.set_ylim(-0.5, 15.5)
        ax = fig.add_subplot(gs[1, :2]); ax.set_facecolor("#0b0f14")
        sw = res["sweep"]; xs_ = [p[1] for p in sw["mean_flip"]]
        ax.plot(xs_, [p[2] for p in sw["mean_flip"]], "-", c="#34d399", label="Mean-FLIP (96 val scenes)"); ax.plot(xs_, [p[2] for p in sw["random"]], "-", c="#f87171", label="Random (96 val scenes)")
        ax.plot(xs_, [p[2] for p in sw["taylor_supervised"]], ":", c="#fbbf24", label="Supervised Taylor"); ax.axvline(k, c="w", lw=.8)
        ax.scatter([k], [row["flip"][2]], c="#34d399", s=30); ax.scatter([k], [row["rand"][2]], c="#f87171", s=30)
        ax.axhline(res["dense_miou"], c="#9ca3af", ls="--", lw=.8, label="Dense bank"); ax.invert_xaxis(); ax.set_ylim(0.2, 0.8)
        ax.set_xlabel("bank rows kept", color="w"); ax.set_ylabel("mIoU (lines: val set; dots: this scene)", color="w"); ax.tick_params(colors="w"); ax.legend(facecolor="#111827", labelcolor="w", fontsize=8)
        ax = fig.add_subplot(gs[1, 2:]); ax.set_facecolor("#0b0f14"); ax.axis("off")
        lat = res["latency_ms"]; est = lat["dense"] * (0.15 + 0.85 * k / N)
        ax.text(0, .95, f"TELEMETRY  (synthetic data - not paper numbers)\nrows kept: {k}/{N} ({100*k/N:.0f}%)\nest. bank latency: {est:.0f} ms (dense {lat['dense']:.0f} ms, CPU batch 8)\n"
                f"Otsu auto-cut (full val): keep {res['kept_rows']}/{N}, remove {res['removed_pct']:.0f}%\nzero-flip dead rows: {res['dead_rows_zero_flip']}\n"
                f"green dots = kept Gaussians   red x = removed   red tint = wrong voxel column", color="w", va="top", fontsize=10, family="monospace")
        fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=70, facecolor=fig.get_facecolor()); plt.close(fig)
        frames.append(Image.open(io.BytesIO(buf.getvalue())).convert("P", palette=Image.ADAPTIVE))
    durs = [800] * (len(frames) - 1) + [2500]
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=durs, loop=0)
    print("wrote", a.out, len(frames), "frames")

if __name__ == "__main__":
    main()
