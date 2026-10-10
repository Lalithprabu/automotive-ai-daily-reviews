"""Synthetic V2X cooperative-perception simulation: input (ego view + degraded incoming message) vs. ground truth
vs. model predictions (ego-only / old way / Sparse2comm), with live telemetry. Renders a GIF from trained checkpoints.
    python simulate.py --out assets/simulation.gif --frames 28
"""
import argparse, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from sparse2comm import ModelCfg, DataCfg, Sparse2comm, sample_scene, make_batch
from sparse2comm.data import advance
from sparse2comm.metrics import extract_peaks

BG, FG, GRID = "#0e1117", "#e6edf3", "#30363d"
COL = dict(gt="#3fb950", ego="#8b949e", old="#f0883e", new="#ff4d6d", msg="#58a6ff")


def load(name):
    ck = torch.load(f"checkpoints/{name}.pt", map_location="cpu")
    m = Sparse2comm(ModelCfg(**ck["cfg"])); m.load_state_dict(ck["state"]); m.eval()
    return m, DataCfg(**ck["data"])


def detections(out, thr=0.3):
    pk = extract_peaks(out["logits"], thr)[0]
    return pk[:, :2].numpy() if len(pk) else np.zeros((0, 2))


def match(det, gt, r=2.0):
    hit = np.zeros(len(gt), bool)
    for d in det:
        if len(gt):
            dist = np.linalg.norm(gt - d, axis=1); dist[hit] = 1e9; j = dist.argmin()
            if dist[j] <= r: hit[j] = True
    return hit


def pick_scene(dcfg, seed, frames=28):
    """Pick a scene where the ego is blind to many vehicles over the whole clip and vehicles are well separated."""
    g = torch.Generator().manual_seed(seed)
    best, bs = None, -1
    for _ in range(150):
        sc = sample_scene(1, dcfg, g); tot = 0; cur = sc
        p = sc["pos"][0][sc["valid"][0] > 0]
        if len(p) < 4 or (torch.cdist(p, p) + 99 * torch.eye(len(p))).min() < 4.5:
            continue
        for _ in range(frames):
            b = make_batch(cur, dcfg, torch.Generator().manual_seed(0), delay=0)
            tot += int((b["valid"] - b["ego_vis"]).sum()); cur = advance(cur, dcfg)
        if tot > bs: best, bs = sc, tot
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/simulation.gif"); ap.add_argument("--frames", type=int, default=28)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--keep", type=float, default=0.3)
    ap.add_argument("--loss", type=float, default=0.25); a = ap.parse_args()
    models = {k: load(k) for k in ("ego_only", "old_way", "sparse2comm")}
    dcfg = models["sparse2comm"][1]
    scene = pick_scene(dcfg, a.seed, a.frames)
    off = torch.tensor([[1.2, -0.9]])
    g = torch.Generator().manual_seed(a.seed)
    hist = {k: [] for k in ("old_way", "sparse2comm", "ego_only")}; hid_hit = {k: [0, 0] for k in hist}
    frames = []
    for t in range(a.frames):
        d = [0, 1, 2, 3, 3, 2, 1][t % 7] if t >= 4 else 2          # delay sweeps up and down
        b = make_batch(scene, dcfg, g, delay=d, keep_ratio=a.keep, loss_p=a.loss, offset=off)
        with torch.no_grad():
            outs = {k: models[k][0](b) for k in models}
        gt = b["pos"][0][b["valid"][0] > 0].numpy(); hidden = (b["ego_vis"][0] < 1)[b["valid"][0] > 0].numpy()
        dets = {k: detections(o) for k, o in outs.items()}
        for k in hist:
            h = match(dets[k], gt); hid_hit[k][0] += int(h[hidden].sum()); hid_hit[k][1] += int(hidden.sum())
            hist[k].append(hid_hit[k][0] / max(1, hid_hit[k][1]))
        fig = plt.figure(figsize=(15, 5.6), facecolor=BG)
        gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.3], hspace=0.38, wspace=0.18, left=0.04, right=0.985, top=0.9, bottom=0.08)
        axs = [fig.add_subplot(gs[0, i]) for i in range(3)]; axt = fig.add_subplot(gs[1, :])
        for ax in axs + [axt]:
            ax.set_facecolor(BG); [s.set_color(GRID) for s in ax.spines.values()]; ax.tick_params(colors=FG, labelsize=7)
        W, H = dcfg.W, dcfg.H
        # (1) ego view
        ax = axs[0]; ax.imshow(b["ego"][0].norm(dim=0), cmap="magma", vmin=0, vmax=2, origin="lower", extent=(-.5, W-.5, -.5, H-.5))
        ox, oy, ow, oh = scene["occ"][0].tolist(); ax.add_patch(Rectangle((ox, oy), ow, oh, fill=False, ec=COL["ego"], ls="--", lw=1.5, hatch="//"))
        ax.set_title("INPUT 1 · ego camera/LiDAR BEV (dashed = blind zone)", color=FG, fontsize=9)
        # (2) incoming message
        ax = axs[1]; ax.set_xlim(-.5, W-.5); ax.set_ylim(-.5, H-.5); ax.set_aspect("equal"); ax.grid(color=GRID, lw=.4)
        mk = b["mask"][0, 1, 0].numpy(); ys, xs = np.nonzero(mk)
        ax.scatter(xs, ys, s=14, c=COL["msg"], marker="s", label=f"received cells ({len(xs)})")
        ax.scatter(*gt.T, s=90, facecolors="none", edgecolors=COL["gt"], lw=1.2, label="true position now (t)")
        ax.set_title(f"INPUT 2 · V2X message: delay={d}f · misaligned · {100*b['bandwidth'][0]:.1f}% of cells", color=FG, fontsize=9)
        ax.legend(loc="lower left", fontsize=6.5, facecolor=BG, labelcolor=FG, edgecolor=GRID)
        # (3) overlay
        ax = axs[2]; ax.set_xlim(-.5, W-.5); ax.set_ylim(-.5, H-.5); ax.set_aspect("equal"); ax.grid(color=GRID, lw=.4)
        ax.add_patch(Rectangle((ox, oy), ow, oh, fill=False, ec=COL["ego"], ls="--", lw=1))
        ax.scatter(*gt.T, s=150, facecolors="none", edgecolors=COL["gt"], lw=2, label="ground truth")
        for k, mk_, lab in (("ego_only", "1", "ego only"), ("old_way", "x", "old way (sparse, unaligned)"), ("sparse2comm", "+", "Sparse2comm-style")):
            if len(dets[k]): ax.scatter(*dets[k].T, s=70, c=COL[{"ego_only": "ego", "old_way": "old", "sparse2comm": "new"}[k]], marker={"1": "1", "x": "x", "+": "+"}[mk_], lw=2, label=lab)
        fl = outs["sparse2comm"]["flow"][0].numpy()
        for p in dets["sparse2comm"]:
            x, y = int(round(p[0])), int(round(p[1])); ax.arrow(p[0], p[1], fl[0, y, x], fl[1, y, x], color=COL["new"], head_width=.5, alpha=.6, length_includes_head=True)
        ax.legend(loc="lower left", fontsize=6.5, facecolor=BG, labelcolor=FG, edgecolor=GRID, ncol=1)
        ax.set_title(f"OUTPUT · prediction vs ground truth (arrows = predicted motion flow) · t={t}", color=FG, fontsize=9)
        # telemetry
        for k, c, lab in (("ego_only", COL["ego"], "ego only"), ("old_way", COL["old"], "old way"), ("sparse2comm", COL["new"], "Sparse2comm-style")):
            axt.plot(hist[k], c=c, lw=2, label=lab)
        axt.set_xlim(0, a.frames - 1); axt.set_ylim(0, 1.02); axt.set_xlabel("frame", color=FG, fontsize=8)
        axt.set_ylabel("cum. recall of\nego-HIDDEN vehicles", color=FG, fontsize=8); axt.grid(color=GRID, lw=.4)
        axt.legend(loc="upper left", fontsize=7, facecolor=BG, labelcolor=FG, edgecolor=GRID, ncol=3)
        fig.suptitle("Synthetic V2X cooperative detection under bandwidth limits, packet loss, delay & misalignment  |  repo's own simulator, not paper data",
                     color=FG, fontsize=10, y=0.975)
        fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())); plt.close(fig)
        scene = advance(scene, dcfg)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=450, loop=0, optimize=True)
    print(f"wrote {a.out} ({len(frames)} frames)  final hidden-vehicle recall: " + ", ".join(f"{k}={hist[k][-1]:.2f}" for k in hist))


if __name__ == "__main__":
    main()
