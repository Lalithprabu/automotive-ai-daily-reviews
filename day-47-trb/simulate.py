"""Render the 'AI in action' GIF: ego-only input | stale collaborator message | ground truth |
old way (pose-warp only) vs new way (Temporal Residual Bottleneck), with live telemetry.
Uses REAL trained checkpoints on a held-out synthetic scene.  python simulate.py [--delay 5]"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from trb.model import AsyncCollabDetector, TRBConfig
from trb.data import make_batch, HALF, GRID, CELL, MAX_DELAY, DT
from trb.metrics import decode, gt_cells, detection_stats

EXTENT = [-HALF, HALF, HALF, -HALF]


def load(name, ck="checkpoints"):
    d = torch.load(f"{ck}/{name}.pt", map_location="cpu")
    kind = "trb" if name.startswith("trb") else name
    extra = {"use_gate": False} if name == "trb_nogate" else {}
    m = AsyncCollabDetector(kind, TRBConfig(**{**d["model_cfg"], **extra})); m.load_state_dict(d["state"]); m.eval()
    return m, d["thr"]


def fig_to_img(fig):
    fig.canvas.draw(); return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def draw_dets(ax, logits, thr, color, marker, size=70):
    for r, c in decode(logits, thr)[0].nonzero().tolist():
        ax.scatter((c + .5) * CELL - HALF, (r + .5) * CELL - HALF, s=size, facecolors="none", edgecolors=color, marker=marker, linewidths=2)


def panel(ax, img, title, cmap="magma"):
    ax.imshow(img, extent=EXTENT, cmap=cmap, vmin=0, vmax=1.6, interpolation="nearest")
    ax.axvline(0, c="cyan", ls="--", lw=.8); ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-HALF, HALF); ax.set_ylim(HALF, -HALF)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ck", default="checkpoints"); ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--delay", type=int, default=5); ap.add_argument("--out", default="assets/trb_simulation.gif")
    ap.add_argument("--seed0", type=int, default=777000); a = ap.parse_args()
    new, thr_n = load("trb", a.ck); old, thr_o = load("warp", a.ck)
    frames, hist_new, hist_old, rows = [], [], [], []
    for f in range(a.n):                                     # each frame = a fresh held-out scene (a "clip" of scenes)
        b = make_batch(1, a.seed0 + f, delays=[a.delay])
        with torch.no_grad():
            on, oo = new(b), old(b)
        sn = detection_stats(on["logits"], b["pos"], b["mask"], thr_n); so = detection_stats(oo["logits"], b["pos"], b["mask"], thr_o)
        hist_new.append(sn["tp_b"] / max(sn["tp_b"] + sn["fn_b"], 1)); hist_old.append(so["tp_b"] / max(so["tp_b"] + so["fn_b"], 1))
        fig = plt.figure(figsize=(15.5, 7.4), facecolor="#101418")
        gs = fig.add_gridspec(2, 4, height_ratios=[3, 1.35], hspace=.28, wspace=.08)
        axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
        panel(axs[0], b["ego"][0].norm(dim=0), "INPUT 1: ego sensor only\n(right half = blind zone)")
        panel(axs[1], b["hist"][0, 0].norm(dim=0), f"INPUT 2: collaborator msg, {a.delay*DT:.2f}s stale\n(own older frame)")
        panel(axs[2], b["target"][0].norm(dim=0), "GROUND TRUTH (objects now)")
        for ax in axs[1:3]: pass
        mask = b["mask"][0] > 0
        for ax in (axs[2], axs[3]):
            ax.scatter(b["pos"][0, mask, 0], b["pos"][0, mask, 1], s=130, marker="s", facecolors="none", edgecolors="#35e08a", linewidths=2)
        # trajectory trails (last 4 steps of true motion)
        for ax in (axs[2],):
            for n in np.nonzero(mask.numpy())[0]:
                pth = b["paths"][0, n, :6]; ax.plot(pth[:, 0], pth[:, 1], c="#35e08a", lw=1, alpha=.7)
        panel(axs[3], on["F_hat"][0].norm(dim=0), "NEW: TRB prediction (circles)\nvs OLD: pose-warp only (x)")
        draw_dets(axs[3], on["logits"], thr_n, "#ffd23f", "o"); draw_dets(axs[3], oo["logits"], thr_o, "#ff5a5f", "x", 60)
        for ax in axs: ax.set_facecolor("#101418"); ax.title.set_color("white")
        t1 = fig.add_subplot(gs[1, :2]); t1.set_facecolor("#171c22")
        t1.plot(hist_old, c="#ff5a5f", label="old way: pose-warp only"); t1.plot(hist_new, c="#ffd23f", label="new way: TRB")
        t1.set_xlim(0, a.n - 1); t1.set_ylim(-.05, 1.05); t1.set_title("Blind-zone recall per scene (live)", color="white", fontsize=9)
        t1.legend(fontsize=7, loc="lower left"); t1.tick_params(colors="#9aa0a6", labelsize=7)
        t2 = fig.add_subplot(gs[1, 2:]); t2.axis("off"); t2.set_facecolor("#171c22")
        G = on["G"][0, 0].mean().item(); dn = on["dF"][0].norm(dim=0).mean().item()
        err_w = ((oo["F_hat"] - b["target"]) ** 2).mean().item(); err_t = ((on["F_hat"] - b["target"]) ** 2).mean().item()
        txt = (f"scene {f+1}/{a.n}   message delay {a.delay*DT:.2f}s ({a.delay} steps)\n"
               f"feature MSE  old {err_w:.4f}  ->  new {err_t:.4f}   ({100*(1-err_t/max(err_w,1e-9)):+.0f}%)\n"
               f"mean object-gate G {G:.2f}   mean applied residual |dF| {dn:.3f}\n"
               f"running blind-zone recall  old {np.mean(hist_old):.2f}  new {np.mean(hist_new):.2f}")
        t2.text(0, .5, txt, color="white", fontsize=11, family="monospace", va="center")
        fig.text(.01, .975, "green squares = ground truth | cyan dashed = ego/blind-zone boundary | synthetic data, this repo's own reconstruction of arXiv:2610.10090", color="#9aa0a6", fontsize=8, va="top")
        frames.append(fig_to_img(fig)); plt.close(fig)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=700, loop=0, optimize=True)
    print(f"saved {a.out}: {len(frames)} frames, running blind recall old {np.mean(hist_old):.3f} new {np.mean(hist_new):.3f}")


if __name__ == "__main__":
    main()
