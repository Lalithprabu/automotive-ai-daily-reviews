"""Render the "AI in action" simulation: ground truth vs. old-way vs. EgoRefine on one scripted V2V drive.

Panels: [INPUT: ego view + delayed collaborator ghosts] [OLD WAY: TraF-Align-style] [EGOREFINE (+ sampling arrows & gate)]
Bottom: live telemetry (delay, running centre-error for naive / old-way / EgoRefine).  Everything is driven by the
real trained checkpoints in ./checkpoints on SYNTHETIC data (not the paper's V2V4Real / DAIR-V2X-Seq).
Usage: python simulate.py --ckpt checkpoints --out egorefine_simulation.gif [--frames 28]
"""
import argparse, io
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image
from egorefine import AsyncFusionNet, simulate_objects, render_obs
from egorefine.data import DMAX, H, W, _grid
from egorefine.metrics import extract_peaks

BG, GT_C, OLD_C, NEW_C, GHOST = "#0b1220", "#2ee59d", "#ff9f43", "#5aa9ff", "#c9d1e3"
CEN, R = torch.tensor([14.0, 16.0]), 14.0


def load(ck, mode):
    d = torch.load(f"{ck}/{mode}.pt", map_location="cpu"); m = AsyncFusionNet(mode, **d["cfg"]); m.load_state_dict(d["state"]); return m.eval()


def pick_scene(frames, seed0=0):
    for seed in range(seed0, seed0 + 500):
        g = torch.Generator().manual_seed(seed)
        pos, vel, valid = simulate_objects(1, N=5, T=DMAX + frames, gen=g, speed=(0.3, 0.6), start=(12, 28))
        p = pos[0][valid[0]]
        inb = ((p > 3) & (p < 36)).all()
        blind_cross = ((p[:, DMAX:] - CEN).norm(dim=-1) > R).any() and ((p[:, DMAX:] - CEN).norm(dim=-1) <= R).any()
        if inb and valid.sum() >= 3 and blind_cross:
            return pos, vel, valid, seed
    raise RuntimeError("no scene")


def frame_inputs(pos, vel, valid, te, delta, gen):
    pe, ve = pos[:, :, te], vel[:, :, te]; pc, vc = pos[:, :, te - delta], vel[:, :, te - delta]
    xs, ys = _grid(); disk = (((xs - CEN[0]) ** 2 + (ys - CEN[1]) ** 2) <= R ** 2).float()[None]
    vis = (((pe - CEN) ** 2).sum(-1) <= R ** 2) & valid
    ego = torch.cat([render_obs(pe, ve, vis, gen=gen) * disk[:, None], disk[:, None]], 1)
    d_est = torch.tensor([float(delta)]) + 0.7 * torch.randn(1, generator=gen)
    col = torch.cat([render_obs(pc, vc, valid, gen=gen), (d_est.clamp(0, DMAX + 1) / DMAX)[:, None, None, None].expand(1, 1, H, W)], 1)
    return ego, col, pe[0][valid[0]], pc[0][valid[0]], vis[0][valid[0]]


def match_err(peaks, gt, cap=8.0, thr=0.3):
    pk = peaks[peaks[:, 2] >= thr] if len(peaks) else peaks
    errs = []
    for g in gt:
        errs.append(min(cap, float(np.hypot(pk[:, 0] - g[0], pk[:, 1] - g[1]).min())) if len(pk) else cap)
    return np.array(errs)


def style(ax, title, color):
    ax.set_xlim(-.5, W - .5); ax.set_ylim(H - .5, -.5); ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    ax.set_title(title, color=color, fontsize=13, fontweight="bold", loc="left")
    for s in ax.spines.values(): s.set_color(color)
    ax.add_patch(Circle((CEN[0], CEN[1]), R, fill=False, ec="#33415c", ls="--", lw=1))
    ax.text(1, 38.2, "dashed = ego sensing range · outside = blind zone", color="#6b7a99", fontsize=7)


def box(ax, xy, color, fill=False, lw=2, ls="-", alpha=1):
    ax.add_patch(Rectangle((xy[0] - 1.8, xy[1] - 1.8), 3.6, 3.6, fill=fill, fc=color, ec=color, lw=lw, ls=ls, alpha=alpha))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="checkpoints"); ap.add_argument("--out", default="egorefine_simulation.gif")
    ap.add_argument("--frames", type=int, default=28); a = ap.parse_args()
    models = {m: load(a.ckpt, m) for m in ("naive", "traf", "egorefine")}
    pos, vel, valid, seed = pick_scene(a.frames); gen = torch.Generator().manual_seed(7)
    delays = [min(DMAX, int(round(3 + 3 * np.sin(f / 4.0)))) for f in range(a.frames)]     # scripted delay profile 0..6
    hist = {k: [] for k in ("naive", "traf", "egorefine")}; imgs = []
    for f in range(a.frames):
        te, dl = DMAX + f, delays[f]
        ego, col, gt, ghost, vis = frame_inputs(pos, vel, valid, te, dl, gen); gt, ghost = gt.numpy(), ghost.numpy()
        with torch.no_grad():
            out = {k: m(ego, col, return_aux=True) for k, m in models.items()}
        pk = {k: extract_peaks(torch.sigmoid(o[0]))[0] for k, o in out.items()}
        for k in hist: hist[k].append(match_err(pk[k], gt).mean())
        aux = out["egorefine"][1]
        fig = plt.figure(figsize=(15, 7.4), facecolor=BG); gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1], hspace=0.28, wspace=0.08, left=.03, right=.985, top=.9, bottom=.07)
        fig.suptitle(f"AI in action — asynchronous V2V perception   |   frame {f+1}/{a.frames}   |   collaborator delay Δ = {dl} steps", color="white", fontsize=14, fontweight="bold", x=.03, ha="left")
        a1, a2, a3 = (fig.add_subplot(gs[0, i]) for i in range(3))
        style(a1, "INPUT  ·  ego view + delayed message", GHOST); style(a2, "OLD WAY  ·  TraF-Align-style", OLD_C); style(a3, "NEW WAY  ·  EgoRefine-style", NEW_C)
        a1.imshow(np.ma.masked_less(ego[0, 0].numpy(), 0.2), cmap="Blues", alpha=.8, vmin=0, vmax=1.2, extent=(-.5, W - .5, H - .5, -.5))
        for g_, gh in zip(gt, ghost): a1.plot([gh[0], g_[0]], [gh[1], g_[1]], color=GHOST, lw=.8, ls=":")
        for gh in ghost: box(a1, gh, GHOST, ls="--", lw=1.5)                          # delayed collaborator report
        a1.text(1, 1.8, "dashed grey = what the collaborator REPORTED (old)", color=GHOST, fontsize=8)
        for g_ in gt: box(a1, g_, GT_C, lw=1.2, alpha=.8)
        a1.text(1, 4.2, "green = where objects are NOW (ground truth)", color=GT_C, fontsize=8)
        for ax, key, col_ in ((a2, "traf", OLD_C), (a3, "egorefine", NEW_C)):
            for g_ in gt: box(ax, g_, GT_C, lw=1.2, alpha=.8)
            P = pk[key]; P = P[P[:, 2] >= .3]
            for x, y, s in P: box(ax, (x, y), col_, lw=2.4); ax.text(x + 2, y - 2, f"{s:.2f}", color=col_, fontsize=7)
            e = match_err(pk[key], gt); ax.text(1, 38.2 - 2.4, f"mean centre error {e.mean():.2f} cells · hits {(e<=2).sum()}/{len(e)}", color="white", fontsize=9, fontweight="bold")
        # EgoRefine: sampling arrows at detected peaks + gate shading
        gate = aux["gate"][0, 0].numpy(); a3.imshow(gate, cmap="magma", alpha=.12, vmin=0, vmax=1, extent=(-.5, W - .5, H - .5, -.5))
        for x, y, s in pk["egorefine"][pk["egorefine"][:, 2] >= .3]:
            xi, yi = int(round(x)), int(round(y)); o = aux["o_ref"][0, :, :, yi, xi].numpy()
            for k in range(o.shape[0]): a3.annotate("", xy=(x + o[k, 0], y + o[k, 1]), xytext=(x, y), arrowprops=dict(arrowstyle="->", color="#ffd166", lw=1.2))
        a3.text(1, 1.8, "yellow arrows = refined sampling trajectory · shading = reliability gate", color="#ffd166", fontsize=8)
        ax4 = fig.add_subplot(gs[1, :]); ax4.set_facecolor(BG); xs_ = np.arange(1, f + 2)
        ax4.plot(xs_, hist["naive"], color="#8892a6", lw=1.5, label="naive (no alignment)"); ax4.plot(xs_, hist["traf"], color=OLD_C, lw=2.2, label="old way (TraF-Align-style)")
        ax4.plot(xs_, hist["egorefine"], color=NEW_C, lw=2.6, label="EgoRefine-style")
        ax4.bar(np.arange(1, a.frames + 1), np.array(delays) / DMAX * 8, color="#22304d", width=.8, zorder=0, label="delay Δ (scaled)")
        ax4.set_xlim(.5, a.frames + .5); ax4.set_ylim(0, 8.2); ax4.set_ylabel("mean centre error (cells)", color="white", fontsize=9)
        ax4.tick_params(colors="#9fb0d0", labelsize=8); [s.set_color("#33415c") for s in ax4.spines.values()]
        ax4.legend(loc="upper left", ncol=4, fontsize=8, facecolor=BG, labelcolor="white", edgecolor="#33415c")
        ax4.set_title(f"running mean — naive {np.mean(hist['naive']):.2f} · old {np.mean(hist['traf']):.2f} · EgoRefine-style {np.mean(hist['egorefine']):.2f} cells", color="white", fontsize=10, loc="right")
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=72, facecolor=BG); plt.close(fig); buf.seek(0); imgs.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    imgs[0].save(a.out, save_all=True, append_images=imgs[1:], duration=[450] * (len(imgs) - 1) + [2000], loop=0, optimize=True)
    print(f"wrote {a.out}: {len(imgs)} frames, seed {seed}; means naive/old/new = {np.mean(hist['naive']):.2f}/{np.mean(hist['traf']):.2f}/{np.mean(hist['egorefine']):.2f}")


if __name__ == "__main__":
    main()
