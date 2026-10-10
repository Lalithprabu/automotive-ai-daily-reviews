"""Synthetic 'AI in action' simulation: raster-only baseline vs. CAA-GMM on the same junction-approach episode.

Left BEV  = old way (raster CNN + MLP).  Right BEV = CAA-GMM.  Green = ground truth, white = observed history,
coloured fan = K Gaussian modes (opacity ~ mixture weight, ellipse = 2-sigma of the final step).
Segment A: full V2X/perception.  Segment B: communication outage -- the lead vehicle's message is lost.
Bottom: running minADE@6 telemetry + CAA-GMM attention over neighbours + the 3-way [motion|interaction|context] gate.

    python simulate.py --out assets/simulation.gif [--seed 2026]
Everything drawn comes from the real trained checkpoints in ./checkpoints (run train.py first).
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from caa_gmm.data import (BRANCHES, J_MIN, T_F, T_H, X0, Y0, GRID, build_raster, perturb_neighbours,  # noqa: E402
                          rollout, sample_to_arrays, static_tables)
from caa_gmm.model import CAAConfig, CAAGMM, RasterBaseline  # noqa: E402

BG, FG, GRID_C = "#0e1117", "#e6edf3", "#30363d"
C_OLD, C_NEW, C_GT, C_ASSET = "#f0883e", "#58a6ff", "#3fb950", "#8b949e"


def load(name, cls):
    ck = torch.load(f"checkpoints/{name}.pt", map_location="cpu")
    m = cls(CAAConfig(**ck["cfg"]))
    m.load_state_dict(ck["state"])
    return m.eval()


def pick_episode(seed):
    """Non-outcome selection rule: lead vehicle, >=1 distractor, a turning branch, >=5 usable windows."""
    rng = np.random.default_rng(seed)
    while True:
        J = 18
        hint = int(rng.integers(0, 3))
        branch = int(rng.choice([0, 2]))
        r = rollout(rng, J, branch, hint, n_future=T_F + 10, force_lead=True)
        if not r["dis"]:
            continue
        idx = [i for i in range(T_H - 1, len(r["s"]) - T_F) if r["Jg"] - r["s"][i] >= J_MIN]
        if len(idx) >= 5:
            return r, idx[:6], rng


def to_batch(a, mask=None):
    b = {k: torch.as_tensor(np.asarray(v))[None] for k, v in a.items() if k not in ("has_lead",)}
    b["has_lead"] = torch.tensor([bool(a["has_lead"])])
    if mask is not None:
        b["mask"] = mask[None]
    return b


@torch.no_grad()
def predict(model, b):
    out = model(build_raster(b), b["ego"], b["nb"], b["mask"], return_attn=True)
    lo, mu, ls, att = out
    return torch.softmax(lo, 1)[0].numpy(), mu[0].numpy(), ls[0].numpy(), att


def veh(ax, x, y, color, ls="-", fill=True, alpha=0.9):
    ax.add_patch(Rectangle((x - 1, y - 2.25), 2, 4.5, fc=color if fill else "none", ec=color, lw=1.5,
                           ls=ls, alpha=alpha, zorder=5))


def draw_bev(ax, b, a, pi, mu, ls, gt, color, title, fd_lead, ade):
    ax.set_facecolor(BG)
    drv, hint, _ = static_tables()
    ji = int(b["J"][0]) - 4
    ext = [X0, X0 + GRID, Y0, Y0 + GRID]
    ax.imshow(drv[ji].numpy(), origin="lower", extent=ext, cmap=matplotlib.colors.ListedColormap([BG, "#2a313c"]), vmin=0, vmax=1, zorder=0)
    ax.imshow(hint[ji, int(b["hint"][0])].numpy(), origin="lower", extent=ext,
              cmap=matplotlib.colors.ListedColormap([(0, 0, 0, 0), (0.85, 0.7, 0.1, 0.30)]), vmin=0, vmax=1, zorder=1)
    lead = int(fd_lead[0])
    for n in range(b["nb"].shape[1]):                             # neighbours (true positions)
        if not bool(a["mask"][n]):
            continue
        p = a["nb"][n]
        if n == lead and not bool(b["mask"][0, n]):
            veh(ax, p[-1, 0], p[-1, 1], "#f85149", ls="--", fill=False)
            ax.text(p[-1, 0] + 1.6, p[-1, 1], "LEAD\nMSG LOST", color="#f85149", fontsize=7, va="center", zorder=6)
        elif n == lead:
            veh(ax, p[-1, 0], p[-1, 1], "#f0883e")
            ax.text(p[-1, 0] + 1.6, p[-1, 1], "lead", color=FG, fontsize=7, va="center", zorder=6)
        else:
            veh(ax, p[-1, 0], p[-1, 1], C_ASSET)
    veh(ax, 0, 0, FG)
    h = a["ego"][:, :2]
    ax.plot(h[:, 0], h[:, 1], "-", color=FG, lw=1.5, alpha=0.8, zorder=4)
    ax.plot(gt[:, 0], gt[:, 1], "-", color=C_GT, lw=3, zorder=6, label="ground truth")
    order = np.argsort(pi)
    for k in order:
        al = 0.15 + 0.85 * float(pi[k]) ** 0.6
        ax.plot(np.r_[0, mu[k, :, 0]], np.r_[0, mu[k, :, 1]], "-", color=color, lw=1 + 3 * pi[k], alpha=al, zorder=7)
        if pi[k] > 0.06:
            s = np.exp(ls[k, -1])
            ax.add_patch(Ellipse(mu[k, -1], 4 * s[0], 4 * s[1], fc=color, ec=color, alpha=0.25 * al + 0.05, zorder=3))
            ax.text(mu[k, -1, 0], mu[k, -1, 1] + 1.0, f"{pi[k]:.2f}", color=FG, fontsize=7, ha="center", zorder=8)
    ax.set_xlim(-22, 22)
    ax.set_ylim(-9, 50)
    ax.set_aspect("equal")
    ax.set_title(f"{title}   minADE@6 = {ade:.2f} m", color=FG, fontsize=11, loc="left")
    ax.tick_params(colors=C_ASSET, labelsize=7)
    for s in ax.spines.values():
        s.set_color(GRID_C)


def render(frames_data, i, fig_size=(14, 7.6)):
    fd = frames_data[i]
    fig = plt.figure(figsize=fig_size, dpi=100, facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1], hspace=0.28, wspace=0.18, left=0.04, right=0.98,
                          top=0.9, bottom=0.07)
    axo, axn = fig.add_subplot(gs[0, 0:1]), fig.add_subplot(gs[0, 1:2])
    axl = fig.add_subplot(gs[0, 2])
    axt, axa, axg = fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2]), None
    gt = fd["gt"]
    draw_bev(axo, fd["b"], fd["a"], *fd["old"][:3], gt, C_OLD, "OLD WAY: raster CNN + MLP", (fd["lead_slot"],), fd["old_ade"])
    draw_bev(axn, fd["b"], fd["a"], *fd["new"][:3], gt, C_NEW, "NEW: CAA-GMM", (fd["lead_slot"],), fd["new_ade"])
    # legend / banner panel
    axl.axis("off")
    seg = "SEGMENT B  -  V2X OUTAGE (lead msg lost)" if fd["outage"] else "SEGMENT A  -  full V2X + perception"
    axl.text(0, 1.0, seg, color="#f85149" if fd["outage"] else C_GT, fontsize=12, weight="bold", va="top")
    lines = ["green = ground truth", "white = observed history", "fan = 6 Gaussian modes", "opacity/label = mode weight",
             "ellipse = 2-sigma @ t+4s", "yellow lane = map route hint", "grey boxes = other vehicles", "",
             f"junction in {fd['jrel']:.0f} m", f"episode step {fd['step']+1}/{fd['nsteps']}"]
    for j, t in enumerate(lines):
        axl.text(0, 0.86 - 0.075 * j, t, color=FG, fontsize=9, va="top")
    # telemetry: running minADE
    axt.set_facecolor(BG)
    xs = np.arange(i + 1)
    axt.plot(xs, [frames_data[j]["old_ade"] for j in xs], "-o", color=C_OLD, ms=3, label="old way")
    axt.plot(xs, [frames_data[j]["new_ade"] for j in xs], "-o", color=C_NEW, ms=3, label="CAA-GMM")
    nA = sum(1 for f in frames_data if not f["outage"])
    axt.axvspan(-0.5, nA - 0.5, color=C_GT, alpha=0.07)
    axt.axvspan(nA - 0.5, len(frames_data) - 0.5, color="#f85149", alpha=0.07)
    axt.set_xlim(-0.5, len(frames_data) - 0.5)
    axt.set_ylim(0, max(2.0, max(max(f["old_ade"], f["new_ade"]) for f in frames_data) * 1.15))
    axt.set_title("telemetry: minADE@6 (m) per prediction step  [A: full comms | B: outage]", color=FG, fontsize=9, loc="left")
    axt.legend(facecolor=BG, labelcolor=FG, fontsize=8, loc="upper left", edgecolor=GRID_C)
    axt.tick_params(colors=C_ASSET, labelsize=7)
    for s in axt.spines.values():
        s.set_color(GRID_C)
    # attention + gate
    axa.set_facecolor(BG)
    att = fd["new"][3]
    w = att["w_int"][0].numpy()
    g = att["gate"][0].numpy()
    labels, vals, cols = [], [], []
    labels.append("null"), vals.append(float(att["w_null"][0])), cols.append(C_ASSET)
    for n in range(len(w)):
        if bool(fd["b"]["mask"][0, n]):
            labels.append("LEAD" if n == fd["lead_slot"] else f"veh{n}")
            vals.append(float(w[n]))
            cols.append("#f0883e" if n == fd["lead_slot"] else C_ASSET)
    labels += ["motion", "interact.", "context"]
    vals += list(map(float, g))
    cols += [C_NEW, C_NEW, C_NEW]
    yy = np.arange(len(vals))[::-1]
    axa.barh(yy, vals, color=cols)
    axa.set_yticks(yy)
    axa.set_yticklabels(labels, color=FG, fontsize=7)
    axa.set_xlim(0, 1)
    axa.set_title("CAA-GMM attention over neighbours + 3-way gate", color=FG, fontsize=9, loc="left")
    axa.tick_params(colors=C_ASSET, labelsize=7)
    for s in axa.spines.values():
        s.set_color(GRID_C)
    fig.suptitle("Day 35 - CAA-GMM (arXiv:2610.09174) reconstruction | synthetic junction-approach data, not nuScenes/AV2",
                 color=FG, fontsize=11, x=0.04, ha="left", y=0.97)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/simulation.gif")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tag", default="", help="checkpoint suffix, e.g. _aug")
    a = ap.parse_args()
    torch.set_num_threads(2)
    new, old = load("caagmm" + a.tag, CAAGMM), load("baseline" + a.tag, RasterBaseline)
    r, idxs, rng = pick_episode(a.seed)
    frames = []
    for outage in (False, True):
        for si, i_now in enumerate(idxs):
            arr = sample_to_arrays(r, i_now, np.random.default_rng(1000 + si))   # same noise in both segments
            b = to_batch(arr)
            lost = int(arr["lead_slot"])
            if outage:
                m = b["mask"].clone()
                m[0, lost] = False
                b["mask"] = m
            po, pn = predict(old, b), predict(new, b)
            gt = arr["fut"]

            def ade(p):
                return float(np.linalg.norm(p[1] - gt[None], axis=-1).mean(-1).min())
            frames.append(dict(b=b, a=arr, gt=gt, old=po, new=pn, old_ade=ade(po), new_ade=ade(pn), lost=lost if outage else -1,
                               lead_slot=lost, outage=outage, jrel=float(r["Jg"] - r["s"][i_now]), step=si,
                               nsteps=len(idxs)))
    # the BEV panel draws the 'lost' marker only in outage frames; in segment A the lead is labelled normally
    imgs = []
    for i in range(len(frames)):
        im = render(frames, i)
        imgs += [im, im]                                                         # hold each step
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    imgs[0].save(a.out, save_all=True, append_images=imgs[1:], duration=450, loop=0, optimize=True)
    last = imgs[-1]
    last.save(a.out.replace(".gif", "_last.png"))
    na = [f["new_ade"] for f in frames]
    oa = [f["old_ade"] for f in frames]
    h = len(idxs)
    print(f"episode: branch={BRANCHES[r['branch']]} hint={BRANCHES[r['hint']]} steps={h}")
    print(f"segment A mean minADE old {np.mean(oa[:h]):.3f} new {np.mean(na[:h]):.3f}")
    print(f"segment B mean minADE old {np.mean(oa[h:]):.3f} new {np.mean(na[h:]):.3f}")
    print(f"wrote {a.out} ({len(imgs)} frames)")


if __name__ == "__main__":
    main()
