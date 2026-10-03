"""Render the 'AI in action' simulation: input history vs ground truth vs three predictions.

Panels:  (1) BEV: history trail (input), expert ground truth, stale-momentum extrapolation (red),
         single-latent baseline (amber), MomWorld-style plan (blue), lead vehicle;
         (2) live centre-distance to lead vs 6 m collision line;
         (3) telemetry: per-step ADE, running ADE, collision banners, reset-gate keep + momentum norm.
Uses the trained checkpoints in outputs/ (run train.py first).
Usage: python simulate.py [--out outputs/momworld_simulation.gif] [--seed 0]
"""
import argparse, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image
from momworld import MomWorld, MomWorldConfig, make_dataset
from momworld.data import index_batch, H, DT, CAR_RADIUS
from momworld.metrics import stale_momentum_extrapolation, collision

BG, FG, GRID = "#0f1419", "#e6edf3", "#2b333d"
COL = dict(gt="#3fb950", stale="#f85149", base="#e3a008", mom="#58a6ff", lead="#bc8cff", hist="#8b949e")


def load(name, seed):
    ck = torch.load(f"outputs/{name}_seed{seed}.pt"); m = MomWorld(MomWorldConfig(**ck["cfg"])); m.load_state_dict(ck["state"]); m.eval(); return m


def pick_scenes(te, preds):
    """one event scene where stale collides & MomWorld does not (largest stale error), one free scene"""
    stale_c = collision(preds["stale"], te.lead_future); mom_c = collision(preds["mom"], te.lead_future)
    ev = torch.where(te.event & stale_c & ~mom_c)[0]
    if len(ev) == 0: ev = torch.where(te.event)[0]
    err = (preds["stale"] - te.future).norm(dim=-1).mean(1)
    a = ev[err[ev].argmax()].item()
    fr = torch.where(~te.event)[0]; b = fr[(preds["base"][fr] - te.future[fr]).norm(dim=-1).mean(1).argmax()].item()
    return [a, b]


def draw(fig, axs, i, k, te, preds, keep, mnorm, label):
    ax, ad, at = axs
    for a in axs: a.clear(); a.set_facecolor(BG)
    hist = te.raw_hist_xy[i].numpy(); lead = te.lead_future[i].numpy(); gt = te.future[i].numpy()
    # ---- BEV ----
    ax.plot(hist[:, 0], hist[:, 1], "-", color=COL["hist"], lw=3, label="input history")
    ax.plot(gt[:, 0], gt[:, 1], ":", color=COL["gt"], lw=2.5, label="ground truth (expert)")
    for key, lab in (("stale", "stale momentum (no scene)"), ("base", "single-latent baseline"), ("mom", "MomWorld-style")):
        p = preds[key][i].numpy(); ax.plot(p[:k + 1, 0], p[:k + 1, 1], "-o", ms=3, color=COL[key], lw=2, label=lab)
    for key in ("stale", "base", "mom"):
        p = preds[key][i].numpy()[max(k - 1, 0)]
        ax.add_patch(Rectangle((p[0] - 2.2, p[1] - 1.0), 4.4, 2.0, fc="none", ec=COL[key], lw=1.5))
    lx, ly = lead[max(k - 1, 0)] if k > 0 else (lead[0][0], lead[0][1])
    ax.add_patch(Rectangle((lx - 2.2, ly - 1.0), 4.4, 2.0, fc=COL["lead"], ec="white", lw=1.2, alpha=0.9)); ax.text(lx, ly + 2.2, "lead", color=COL["lead"], ha="center", fontsize=8)
    pall = np.concatenate([hist, gt] + [preds[k_][i].numpy() for k_ in ("stale", "base", "mom")])
    xr = (pall[:, 0].min() - 4, max(pall[:, 0].max(), lx if lx < pall[:, 0].max() + 25 else 0) + 8)
    yc = pall[:, 1].mean(); yh = max(8.0, 0.3 * (xr[1] - xr[0]))
    ax.set_xlim(*xr); ax.set_ylim(yc - yh, yc + yh)
    if lx > xr[1]: ax.text(xr[1] - 1, yc + yh * 0.75, f"lead vehicle {lx:.0f} m ahead  \u2192", color=COL["lead"], ha="right", fontsize=9)
    ax.legend(loc="lower right", fontsize=7, facecolor=BG, labelcolor=FG, framealpha=0.8)
    ax.set_title(f"BEV  |  {label}  |  t = +{k*DT:.1f}s", color=FG, fontsize=11)
    # ---- distance to lead ----
    t = np.arange(1, H + 1) * DT
    for key in ("gt", "stale", "base", "mom"):
        p = (te.future if key == "gt" else preds[key])[i].numpy()
        d = np.linalg.norm(p - lead, axis=1)
        ad.plot(t[:max(k, 1)], d[:max(k, 1)], color=COL[key], lw=2, label={"gt": "expert", "stale": "stale", "base": "baseline", "mom": "MomWorld"}[key])
    ad.axhline(2 * CAR_RADIUS, color="#f85149", ls="--", lw=1); ad.text(0.2, 2 * CAR_RADIUS + 0.5, "collision < 6 m", color="#f85149", fontsize=8)
    ad.set_xlim(0, H * DT); ad.set_ylim(0, max(70, np.linalg.norm(lead - gt, axis=1).max() * 1.2)); ad.set_title("centre distance to lead vehicle (m)", color=FG, fontsize=11)
    ad.legend(fontsize=7, facecolor=BG, labelcolor=FG, loc="upper right")
    # ---- telemetry ----
    at.axis("off"); y = 0.96
    at.text(0, y, "LIVE TELEMETRY", color=FG, fontsize=11, weight="bold", transform=at.transAxes); y -= 0.09
    for key, name in (("stale", "Stale momentum"), ("base", "Single-latent"), ("mom", "MomWorld-style")):
        p = preds[key][i].numpy(); e = np.linalg.norm(p - gt, axis=1)
        run = e[:max(k, 1)].mean(); dmin = np.linalg.norm(p - lead, axis=1)[:max(k, 1)].min()
        crash = dmin < 2 * CAR_RADIUS
        at.text(0, y, f"{name:16s} ADE {run:5.2f} m   min-dist {dmin:5.1f} m", color=COL[key], fontsize=9, family="monospace", transform=at.transAxes)
        if crash: at.text(0.0, y - 0.065, "  >> COLLISION <<", color="white", fontsize=9, weight="bold", family="monospace", transform=at.transAxes, bbox=dict(fc="#b62324", ec="none", pad=2))
        y -= 0.15
    at.text(0, y, "MomWorld internals (mean over latent dims)", color=FG, fontsize=9, transform=at.transAxes); y -= 0.06
    kk = keep[i].mean(-1).numpy(); mm = mnorm[i].numpy()
    ia = at.inset_axes([0.0, 0.0, 1.0, y - 0.04]); ia.set_facecolor(BG)
    ia.plot(np.arange(1, H + 1) * DT, kk, color="#58a6ff", lw=2, label="reset-gate keep")
    ia.plot(np.arange(1, H + 1) * DT, mm / max(mm.max(), 1e-6), color="#e3a008", lw=2, label="|momentum| (norm.)")
    ia.axvline(max(k, 1) * DT, color="white", lw=0.8)
    ia.set_ylim(0, 1.05); ia.legend(fontsize=7, facecolor=BG, labelcolor=FG, loc="lower left")
    ia.tick_params(colors=FG, labelsize=7); [s.set_color(GRID) for s in ia.spines.values()]
    for a in (ax, ad):
        a.tick_params(colors=FG, labelsize=8); [s.set_color(GRID) for s in a.spines.values()]; a.grid(color=GRID, lw=0.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs/momworld_simulation.gif"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    te = make_dataset(600, 300)
    mom, base = load("momworld", a.seed), load("baseline", a.seed)
    with torch.no_grad():
        om, ob = mom(te.hist), base(te.hist)
    preds = dict(stale=stale_momentum_extrapolation(te.hist), base=ob["plan"], mom=om["plan"])
    keep, mnorm = om["keep"], om["m"].norm(dim=-1)
    scenes = pick_scenes(te, preds)
    labels = ["EVENT scene: lead vehicle ahead, expert must brake", "FREE scene: trend persists, no event"]
    fig = plt.figure(figsize=(15, 5.4), dpi=72, facecolor=BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.05], left=0.04, right=0.98, top=0.9, bottom=0.1, wspace=0.18)
    axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
    frames, durs = [], []
    for i, lab in zip(scenes, labels):
        for k in list(range(0, H + 1)) + [H] * 4:
            draw(fig, axs, i, k, te, preds, keep, mnorm, lab); fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())); durs.append(450 if k < H else 700)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=False)
    st = json.load(open("outputs/results.json"))["stale_momentum"]
    print(f"wrote {a.out}: {len(frames)} frames; scenes {scenes}; stale collision (event) {st['collision_event']:.2f}")

if __name__ == "__main__":
    main()
