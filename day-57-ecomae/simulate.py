"""Synthetic 'AI in action' simulation: a pedestrian approaches a kerb in FOG; the sensor drops steps and is noisy.
python simulate.py --cls left --out assets/ecomae_demo   (needs checkpoints/{sup,ecomae}.pt from train.py; writes .gif + .mp4 + .json)

Left  : top-down scene. INPUT = observed track (filled dots; hollow grey = dropped sensor steps).  GROUND TRUTH = green future path.
        PREDICTION = arrows from the pedestrian, length ~ predicted intent probability: blue = eCoMAE-style, orange = supervised-only.
Right : intent probabilities, eCoMAE epistemic uncertainty (from the NIG head on the observed window), probability on the TRUE intent.
"""
import argparse, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from ecomae.synthetic import track, observe, L, CLASSES
from ecomae.model import eCoMAE

ap = argparse.ArgumentParser(); ap.add_argument("--cls", default="left", choices=CLASSES); ap.add_argument("--env", default="fog")
ap.add_argument("--seed", type=int, default=5); ap.add_argument("--out", default="assets/ecomae_demo"); a = ap.parse_args()
cls, T, S0 = CLASSES.index(a.cls), 70, 46
rng = np.random.default_rng(a.seed)
feats, pos = track(cls, T, S0, a.env, rng)
X, V = observe(feats, a.env, rng)
fut_pos = track(cls, T + 8, S0, a.env, np.random.default_rng(a.seed))[1]             # same params (same rng stream), longer horizon

mods = {}
for k in ("sup", "ecomae"):
    m = eCoMAE(); m.load_state_dict(torch.load(f"checkpoints/{k}.pt")); mods[k] = m.eval()
P = {k: [] for k in mods}; EPI = []
with torch.no_grad():
    for t in range(T):
        lo = t + 1 - L; x = np.zeros((L, 5), np.float32); v = np.zeros(L, bool)
        src = slice(max(lo, 0), t + 1); n = t + 1 - max(lo, 0)
        x[L - n:] = X[src]; v[L - n:] = V[src]                                           # left-pad short histories
        xt, vt = torch.tensor(x)[None], torch.tensor(v)[None]
        if not vt.any(): vt[0, -1] = True
        for k, m in mods.items(): P[k].append(torch.softmax(m.classify(xt, vt), -1)[0].numpy())
        tok = mods["ecomae"].encode(xt, vt); o = mods["ecomae"].decode(tok, vt, vt)       # NIG head on the observed window
        _, _, epi = eCoMAE.variances(o); EPI.append(float(epi[0][vt[0]].mean()))
P = {k: np.array(p) for k, p in P.items()}; EPI = np.array(EPI)

fig = plt.figure(figsize=(12.4, 6.4), dpi=100)
gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.1], left=.05, right=.98, top=.83, bottom=.08, hspace=.7, wspace=.2)
axs, axp, axu, axc = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 1])
fig.text(.05, .94, f"AI in action: pedestrian intent in {a.env.upper()} - sensor drops steps, an evidential MAE encoder reads the rest (synthetic)", fontsize=12, weight="bold")
sub = fig.text(.05, .885, "", fontsize=9, family="monospace")
fig.text(.98, .002, "Synthetic demo - eCoMAE-style re-implementation (arXiv 2610.39958), not IMPTC/Waymo/nuScenes/inD data", ha="right", fontsize=8, color="#555")
DIRS = {0: (0, 1), 1: (-1, 0), 2: (1, 0), 3: (0, -1)}                                    # screen direction of each intent
ymin, ymax = pos[:, 1].min() - 3, fut_pos[:, 1].max() + 2


def draw(f):
    t = min(f, T - 1)
    axs.cla(); axs.add_patch(plt.Rectangle((-30, ymax - 4), 60, 1.5, color="#bbb", zorder=0)); axs.text(-8.7, ymax - 3.4, "kerb / crossing", fontsize=7, color="#555")
    axs.plot(fut_pos[:, 0], fut_pos[:, 1], color="#1a9850", lw=3, alpha=.7, label="GROUND TRUTH (future path)", zorder=2)
    ok = V[:t + 1]
    axs.scatter(pos[:t + 1][ok, 0], pos[:t + 1][ok, 1], s=14, c="k", label="INPUT observed step", zorder=4)
    axs.scatter(pos[:t + 1][~ok, 0], pos[:t + 1][~ok, 1], s=22, facecolors="none", edgecolors="#999", label="dropped step", zorder=3)
    cx, cy = pos[t]
    for k, col, off in (("sup", "#f28e2b", -.18), ("ecomae", "#1f5fd1", .18)):
        for c in range(4):
            dx, dy = DIRS[c]; ln = 5 * P[k][t][c]
            if ln > .2: axs.annotate("", (cx + dx * ln + off * dy * 4, cy + dy * ln + off * dx * 4), (cx + off * dy * 4, cy + off * dx * 4), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5, alpha=.9), zorder=6)
    axs.plot([], [], color="#1f5fd1", lw=3, label="PREDICTION eCoMAE-style"); axs.plot([], [], color="#f28e2b", lw=3, label="PREDICTION supervised-only")
    axs.set_xlim(-9, 9); axs.set_ylim(ymin, ymax); axs.legend(fontsize=7, loc="upper right", framealpha=.9); axs.tick_params(labelsize=7)
    axs.set_title("top-down: input -> ground truth -> predicted intent (arrow length ~ probability)", fontsize=8.5)
    axp.cla(); w = .38; xs = np.arange(4)
    axp.bar(xs - w / 2, P["sup"][t], w, color="#f28e2b", label="supervised-only"); axp.bar(xs + w / 2, P["ecomae"][t], w, color="#1f5fd1", label="eCoMAE-style")
    axp.set_xticks(xs); axp.set_xticklabels(CLASSES, fontsize=8); axp.set_ylim(0, 1.05); axp.axvline(cls, color="#1a9850", lw=8, alpha=.15); axp.legend(fontsize=7); axp.set_title("intent probabilities (green band = true intent)", fontsize=8.5)
    tt = np.arange(t + 1)
    axu.cla(); axu.plot(tt, EPI[:t + 1], color="#8e44ad", lw=2); axu.axvline(S0, color="#1a9850", ls="--", lw=1); axu.set_xlim(0, T); axu.set_ylim(0, max(EPI.max() * 1.1, 1e-3)); axu.grid(alpha=.3)
    axu.set_title("eCoMAE epistemic uncertainty (NIG head, mean over observed steps); dashed = manoeuvre onset", fontsize=8.5); axu.tick_params(labelsize=7)
    axc.cla()
    for k, col in (("sup", "#f28e2b"), ("ecomae", "#1f5fd1")): axc.plot(tt, P[k][:t + 1, cls], color=col, lw=2.2, label=k)
    axc.axvline(S0, color="#1a9850", ls="--", lw=1); axc.set_xlim(0, T); axc.set_ylim(0, 1.02); axc.grid(alpha=.3); axc.legend(fontsize=7, loc="upper left"); axc.tick_params(labelsize=7)
    axc.set_xlabel("frame (0.1 s)", fontsize=8); axc.set_title(f"probability assigned to the TRUE intent ({a.cls})", fontsize=8.5)
    sub.set_text(f"frame {t:02d}  dropped so far {int((~ok).sum()):02d}  sup -> {CLASSES[P['sup'][t].argmax()]:<8s} ({P['sup'][t].max():.2f})   eCoMAE -> {CLASSES[P['ecomae'][t].argmax()]:<8s} ({P['ecomae'][t].max():.2f})   true: {a.cls}")
    return []


anim = FuncAnimation(fig, draw, frames=T + 5, interval=140, blit=False)
anim.save(a.out + ".gif", writer=PillowWriter(fps=7)); anim.save(a.out + ".mp4", writer=FFMpegWriter(fps=7, bitrate=2400))
fin = {k: dict(final_pred=CLASSES[int(P[k][-1].argmax())], p_true_final=round(float(P[k][-1][cls]), 3), frames_correct=int((P[k][S0 + 8:].argmax(1) == cls).sum()), of=int(len(P[k][S0 + 8:]))) for k in P}
json.dump(dict(cls=a.cls, env=a.env, seed=a.seed, summary=fin), open(a.out + ".json", "w"), indent=1); print(fin)
