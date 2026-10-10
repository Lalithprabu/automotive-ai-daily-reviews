"""Real-time-style synthetic simulation overlay: input vs ground truth vs model prediction.

Streams held-out OCCLUDED-regime scenarios (the hard, shifted case). Each scenario = 2 frames:
  1) plan submitted: ego candidate trajectory + other agent's observed state; predictions shown.
  2) outcome: ground-truth reaction of the other agent, collision marker, verdicts scored.
Right-hand telemetry: running unsafe-recall / false-alarm for argmax-VLM-stand-in vs SLLCP,
and the live prediction SET {safe, unsafe} for the current scene.
Output: assets/sllcp_simulation.gif (all numbers come from the real trained checkpoint).
"""
import argparse, io, yaml, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sllcp.data import make_dataset, rollout, simulate_collisions
from sllcp.models import SafetyScorer
from evaluate import evaluate

BG, FG, GRID = "#0f1419", "#e6edf3", "#2b3440"
C_EGO, C_OTH, C_GT, C_OLD, C_NEW = "#58a6ff", "#f0b429", "#3fb950", "#8b949e", "#ff7b72"


def main(cfg_path="config.yaml", n_scen=22, out="assets/sllcp_simulation.gif"):
    cfg = yaml.safe_load(open(cfg_path)); torch.manual_seed(cfg["seed"])
    model = SafetyScorer(); model.load_state_dict(torch.load("checkpoints/scorer.pt")); model.eval()
    cal = make_dataset(cfg["n_cal"], cfg["cal_mix"], cfg["seed"] + 2)
    te = make_dataset(6000, [0.0, 0.0, 1.0], cfg["seed"] + 9)  # occluded-only stream
    res, flags, sets, (glob, loc) = evaluate(model, cal, te, cfg["alpha"], cfg["bandwidth"])
    xe, yo = rollout(te["params"]); y = te["y"]
    pu = model.probs(te["x"])[:, 1]
    # stream: first n_scen scenarios, but guarantee a realistic share of unsafe ones (every 4th is unsafe)
    unsafe_idx = (y == 1).nonzero().flatten().tolist(); safe_idx = (y == 0).nonzero().flatten().tolist()
    order, ui, si = [], 0, 0
    for k in range(n_scen):
        if k % 4 == 1: order.append(unsafe_idx[ui]); ui += 1
        else: order.append(safe_idx[si]); si += 1
    tally = {"old": [0, 0, 0, 0], "new": [0, 0, 0, 0]}  # unsafe hit, unsafe n, false alarm, safe n
    hist = {"old": [], "new": []}
    frames = []
    for k, i in enumerate(order):
        old, new = bool(flags["argmax (old way)"][i]), bool(flags["SLLCP (new way)"][i])
        for stage in (0, 1):
            fig = plt.figure(figsize=(13, 5.6), facecolor=BG)
            ax = fig.add_axes([0.04, 0.10, 0.44, 0.80], facecolor=BG)
            ax2 = fig.add_axes([0.57, 0.52, 0.40, 0.36], facecolor=BG)
            ax3 = fig.add_axes([0.57, 0.10, 0.40, 0.30], facecolor=BG)
            ex, oy = xe[i].numpy(), yo[i].numpy()
            ax.axhline(0, color=GRID, lw=18, alpha=.6); ax.axvline(0, color=GRID, lw=18, alpha=.6)
            ax.plot(ex, np.zeros_like(ex), color=C_EGO, lw=2, ls="--", alpha=.8, label="ego plan (input)")
            if stage == 0:
                ax.scatter([ex[0]], [0], c=C_EGO, s=140, marker="s", zorder=5)
                ax.scatter([0], [oy[0]], c=C_OTH, s=140, marker="s", zorder=5)
                ax.text(0.02, 0.97, "STAGE 1/2  plan submitted - predictions pending outcome", transform=ax.transAxes,
                        color=FG, va="top", fontsize=9)
            else:
                ax.plot(np.zeros_like(oy), oy, color=C_GT, lw=2.5, label="other agent GT reaction")
                tt = int(np.argmin(np.hypot(ex, oy))); hit = bool(y[i])
                ax.scatter([ex[tt]], [0], c=C_EGO, s=140, marker="s", zorder=5)
                ax.scatter([0], [oy[tt]], c=C_OTH, s=140, marker="s", zorder=5)
                if hit:
                    ax.scatter([0], [0], s=900, facecolors="none", edgecolors=C_NEW, lw=3, zorder=6)
                    ax.text(0, 4, "COLLISION", color=C_NEW, ha="center", fontsize=11, weight="bold")
                ax.text(0.02, 0.97, f"STAGE 2/2  ground truth: {'UNSAFE (collision)' if hit else 'safe'}",
                        transform=ax.transAxes, color=C_GT if not hit else C_NEW, va="top", fontsize=9)
            lim = max(abs(ex).max(), abs(oy).max()) + 8
            ax.set_xlim(-lim, 25); ax.set_ylim(-lim, 25); ax.set_aspect("equal")
            ax.set_title(f"Scene {k+1}/{n_scen} - OCCLUDED regime  (BEV, conflict point at origin)", color=FG, fontsize=10)
            ax.legend(loc="lower right", facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
            ax.tick_params(colors=FG, labelsize=7)
            for sp in ax.spines.values(): sp.set_color(GRID)
            # prediction panel
            ax2.axis("off")
            sset = [n for n, b in zip(["safe", "unsafe"], sets[i].tolist()) if b]
            ax2.text(0, 0.98, "MODEL PREDICTION OVERLAY", color=FG, fontsize=10, weight="bold", va="top", transform=ax2.transAxes)
            ax2.text(0, 0.78, f"P(unsafe) from predictor: {pu[i]:.2f}", color=FG, fontsize=10, transform=ax2.transAxes)
            ax2.text(0, 0.56, f"Old way (argmax):  {'FLAG UNSAFE' if old else 'says safe'}", color=C_OLD if not old else C_NEW,
                     fontsize=11, transform=ax2.transAxes)
            ax2.text(0, 0.34, f"SLLCP set: {{{', '.join(sset)}}}  ->  {'FLAG UNSAFE' if new else 'says safe'}",
                     color=C_NEW if new else C_OLD, fontsize=11, transform=ax2.transAxes)
            if stage == 1:
                v = lambda f: "correct" if f == bool(y[i]) else ("MISSED collision" if y[i] else "false alarm")
                ax2.text(0, 0.10, f"verdict: old = {v(old)}   |   SLLCP = {v(new)}", color=FG, fontsize=9, transform=ax2.transAxes)
                if stage == 1 and not (k * 2 + stage) % 2 == 0:
                    pass
            # telemetry (updated after outcome of each scene)
            if stage == 1:
                for key, f in (("old", old), ("new", new)):
                    t = tally[key]
                    if y[i]: t[1] += 1; t[0] += int(f)
                    else: t[3] += 1; t[2] += int(f)
                    hist[key].append(t[0] / max(t[1], 1))
            xs = np.arange(1, len(hist["old"]) + 1)
            ax3.set_facecolor(BG)
            if len(xs): ax3.plot(xs, hist["old"], color=C_OLD, lw=2, label="argmax"); ax3.plot(xs, hist["new"], color=C_NEW, lw=2, label="SLLCP")
            ax3.axhline(1 - cfg["alpha"], color=C_GT, ls=":", lw=1, label="target 90%")
            ax3.set_xlim(1, n_scen); ax3.set_ylim(-0.05, 1.05)
            ax3.set_title("running unsafe-trajectory recall (telemetry)", color=FG, fontsize=9)
            ax3.tick_params(colors=FG, labelsize=7); ax3.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=7, loc="center right")
            for sp in ax3.spines.values(): sp.set_color(GRID)
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=80, facecolor=BG); plt.close(fig)
            buf.seek(0); frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=900, loop=0)
    print(f"wrote {out}: {len(frames)} frames; stream recall old={tally['old'][0]}/{tally['old'][1]} new={tally['new'][0]}/{tally['new'][1]}; "
          f"false alarms old={tally['old'][2]}/{tally['old'][3]} new={tally['new'][2]}/{tally['new'][3]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--scenes", type=int, default=22)
    a = ap.parse_args(); main(a.config, a.scenes)
