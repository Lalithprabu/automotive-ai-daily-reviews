"""'AI in action' closed-loop synthetic simulation:  python simulate.py --seed 3 --out assets/rfworld_demo
Needs checkpoints/rfworld.pt (python train.py). Writes .gif + .mp4 + .json.

Left  INPUT (what the model sees: BEV camera-like render) + PREDICTION (all 256 anchors coloured by predicted PDMS,
      blue = model's pick, dashed) + GROUND TRUTH (green = best anchor under the privileged scorer, red rings = where
      the objects really are at +1.5 s / +3 s). Right: predicted-vs-true score of all anchors, telemetry, score error.
"""
import argparse, json, warnings
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Circle, Polygon
from rfworld.world import World, ANCHORS, XMAX, YMAX, R_OBJ, DT
from rfworld.model import RFWorldToy
from rfworld.evaluate import to_t

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=3); ap.add_argument("--steps", type=int, default=16)
ap.add_argument("--out", default="assets/rfworld_demo"); a = ap.parse_args(); torch.set_num_threads(2)
m = RFWorldToy(); m.load_state_dict(torch.load("checkpoints/rfworld.pt")); m.eval()
At = torch.as_tensor(ANCHORS)[None]

# pick a seed whose scene has a nearby vehicle in the lane (so the pick visibly matters); not chosen for model success
s = a.seed
while True:
    w = World(np.random.default_rng(s)); sc = w.score()
    if (sc["nc"] == 0).mean() > 0.12 and (sc["dac"] == 0).mean() < 0.6: break
    s += 1
rec = []
with torch.no_grad():
    for t in range(a.steps):
        o = w.observe(); out = m(to_t({k: v[None] for k, v in o.items()}), At, with_aux=True)
        # with_aux only for visualising the training-time head; selection uses ONLY the sparse readout scores
        pr = out["pdms"][0].numpy(); tr = w.score(); i = int(pr.argmax()); j = int(tr["pdms"].argmax())
        fut = torch.sigmoid(out["fut"][0]).numpy()
        r = dict(cam=o["cam"].transpose(1, 2, 0).copy(), pred=pr, true=tr["pdms"], i=i, j=j, v=w.v,
                 gap=float(np.min(np.linalg.norm(w.to_ego(w.obj_p), axis=1))), fut=fut,
                 obj15=w.to_ego(w.obj_at(1.5)), obj30=w.to_ego(w.obj_at(3.0)), obj0=w.to_ego(w.obj_p),
                 err=float(np.abs(pr - tr["pdms"]).mean()), regret=float(tr["pdms"][j] - tr["pdms"][i]))
        r["col"] = w.step(ANCHORS[i]); rec.append(r)
T = len(rec); fig = plt.figure(figsize=(12.4, 6.4), dpi=100)
gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1], left=.04, right=.985, top=.84, bottom=.08, hspace=.85, wspace=.2)
axm, axs, axt, axe = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 1])
fig.text(.04, .94, "AI in action: RFWorld-style rollout-free scorer ranks 256 candidate trajectories in ONE forward pass (synthetic)", fontsize=11.5, weight="bold")
sub = fig.text(.04, .895, "", fontsize=9, family="monospace")
fig.text(.985, .002, f"Synthetic demo - toy re-implementation on my own BEV world (seed {s}), NOT NAVSIM, NOT the paper's results", ha="right", fontsize=7.5, color="#555")
cmap = plt.cm.viridis


def draw(fi):
    t = min(fi, T - 1); r = rec[t]
    axm.cla(); axm.imshow(np.clip(r["cam"], 0, 1), extent=[YMAX, -YMAX, 0, XMAX], interpolation="nearest", aspect="equal")
    for k in np.argsort(r["pred"]):                                                  # PREDICTION: all anchors, colour = predicted PDMS
        p = ANCHORS[k]; axm.plot(np.r_[0, p[:, 1]], np.r_[0, p[:, 0]], color=cmap(r["pred"][k]), lw=.7, alpha=.35 if k != r["i"] else 1)
    jj, ii = r["j"], r["i"]
    axm.plot(np.r_[0, ANCHORS[jj][:, 1]], np.r_[0, ANCHORS[jj][:, 0]], c="#2ecc71", lw=3, label="GROUND TRUTH best anchor")
    axm.plot(np.r_[0, ANCHORS[ii][:, 1]], np.r_[0, ANCHORS[ii][:, 0]], c="#1f77ff", lw=2.2, ls="--", marker="o", ms=3, label="PREDICTION: model's pick")
    for P, ls, lab in ((r["obj15"], ":", "objects +1.5 s (truth)"), (r["obj30"], "-", "objects +3.0 s (truth)")):
        for q in P:
            if 0 < q[0] < XMAX and abs(q[1]) < YMAX: axm.add_patch(Circle((q[1], q[0]), R_OBJ, fill=False, ec="#ff4d4d", lw=1.3, ls=ls))
    axm.plot([], [], c="#ff4d4d", lw=1.3, label="objects +1.5 s (dotted) / +3 s (solid)")
    f = r["fut"][1]; axm.imshow(np.ma.masked_less(f, .08), extent=[YMAX, -YMAX, 0, XMAX], cmap="cool", alpha=.6, interpolation="nearest")   # aux head, +3 s occupancy
    axm.plot(0, 0, marker="^", c="#1f77ff", ms=11, mec="w"); axm.set_xlim(YMAX, -YMAX); axm.set_ylim(0, XMAX); axm.set_xticks([]); axm.set_yticks([])
    axm.legend(fontsize=6.5, loc="lower left", framealpha=.85)
    axm.set_title("INPUT BEV (orange = objects now) + 256 anchors (colour = predicted PDMS) + truth", fontsize=7.5)
    axs.cla(); axs.scatter(r["true"], r["pred"], s=6, alpha=.5, c="#1f5fd1"); axs.plot([0, 1], [0, 1], "k--", lw=.8)
    axs.scatter(r["true"][ii], r["pred"][ii], s=50, c="#1f77ff", ec="k", label="model's pick"); axs.scatter(r["true"][jj], r["pred"][jj], s=50, c="#2ecc71", ec="k", label="true best")
    axs.set_xlabel("true PDMS (privileged scorer)", fontsize=7); axs.set_ylabel("predicted", fontsize=7); axs.tick_params(labelsize=7); axs.legend(fontsize=6.5); axs.set_title("predicted vs true score, all 256 anchors", fontsize=8.5)
    ts = np.arange(t + 1); axt.cla(); axt.plot(ts, [q["v"] for q in rec[:t + 1]], c="#1f5fd1", lw=2, label="ego speed (m/s)")
    axt.plot(ts, [min(q["gap"], 40) / 4 for q in rec[:t + 1]], c="#e67e22", lw=2, label="nearest object (m) / 4")
    for i_ in range(t + 1):
        if rec[i_]["col"]: axt.axvline(i_, c="k", lw=2)
    axt.set_xlim(0, T); axt.set_ylim(0, 12); axt.grid(alpha=.3); axt.tick_params(labelsize=7); axt.legend(fontsize=6.5, loc="upper right"); axt.set_title("telemetry (black line = collision)", fontsize=8.5)
    axe.cla(); axe.plot(ts, [q["err"] for q in rec[:t + 1]], c="#c0392b", lw=2, label="mean |pred - true| over anchors")
    axe.plot(ts, [q["regret"] for q in rec[:t + 1]], c="#8e44ad", lw=2, label="regret (true best - pick)"); axe.set_xlim(0, T); axe.set_ylim(0, .5); axe.grid(alpha=.3); axe.tick_params(labelsize=7)
    axe.legend(fontsize=6.5); axe.set_xlabel("frame (0.5 s)", fontsize=8); axe.set_title("scoring error and selection regret", fontsize=8.5)
    sub.set_text(f"frame {t:02d}  ego {r['v']:.1f} m/s  pick #{ii} pred {r['pred'][ii]:.2f} true {r['true'][ii]:.2f} | best #{jj} true {r['true'][jj]:.2f} | err {r['err']:.3f}")
    return []


anim = FuncAnimation(fig, draw, frames=T + 3, interval=300, blit=False)
anim.save(a.out + ".gif", writer=PillowWriter(fps=3)); anim.save(a.out + ".mp4", writer=FFMpegWriter(fps=3, bitrate=2200))
summ = dict(seed=s, frames=T, collisions=int(sum(q["col"] for q in rec)), mean_score_err=round(float(np.mean([q["err"] for q in rec])), 4),
            mean_regret=round(float(np.mean([q["regret"] for q in rec])), 4), pick_is_best=round(float(np.mean([q["i"] == q["j"] for q in rec])), 3))
json.dump(summ, open(a.out + ".json", "w"), indent=1); print(summ)
