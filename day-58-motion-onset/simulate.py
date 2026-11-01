"""'AI in action' synthetic simulation: kerbside scene, motion-ONSET prediction.   python simulate.py --seed 7 --out assets/onset_demo
Needs checkpoints/{full,stateless}.pt from train.py.  Writes .gif + .mp4 + .json.

Left  : top-down street. INPUT = static road users (boxes) + observed velocity arrows once they move.
        PREDICTION = halo radius + label ~ P(start moving within 3 s) from the recurrent scene-graph GNN.
        GROUND TRUTH = green ring when the object really starts moving within the next 3 s (dashed = about to), solid green trail after.
Right : bar chart of current onset probabilities; probability trace for the 'star' object (full vs stateless vs truth);
        ego telemetry: a toy planner slows when max onset risk ahead is high.
"""
import argparse, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Rectangle, Circle
from motion_onset.synthetic import episode, CLASSES, N, T, H, DT, AREA
from motion_onset.model import MotionOnsetGNN

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--out", default="assets/onset_demo"); a = ap.parse_args()
# pick the first seed (>= given) whose episode has >=4 in-horizon onsets between frames 15-40, so the clip is informative
s = a.seed
while True:
    e = episode(np.random.default_rng(s)); on = e["onset"][(e["onset"] > 14) & (e["onset"] < 42)]
    if len(on) >= 3: break
    s += 1
tt = {k: torch.as_tensor(v)[None].float() if k != "vis" else torch.as_tensor(v)[None] for k, v in e.items() if k in ("pos", "vel", "vis", "sem")}
P = {}
for k, rec in (("full", True), ("stateless", False)):
    m = MotionOnsetGNN(recurrent=rec); m.load_state_dict(torch.load(f"checkpoints/{k}.pt")); m.eval()
    with torch.no_grad(): P[k] = m.rollout(tt["pos"], tt["vel"], tt["vis"], tt["sem"])[0].numpy()          # [T,N]
star = int(np.argmax(np.where((e["onset"] > 14) & (e["onset"] < 42), e["onset"], -1)))                    # last object to start moving
col = dict(pedestrian="#d95f02", cyclist="#7570b3", parked_car="#555", delivery_van="#1b9e77", e_scooter="#e7298a", traffic_cone="#e6ab02")
size = dict(pedestrian=(.8, .8), cyclist=(1.8, .8), parked_car=(4.2, 1.8), delivery_van=(5.5, 2.2), e_scooter=(1.2, .6), traffic_cone=(.5, .5))
ego_x = lambda t: 2 + 0.55 * t                                                                           # ego at lane y=-2.5 driving right
fig = plt.figure(figsize=(12.4, 6.4), dpi=100)
gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1], left=.04, right=.985, top=.84, bottom=.08, hspace=.75, wspace=.12)
axm, axb, axt, axe = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 1])
fig.text(.04, .94, "AI in action: will that parked/static road user start moving? Recurrent scene-graph GNN predicts motion ONSET (synthetic)", fontsize=11.5, weight="bold")
sub = fig.text(.04, .895, "", fontsize=9, family="monospace")
fig.text(.985, .002, f"Synthetic demo - arXiv 2610.38993-style re-implementation, NOT the paper's data (warehouse IsaacSim/hardware) nor real driving data. episode seed {s}", ha="right", fontsize=7.5, color="#555")
ego_v = []
for t in range(T):
    ahead = [P["full"][t, i] for i in range(N) if e["q"][t, i] and 0 < e["pos"][t, i, 0] - ego_x(t) < 14]
    ego_v.append(8 * (1 - .75 * max(ahead, default=0)))


def draw(f):
    t = min(f, T - 1)
    axm.cla(); axm.set_xlim(-1, AREA[0] + 1); axm.set_ylim(-5, AREA[1] + 5); axm.set_aspect("equal"); axm.tick_params(labelsize=7)
    axm.add_patch(Rectangle((-1, -4), AREA[0] + 2, 3, color="#ddd", zorder=0)); axm.text(0, -3.7, "ego lane", fontsize=7, color="#666")
    axm.add_patch(Rectangle((ego_x(t) - 2, -3.3), 4, 1.6, color="#1f5fd1", zorder=5)); axm.text(ego_x(t) - 1.7, -3.1, "EGO", color="w", fontsize=6.5, zorder=6)
    for i in range(N):
        if not e["vis"][t, i]: continue
        nm = CLASSES[e["cls"][i]]; x, y = e["pos"][t, i]; w, h = size[nm]; mv = e["onset"][i] <= t
        axm.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, color=col[nm], alpha=.55 if not mv else .95, zorder=3))
        if mv:
            v = e["vel"][t, i]; axm.annotate("", (x + v[0] * 1.2, y + v[1] * 1.2), (x, y), arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5), zorder=7)
            o0 = max(e["onset"][i], t - 6); axm.plot(e["pos"][o0:t + 1, i, 0], e["pos"][o0:t + 1, i, 1], color="#1a9850", lw=2, zorder=2)      # truth trail
        elif e["q"][t, i]:
            p = P["full"][t, i]
            axm.add_patch(Circle((x, y), .6 + 2.6 * p, fill=False, ec="#c0392b", lw=1 + 3 * p, alpha=.35 + .6 * p, zorder=4))        # prediction halo
            axm.text(x, y + max(h, 1) / 2 + .7 + 2.6 * p * .3, f"{nm[:6]} {p:.2f}", fontsize=6.3, ha="center", color="#c0392b", zorder=8)
            if e["y"][t, i]: axm.add_patch(Circle((x, y), 3.6, fill=False, ec="#1a9850", lw=2, ls="--", zorder=4))                    # truth: starts within 3 s
    axm.plot([], [], c="#c0392b", lw=3, label="PREDICTION: onset prob (halo size)"); axm.plot([], [], c="#1a9850", ls="--", lw=2, label="GROUND TRUTH: moves within 3 s"); axm.plot([], [], c="#1a9850", lw=2, label="GROUND TRUTH: recent path")
    axm.legend(fontsize=6.5, loc="upper left", ncol=3, framealpha=.9); axm.set_title("INPUT (boxes, observed velocity) -> PREDICTION (halos) -> GROUND TRUTH (green)", fontsize=8.5)
    axb.cla(); idx = [i for i in range(N) if e["q"][t, i]]; idx.sort(key=lambda i: -P["full"][t, i]); idx = idx[:7]
    axb.barh(range(len(idx)), [P["full"][t, i] for i in idx], color=["#1a9850" if e["y"][t, i] else "#bbb" for i in idx]); axb.set_yticks(range(len(idx)))
    axb.set_yticklabels([f"{CLASSES[e['cls'][i]][:9]} #{i}" for i in idx], fontsize=7); axb.invert_yaxis(); axb.set_xlim(0, 1); axb.tick_params(labelsize=7)
    axb.set_title("top-7 onset probabilities (green = truly moves in 3 s)", fontsize=8.5)
    x = np.arange(t + 1); axt.cla()
    axt.plot(x, P["full"][:t + 1, star], c="#1f5fd1", lw=2, label="full (GRU memory)"); axt.plot(x, P["stateless"][:t + 1, star], c="#f28e2b", lw=2, label="stateless")
    axt.fill_between(np.arange(T), 0, 1, where=e["y"][:, star] > 0, color="#1a9850", alpha=.18, label="truth: onset <= 3 s"); axt.axvline(e["onset"][star], c="#1a9850", ls="--", lw=1)
    axt.set_xlim(0, T); axt.set_ylim(0, 1); axt.legend(fontsize=6.5, loc="upper left"); axt.tick_params(labelsize=7); axt.grid(alpha=.3)
    axt.set_title(f"P(onset) for '{CLASSES[e['cls'][star]]} #{star}' (dashed = true onset)", fontsize=8.5)
    axe.cla(); axe.plot(x, ego_v[:t + 1], c="#1f5fd1", lw=2); axe.set_xlim(0, T); axe.set_ylim(0, 9); axe.grid(alpha=.3); axe.tick_params(labelsize=7)
    axe.set_xlabel("frame (0.5 s)", fontsize=8); axe.set_title("toy ego planner: speed advisory (m/s) = 8*(1 - 0.75*max onset risk ahead)", fontsize=8.5)
    nq = int(e["q"][t].sum()); sub.set_text(f"frame {t:02d}  static tracked {nq:02d}  moving {int((e['onset'] <= t).sum()):02d}  star p(full)={P['full'][t, star]:.2f} p(stateless)={P['stateless'][t, star]:.2f}  ego advisory {ego_v[t]:.1f} m/s")
    return []


anim = FuncAnimation(fig, draw, frames=T + 4, interval=170, blit=False)
anim.save(a.out + ".gif", writer=PillowWriter(fps=6)); anim.save(a.out + ".mp4", writer=FFMpegWriter(fps=6, bitrate=2200))
summ = dict(seed=s, star=dict(cls=CLASSES[e["cls"][star]], onset=int(e["onset"][star]), p_full_at_onset_minus_3=round(float(P["full"][max(e["onset"][star] - 3, 0), star]), 3), p_stateless_at_onset_minus_3=round(float(P["stateless"][max(e["onset"][star] - 3, 0), star]), 3)))
json.dump(summ, open(a.out + ".json", "w"), indent=1); print(summ)
