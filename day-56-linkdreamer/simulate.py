"""Synthetic 'AI in action' simulation: one scripted V2X drive with link outages.

python simulate.py --out assets/linkdreamer_demo        (needs checkpoints/ from train.py; writes .gif + .mp4 + .json)

BEV panel (INPUT -> GROUND TRUTH -> PREDICTION):
  grey shading        = ego blind area (ego sensor cannot see: truck shadows / range)
  blue heat           = the collaborator feature the detector is actually fed this frame (live / stale / imagined / coarse / none)
  green boxes         = GROUND TRUTH vehicles      orange x = ego-only detections      blue diamonds = LinkDreamer-style controller
Right: link quality + chosen mode timeline, telemetry (uncertainty, memory age, cumulative comm vs continuous high-rate), running F1.
"""
import argparse, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from linkdreamer import agent
from linkdreamer.synthetic import make_dataset, G, CELL, XMIN, YMIN, T_EP
from linkdreamer.models import LatentWorldModel, FusionDetector, TSNeuroFuzzy, MODES
from linkdreamer.metrics import match

MCOL = ["#888888", "#a0522d", "#8e44ad", "#17a2a2", "#1f5fd1"]
QCOL = {1.0: "#2ca02c", 0.5: "#f0a500", 0.0: "#d62728"}
LINK = np.ones((1, T_EP), np.float32); LINK[0, 5:13] = 0; LINK[0, 16:19] = .5; LINK[0, 19:] = 0


def load():
    wm, det, ct = LatentWorldModel(), FusionDetector(), TSNeuroFuzzy()
    for n, m in (("wm", wm), ("det", det), ("ctrl", ct)): m.load_state_dict(torch.load(f"checkpoints/{n}.pt"))
    agent.RISK_REF["v"] = json.load(open("checkpoints/meta.json"))["risk_ref"]
    return wm.eval(), det.eval(), ct.eval()


def running_f1(dets, objs):
    tp = fp = fn = 0; out = []
    for t in range(T_EP):
        if t >= 2:
            r, miss = match(dets[t], [tuple(o) for o in objs[t]])
            tp += sum(x for _, x in r); fp += sum(not x for _, x in r); fn += miss
        out.append(2 * tp / max(2 * tp + fp + fn, 1))
    return out


def main(a):
    wm, det, ct = load()
    ds = make_dataset(1, a.seed)
    runs = {k: agent.run(ds, [0], LINK, p, wm, det, ct, trace=(p == "ctrl")) for k, p in
            (("ego-only", "ego"), ("reuse", "reuse"), ("LinkDreamer-style", "ctrl"), ("upper ref", "upper"))}
    f1 = {k: running_f1(r["dets"][0], ds["objs"][0]) for k, r in runs.items()}
    ctr = runs["LinkDreamer-style"]; tr = ctr["trace"]
    modes = ctr["modes"][0].numpy(); cum = np.cumsum(ctr["comm"][0].numpy())
    fig = plt.figure(figsize=(12.4, 6.4), dpi=100)
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.15], height_ratios=[.55, 1, 1], left=.05, right=.98, top=.83, bottom=.08, hspace=.85, wspace=.18)
    axb, axt, axm, axf = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 1])
    fig.text(.05, .94, "AI in action: V2X link drops - an agent picks ego-only / reuse / imagine / refresh per frame (synthetic)", fontsize=12, weight="bold")
    sub = fig.text(.05, .885, "", fontsize=9, family="monospace")
    fig.text(.98, .002, "Synthetic demo - LinkDreamer-style re-implementation (arXiv 2610.39258), not V2X-Real data", ha="right", fontsize=8, color="#555")
    ext = [YMIN, -YMIN, XMIN, XMIN + G * CELL]

    def draw(f):
        t = min(f, T_EP - 1); ego = ds["ego"][0, t].numpy(); q = float(LINK[0, t]); m = int(modes[t])
        axb.cla()
        axb.imshow((1 - ego[1]), origin="lower", extent=ext, cmap=ListedColormap([(0, 0, 0, 0), (.35, .35, .35, .45)]), vmin=0, vmax=1, zorder=1)
        axb.imshow(tr[t]["coll"][0, 0].numpy(), origin="lower", extent=ext, cmap="Blues", alpha=.65, vmin=0, vmax=1.1, zorder=2)
        for ox, oy in ds["objs"][0][t]:
            axb.add_patch(Rectangle((oy - 1.0, ox - 2.2), 2.0, 4.4, fill=False, ec="#1a9850", lw=2.4, zorder=4))
        for key, mk, c, s in (("ego-only", "x", "#f28e2b", 70), ("LinkDreamer-style", "D", "#1f5fd1", 46)):
            d = runs[key]["dets"][0][t]
            if d: axb.scatter([p[1] for p in d], [p[0] for p in d], marker=mk, s=s, c=c, edgecolors="w" if mk == "D" else None, linewidths=.8, zorder=6, label=key)
        axb.plot(0, 0, "k^", ms=9, zorder=7); axb.plot([], [], color="#1a9850", lw=2.4, label="GROUND TRUTH"); axb.set_xlim(-24, 24); axb.set_ylim(XMIN, XMIN + G * CELL)
        axb.set_title(f"BEV: grey = ego blind, blue = fed collaborator feature [{MODES[m]}]", fontsize=8.5)
        axb.legend(loc="upper right", fontsize=7, framealpha=.9); axb.tick_params(labelsize=7); axb.set_xlabel("lateral (m)", fontsize=7); axb.set_ylabel("forward (m)", fontsize=7)
        axt.cla()
        for k in range(t + 1):
            axt.add_patch(Rectangle((k, 1), 1, 1, color=QCOL[float(LINK[0, k])])); axt.add_patch(Rectangle((k, 0), 1, 1, color=MCOL[int(modes[k])]))
        axt.set_xlim(0, T_EP); axt.set_ylim(0, 2); axt.set_yticks([.5, 1.5]); axt.set_yticklabels(["mode", "link"], fontsize=7); axt.tick_params(labelsize=7)
        axt.set_title("link (green up / amber degraded / red down) and chosen mode", fontsize=8.5)
        for i, nm in enumerate(MODES): axt.plot([], [], "s", color=MCOL[i], label=nm)
        axt.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.28), frameon=False)
        axm.cla(); tt = np.arange(t + 1)
        u = [float(tr[k]["state"][0, 0]) for k in tt]; fr = [float(tr[k]["state"][0, 2]) for k in tt]
        axm.plot(tt, u, color="#8e44ad", lw=2, label="prediction uncertainty"); axm.plot(tt, fr, color="#a0522d", lw=2, label="memory freshness")
        axm.plot(tt, cum[:t + 1] / T_EP, color="#1f5fd1", lw=2, ls="--", label="comm used / continuous high-rate"); axm.set_xlim(0, T_EP); axm.set_ylim(0, 1.05)
        axm.legend(fontsize=6.5, loc="upper right"); axm.grid(alpha=.3); axm.tick_params(labelsize=7); axm.set_title("controller telemetry", fontsize=8.5)
        axf.cla()
        for k, c in (("ego-only", "#f28e2b"), ("reuse", "#a0522d"), ("LinkDreamer-style", "#1f5fd1"), ("upper ref", "#2ca02c")):
            axf.plot(tt, f1[k][:t + 1], color=c, lw=2.2, ls=":" if k == "upper ref" else "-", label=k)
        axf.set_xlim(0, T_EP); axf.set_ylim(0, 1.02); axf.grid(alpha=.3); axf.legend(fontsize=6.5, loc="lower left", ncol=2); axf.tick_params(labelsize=7)
        axf.set_xlabel("frame", fontsize=8); axf.set_title("running F1 (cumulative, t>=2)", fontsize=8.5)
        s = tr[t]["state"][0]
        sub.set_text(f"frame {t:02d}  link q={q:.1f}  mode={MODES[m]:<22s} state u={s[0]:.2f} comp={s[1]:.2f} fresh={s[2]:.2f} budget={s[4]:.2f}  F1 ctrl {f1['LinkDreamer-style'][t]:.2f}")
        return []
    anim = FuncAnimation(fig, draw, frames=T_EP + 4, interval=200, blit=False)
    anim.save(a.out + ".gif", writer=PillowWriter(fps=5)); anim.save(a.out + ".mp4", writer=FFMpegWriter(fps=5, bitrate=2400))
    fin = {k: round(v[-1], 3) for k, v in f1.items()}
    json.dump(dict(final_running_f1=fin, comm_per_frame={k: round(r["comm_per_frame"], 3) for k, r in runs.items()},
                   modes=[MODES[int(x)] for x in modes]), open(a.out + ".json", "w"), indent=1)
    print(fin)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="assets/linkdreamer_demo"); ap.add_argument("--seed", type=int, default=11)
    main(ap.parse_args())
