"""Render the 'AI in action' GIF: same cut-in, two ego strategies, live KG-Bayes posterior + telemetry.

Panels:  (1) top-down highway: cutter + ghost-ego(reactive, red) + ego(predictive, green), predicted-intent halo
         (2) live posterior bars  P(LLC)/P(LK)/P(RLC) + evidence triples that drive them
         (3) telemetry: decel (m/s^2) and bumper gap (m) over time, reactive vs predictive.
Usage: python simulate.py [--seed N] [--frames 48] [--out assets/simulation.gif]
"""
import argparse, sys, pathlib, numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from evaluate import load
from kglc.decision import run_cutin, lc_log_prior
from kglc.data import RELATIONS, MANEUVERS, cutter_features, discretise, ENTITIES, LANE_W

BG, FG, GRID = "#0e1117", "#e6edf3", "#30363d"
C_RE, C_PR, C_CUT, C_GT = "#ff5c5c", "#3ddc97", "#58a6ff", "#f2cc60"


def pick_seed(kg, cfg, LP, start=1000):
    """First seed where reactive collides and predictive does not (a representative 'AI helps' case)."""
    for s in range(start, start + 400):
        a = run_cutin(kg, "reactive", cfg, s, LP); b = run_cutin(kg, "predictive", cfg, s, LP)
        if a.collided and not b.collided: return s
    return start


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--frames", type=int, default=48); ap.add_argument("--out", default="assets/simulation.gif"); a = ap.parse_args()
    kg, _, cfg = load(); dc = dict(cfg["decision"]); dc["crossing_gap_m"] = dc["demo_gap_m"]
    LP = lc_log_prior(dc)
    seed = a.seed if a.seed is not None else pick_seed(kg, dc, LP)
    re_, pr = run_cutin(kg, "reactive", dc, seed, LP), run_cutin(kg, "predictive", dc, seed, LP)
    n = min(len(re_.t), len(pr.t)) if not (re_.collided and not pr.collided) else len(re_.t)
    n = len(re_.t) + 10 if re_.collided else len(re_.t)
    n = min(n, len(pr.t)); idxs = np.linspace(0, n - 1, a.frames).astype(int)
    frames = []
    for fi in idxs:
        fig = plt.figure(figsize=(13, 5.6), facecolor=BG, dpi=90)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1], height_ratios=[1.15, 1], hspace=0.38, wspace=0.18)
        ax = fig.add_subplot(gs[0, 0]); axp = fig.add_subplot(gs[0, 1]); axt = fig.add_subplot(gs[1, :])
        for x in (ax, axp, axt):
            x.set_facecolor(BG); [s.set_color(GRID) for s in x.spines.values()]; x.tick_params(colors=FG, labelsize=8)
        # ---- BEV (ego-centred on predictive ego)
        s0 = pr.s_ego[min(fi, len(pr.s_ego)-1)]; ax.set_xlim(s0 - 25, s0 + 85); ax.set_ylim(-3 * LANE_W / 2 - .3, 3 * LANE_W / 2 + .3)
        for y in (-LANE_W/2, LANE_W/2): ax.axhline(y, color=FG, ls="--", lw=.8, alpha=.5)
        ax.axhline(-3*LANE_W/2, color=FG, lw=1.5); ax.axhline(3*LANE_W/2, color=FG, lw=1.5)
        def car(x, y, c, lab, a_=1.0, hatch=None):
            ax.add_patch(Rectangle((x - 4.5, y - .95), 4.5, 1.9, fc=c, ec=FG, lw=.8, alpha=a_, hatch=hatch)); ax.text(x - 2.2, y + 1.15, lab, color=c, fontsize=7, ha="center")
        k_r = min(fi, len(re_.s_ego)-1); k_p = min(fi, len(pr.s_ego)-1)
        car(re_.s_ego[k_r], -0.25, C_RE, "ego (reactive)", .55); car(pr.s_ego[k_p], 0.25, C_PR, "ego (KG-Bayes)", .95)
        car(pr.s_cut[k_p], pr.y_cut[k_p], C_CUT, "cut-in vehicle")
        p_ent = pr.p_enter[k_p]
        ax.add_patch(plt.Circle((pr.s_cut[k_p] - 2.2, pr.y_cut[k_p]), 1.2 + 4 * p_ent, fc="none", ec=C_GT, lw=1.5 + 3 * p_ent, alpha=.25 + .7 * p_ent))
        ax.text(pr.s_cut[k_p] + 1, pr.y_cut[k_p] - 2.0, f"P(enter lane)={p_ent:.2f}", color=C_GT, fontsize=8)
        if re_.collided and k_r >= len(re_.t) - 1 or (re_.collided and idxs[-1] >= len(re_.t)-1 and fi >= len(re_.t)-1):
            ax.text(re_.s_ego[-1] - 20, 2.6, "REACTIVE: COLLISION", color=C_RE, fontsize=11, weight="bold")
        ax.set_title("Input: highway scene  |  overlay: model's intent belief", color=FG, fontsize=10); ax.set_yticks([])
        ax.set_xlabel("longitudinal position (m)", color=FG, fontsize=8)
        # ---- posterior bars
        post = pr.post[k_p]; axp.barh(MANEUVERS, post, color=[C_CUT, "#8b949e", C_CUT]); axp.set_xlim(0, 1); axp.invert_yaxis()
        axp.set_title(f"Posterior  P(h | evidence)   t={pr.t[k_p]:.2f}s", color=FG, fontsize=10)
        for i, v in enumerate(post): axp.text(min(v + .02, .85), i, f"{v:.2f}", color=FG, va="center", fontsize=9)
        # ---- telemetry
        tt = np.array(pr.t); axt.set_title("Telemetry: ego deceleration (solid, left) & bumper gap (dashed, right)", color=FG, fontsize=10)
        axt.plot(re_.t, -np.array(re_.a_ego), color=C_RE, lw=1.4, label="reactive decel"); axt.plot(pr.t, -np.array(pr.a_ego), color=C_PR, lw=1.4, label="KG-Bayes decel")
        axt.axvline(pr.t[k_p], color=FG, lw=1); axt.set_ylabel("m/s²", color=FG, fontsize=8); axt.set_xlabel("time (s)", color=FG, fontsize=8)
        axg = axt.twinx(); axg.plot(re_.t, re_.gap, color=C_RE, ls="--", lw=1, alpha=.7); axg.plot(pr.t, pr.gap, color=C_PR, ls="--", lw=1, alpha=.7)
        axg.axhline(0, color=C_RE, lw=.6, alpha=.5); axg.tick_params(colors=FG, labelsize=8); axg.set_facecolor(BG)
        axt.legend(loc="upper left", fontsize=8, facecolor=BG, labelcolor=FG, edgecolor=GRID)
        fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])); plt.close(fig)
    pathlib.Path(a.out).parent.mkdir(exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=140, loop=0, optimize=True)
    print(f"seed={seed}  reactive: {re_.stats()}\n          predictive: {pr.stats()}\nwrote {a.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
