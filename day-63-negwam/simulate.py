"""'AI in action' simulation: replay held-out hard scenes with three trajectories overlaid on a road-aligned BEV
  GREEN = expert (ground truth) | RED = baseline WAM (flow matching only) | BLUE = NegWAM-style (near-expert negatives)
plus live telemetry (gap to lead, lateral offset) and a running scoreboard. All data is the synthetic world in
negwam/world.py with the real trained checkpoints from train.py.

Usage:
  python simulate.py --ckpt checkpoints --seed 0 --scenes first --n 6 --out assets/negwam_demo
  --scenes first     : first N held-out hard scenes (not cherry-picked)
  --scenes disagree  : scenes where the two models' safety outcomes differ (labelled on screen as such)
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio.v2 as imageio
from negwam.world import (build_dataset, norm_scene, denorm_traj, safety, centreline, lead_positions, T, DT,
                          CAR_LEN, LANE_HALF)
from negwam.model import NegWAM

BG, FG, GRID = "#0d1117", "#e6edf3", "#30363d"
COL = {"expert": "#3fb950", "base": "#f85149", "neg": "#58a6ff", "lead": "#d2a8ff"}
NAME = {"expert": "Expert (ground truth)", "base": "Baseline WAM", "neg": "NegWAM-style"}


def load(ckpt, tag, seed):
    m = NegWAM(); m.load_state_dict(torch.load(os.path.join(ckpt, f"{tag}_seed{seed}.pt"))); m.eval(); return m


@torch.no_grad()
def predict(m, sc, seed=123):
    g = torch.Generator().manual_seed(seed)
    scn = torch.tensor(norm_scene(sc))
    tr = m.sample(scn, steps=8, generator=g)
    _, lead, _ = m(scn, tr, torch.ones(len(sc)))
    return denorm_traj(tr.numpy()).astype(np.float64), (lead.numpy() * 60.0)


def interp(traj, sc0, tq):
    """Interpolate a (T,2) trajectory (+ t=0 anchor) at times tq -> (len(tq),2) in road-aligned coords (x, lateral)."""
    kappa, y0 = sc0[1], sc0[2]
    ts = np.concatenate([[0.0], np.arange(1, T + 1) * DT])
    x = np.concatenate([[0.0], traj[:, 0]]); y = np.concatenate([[y0], traj[:, 1]])
    lat = y - centreline(x, kappa)
    return np.stack([np.interp(tq, ts, x), np.interp(tq, ts, lat)], 1)


def render(args):
    D = build_dataset(1000, seed=2001, hard=True)             # same held-out hard split as train.py
    sc_all = D["scene"]
    base, neg = load(args.ckpt, "base", args.seed), load(args.ckpt, "neg", args.seed)
    pb, lb = predict(base, sc_all); pn, ln = predict(neg, sc_all)
    ex = D["expert"].astype(np.float64)
    s = {k: safety(sc_all.astype(np.float64), v) for k, v in dict(expert=ex, base=pb, neg=pn).items()}
    bad = lambda k: s[k]["collision"] | s[k]["offlane"] | ~s[k]["ttc_ok"]
    if args.scenes == "disagree":
        idx = np.where(bad("base") != bad("neg"))[0][: args.n]
        mode = "scenes where the two models' safety outcomes differ"
    else:
        idx = np.arange(args.n); mode = f"first {args.n} held-out hard scenes (not cherry-picked)"
    print("scene ids", idx.tolist(), "|", mode)
    FPS, F_PER, HOLD = 12, 24, 6
    tq_all = np.linspace(0, T * DT, F_PER)
    frames = []
    tally = {k: dict(col=0, off=0, hw=0, n=0) for k in ("base", "neg")}
    for si, i in enumerate(idx):
        sc = sc_all[i].astype(np.float64)
        xl = np.concatenate([[sc[3]], lead_positions(sc[None])[0]]); tl = np.concatenate([[0], np.arange(1, T + 1) * DT])
        trs = dict(expert=ex[i], base=pb[i], neg=pn[i])
        for k in ("base", "neg"):
            tally[k]["n"] += 1
            tally[k]["col"] += int(s[k]["collision"][i]); tally[k]["off"] += int(s[k]["offlane"][i]); tally[k]["hw"] += int(not s[k]["ttc_ok"][i])
        for f in range(F_PER + HOLD):
            tq = tq_all[min(f, F_PER - 1)]
            tcur = tq
            fig = plt.figure(figsize=(16, 9), dpi=80, facecolor=BG)
            gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.0, 0.05], hspace=0.38, wspace=0.22, left=0.05, right=0.98, top=0.9, bottom=0.07)
            ax = fig.add_subplot(gs[0, :]); a2 = fig.add_subplot(gs[1, 0]); a3 = fig.add_subplot(gs[1, 1]); a4 = fig.add_subplot(gs[1, 2])
            for a in (ax, a2, a3, a4):
                a.set_facecolor(BG); a.tick_params(colors=FG, labelsize=9)
                for sp in a.spines.values(): sp.set_color(GRID)
            fig.suptitle(f"Day 63 | NegWAM-style negative-aware world-action model  -  synthetic replay, scene {si+1}/{len(idx)}", color=FG, fontsize=15, x=0.05, ha="left")
            # ---- road-aligned BEV (lateral axis exaggerated) ----
            ax.axhspan(-1.75, 1.75, color="#161b22"); ax.axhline(1.75, color=GRID, lw=2); ax.axhline(-1.75, color=GRID, lw=2)
            ax.axhline(LANE_HALF, color="#8b949e", lw=0.8, ls=":"); ax.axhline(-LANE_HALF, color="#8b949e", lw=0.8, ls=":")
            ax.set_xlim(-6, 78); ax.set_ylim(-3.2, 3.2); ax.set_xlabel("distance along lane [m]  (lateral axis exaggerated)", color=FG, fontsize=9)
            ax.set_ylabel("lateral [m]", color=FG, fontsize=9)
            # lead car now + predicted world (NegWAM world head: ghost boxes)
            xlt = np.interp(tcur, tl, xl)
            ax.add_patch(Rectangle((xlt - CAR_LEN / 2, -0.9), CAR_LEN, 1.8, fc=COL["lead"], ec="white", lw=1.2, alpha=0.95, zorder=5))
            ax.text(xlt, 1.15, "lead", color=COL["lead"], ha="center", fontsize=9)
            for j in range(0, T, 2):
                ax.add_patch(Rectangle((ln[i, j] - CAR_LEN / 2, -0.9), CAR_LEN, 1.8, fc="none", ec=COL["lead"], lw=0.8, ls="--", alpha=0.35 + 0.08 * j))
            for k in ("expert", "base", "neg"):
                full = interp(trs[k], sc, np.linspace(0, T * DT, 60))
                ax.plot(full[:, 0], full[:, 1], color=COL[k], lw=2.2 if k != "expert" else 3.5, alpha=0.9 if k != "expert" else 0.45, zorder=3)
                cur = interp(trs[k], sc, np.array([tcur]))[0]
                hit = k in s and bool(s[k]["collision"][i]) and (xlt - cur[0]) < CAR_LEN
                ax.add_patch(Rectangle((cur[0] - CAR_LEN / 2, cur[1] - 0.45), CAR_LEN, 0.9, fc=COL[k], ec="white" if hit else COL[k], lw=2.5 if hit else 0.5, alpha=0.8, zorder=6))
            ax.text(-5, 2.6, "dashed purple = world head's predicted lead positions (NegWAM)", color=COL["lead"], fontsize=9)
            for r, k in enumerate(("expert", "base", "neg")):
                ax.text(60, 2.75 - 0.5 * r, f"{NAME[k]}", color=COL[k], fontsize=10, fontweight="bold")
            # ---- telemetry ----
            tt = np.linspace(0, T * DT, 60)
            xlf = np.interp(tt, tl, xl)
            for k in ("expert", "base", "neg"):
                cu = interp(trs[k], sc, tt)
                a2.plot(tt, xlf - cu[:, 0], color=COL[k], lw=2); a3.plot(tt, cu[:, 1], color=COL[k], lw=2)
            a2.axhline(CAR_LEN, color="#f0883e", ls="--", lw=1.2); a2.text(0.05, CAR_LEN + 0.6, "collision (gap < 5 m)", color="#f0883e", fontsize=8)
            a2.set_title("gap to lead car [m]", color=FG, fontsize=11); a2.set_xlabel("time [s]", color=FG, fontsize=9)
            a3.axhline(LANE_HALF, color="#f0883e", ls="--", lw=1.2); a3.axhline(-LANE_HALF, color="#f0883e", ls="--", lw=1.2)
            a3.set_title("lateral offset from lane centre [m]", color=FG, fontsize=11); a3.set_xlabel("time [s]", color=FG, fontsize=9)
            for a in (a2, a3): a.axvline(tcur, color=FG, lw=1, alpha=0.6); a.grid(color=GRID, alpha=0.5)
            a4.axis("off")
            lines = [f"scene: v0={sc[0]:.1f} m/s, lead gap={sc[3]:.0f} m", f"       v_lead={sc[4]:.1f} m/s, a_lead={sc[5]:.1f} m/s2",
                     f"world-head lead err (mean): {np.abs(ln[i]-lead_positions(sc[None])[0]).mean():.2f} m", "",
                     "SCOREBOARD (scenes so far; lower is better)", f"{'':14s}{'coll':>6s}{'offlane':>9s}{'headway':>9s}"]
            for k in ("base", "neg"):
                t_ = tally[k]; lines.append(f"{NAME[k][:13]:14s}{t_['col']:>6d}{t_['off']:>9d}{t_['hw']:>9d}")
            lines += ["", "mode:", f"  {mode[:44]}", f"  {mode[44:]}", "synthetic world, 1 model seed.", "NOT NAVSIM, NOT paper numbers."]
            a4.text(0, 1, "\n".join(lines), color=FG, family="monospace", fontsize=10, va="top")
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
            plt.close(fig)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(args.out + ".gif", frames, fps=FPS, loop=0)
    imageio.mimsave(args.out + ".mp4", frames, fps=FPS, macro_block_size=1)
    print("wrote", args.out + ".gif/.mp4", len(frames), "frames", frames[0].shape)
    return tally


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenes", choices=["first", "disagree"], default="first"); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="assets/negwam_demo")
    print(render(ap.parse_args()))
