"""Render the 'AI in action' simulation: observed scene -> one-step intent-conditioned candidate fan ->
play-out of the TRUE future against ground truth (green), old-way multi-query pick (red) and ManeuverFlow pick (blue).
All paths/scores come from the real trained checkpoints on a held-out synthetic scene. Usage: python simulate.py [--out sim.gif]"""
import argparse, io
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from maneuverflow.data import make_dataset
from maneuverflow.model import ManeuverFlow
from maneuverflow.baseline import MultiQueryPlanner
from maneuverflow.scoring import factors, pdms, TF, DT, MANEUVERS, classify_maneuver, COLL_DX, COLL_DY

BLUE, RED, GREEN, GREY, GOLD = "#1f6fd6", "#d6403a", "#2a9d4b", "#6b6f76", "#e0a800"


def load():
    ck, bk = torch.load("maneuverflow.pt"), torch.load("baseline.pt")
    c = ck["cfg"]
    mf = ManeuverFlow(c["d_model"], c["topk"], c["per_man"], ck["prior"], c["prior_sigma"]); mf.load_state_dict(ck["state"]); mf.eval()
    bl = MultiQueryPlanner(c["d_model"], c["topk"] * c["per_man"]); bl.load_state_dict(bk["state"]); bl.eval()
    return mf, bl


def pick_scene(mf, bl, n=400, seed=2026):
    d = make_dataset(n, seed)
    best, bscore = None, -9
    for i in range(n):
        s = {k: v[i:i + 1] for k, v in d.items()}
        g = torch.Generator().manual_seed(5)
        o, b = mf.predict(s["hist"], s["agents"], gen=g), bl.predict(s["hist"], s["agents"])
        v = s["agents"][..., 5]
        fn, fo = factors(o["best"], s["fut_agents"], v), factors(b["best"], s["fut_agents"], v)
        cv = s["agents"][..., None, :2] + s["agents"][..., None, 2:4] * (torch.arange(1, TF + 1) * DT)[None, None, :, None]
        event = ((cv - s["fut_agents"]).norm(dim=-1).amax(-1) * v).max().item()       # a cut-in / hard-brake occurs
        sc = (fn["nc"] - fo["nc"]).item() * 3 + float(pdms(fn) - pdms(fo)) + min(event, 6) / 6
        if fn["nc"].item() == 1 and fo["nc"].item() == 0 and event > 2 and sc > bscore:
            best, bscore = (s, i), sc
    return best if best else (None, None)


def box(ax, x, y, c, a=1.0, fill=True, lw=1.2, ls="-"):
    ax.add_patch(Rectangle((x - 2.25, y - 1.0), 4.5, 2.0, fc=c if fill else "none", ec=c, alpha=a, lw=lw, ls=ls))


def render(s, mf, bl, out):
    g = torch.Generator().manual_seed(5)
    o, b = mf.predict(s["hist"], s["agents"], gen=g), bl.predict(s["hist"], s["agents"])
    ag, fa, gt = s["agents"][0], s["fut_agents"][0], s["gt"][0]
    valid = ag[:, 5] > 0.5
    newp, oldp = o["best"][0], b["best"][0]
    cands, sc, man = o["cands"][0], o["scores"][0], o["maneuvers"][0]
    lab = classify_maneuver(cands)
    fpos = o["fut_pos"][0]
    f_new, f_old = factors(newp, fa, valid.float()), factors(oldp, fa, valid.float())
    hist = s["hist"][0, :, :2]
    nsteps_obs, nsteps_gen, nsteps_play = 6, 8, TF + 2
    frames = []

    def clearance(path, k):
        d = []
        for n in range(len(valid)):
            if valid[n]:
                d.append(torch.maximum((path[k, 0] - fa[n, k, 0]).abs() / COLL_DX, (path[k, 1] - fa[n, k, 1]).abs() / COLL_DY).item())
        return min(d) if d else 9.0

    def frame(phase, k):
        fig = plt.figure(figsize=(11, 6.4), dpi=100)
        ax = fig.add_axes([0.02, 0.45, 0.96, 0.46]); ax2 = fig.add_axes([0.08, 0.07, 0.38, 0.28]); ax3 = fig.add_axes([0.58, 0.07, 0.38, 0.28])
        for y in (-5.25, 5.25): ax.axhline(y, color="#222", lw=2)
        for y in (-1.75, 1.75): ax.axhline(y, color="#bbb", lw=1, ls=(0, (6, 6)))
        ax.set_xlim(-16, 66); ax.set_ylim(-6.6, 6.6); ax.set_xticks([]); ax.set_yticks([])
        ax.plot(hist[:, 0], hist[:, 1], ":", color=GREY, lw=2)
        # agents: current state (grey ghost) always; true motion in play-out; predicted-future ghosts in generate phase
        for n in range(len(valid)):
            if not valid[n]: continue
            box(ax, ag[n, 0].item(), ag[n, 1].item(), GREY, 0.25)
            if phase >= 1:
                ax.plot(fpos[n, :, 0], fpos[n, :, 1], "--", color=GOLD, lw=1.2)
                ax.plot(fpos[n, -1, 0], fpos[n, -1, 1], "x", color=GOLD, ms=5)
            if phase == 2: box(ax, fa[n, k, 0].item(), fa[n, k, 1].item(), "#444", 0.9)
            if ag[n, 4] > 0.5: ax.text(ag[n, 0].item(), ag[n, 1].item() + 1.6, "signal/brake", color=GOLD, fontsize=7, ha="center")
        box(ax, 0, 0, "black", 0.9)
        if phase >= 1:
            nshow = len(cands) if phase == 2 else min(len(cands), 2 * (k + 1))
            order = sc.argsort(descending=True)
            for j in order[:nshow]:
                ax.plot(cands[j, :, 0], cands[j, :, 1], color=BLUE, alpha=0.18 + 0.5 * float(sc[j]), lw=1.2)
        if phase == 2:
            ax.plot(gt[:k + 1, 0], gt[:k + 1, 1], color=GREEN, lw=3)
            ax.plot(oldp[:k + 1, 0], oldp[:k + 1, 1], color=RED, lw=2.2, ls="--")
            ax.plot(newp[:k + 1, 0], newp[:k + 1, 1], color=BLUE, lw=3)
            for p, c in ((gt, GREEN), (oldp, RED), (newp, BLUE)): box(ax, p[k, 0].item(), p[k, 1].item(), c, 0.85, fill=False, lw=2)
        elif phase == 1 and k >= nsteps_gen - 1:
            ax.plot(oldp[:, 0], oldp[:, 1], color=RED, lw=1.5, ls="--", alpha=0.7); ax.plot(newp[:, 0], newp[:, 1], color=BLUE, lw=3)
        title = ["1. OBSERVE: history + agent state (hint flags in gold)", "2. ONE-STEP GENERATE + SCORE with current & predicted-future evidence", "3. PLAY-OUT of the true future"][phase]
        ax.set_title(title, fontsize=10, loc="left")
        ax.text(-15.5, -6.3, "grey=observed  green=truth  red=old multi-query  blue=ManeuverFlow  gold=world-model forecast", fontsize=7)
        ax.text(-15.5, 5.6, "Synthetic data - illustrative reconstruction", fontsize=7, color="#a33")
        # panel 2: candidate scores
        ax2.set_title("candidate score (learned PDM)", fontsize=8)
        if phase >= 1:
            order = sc.argsort(descending=True)[:8]
            ax2.barh(range(len(order)), [float(sc[j]) for j in order], color=[BLUE if int(j) == int(o["best_idx"][0]) else "#9bbbe8" for j in order])
            ax2.set_yticks(range(len(order))); ax2.set_yticklabels([MANEUVERS[int(man[j])] + ("*" if int(j) >= len(cands) // 2 else "") for j in order], fontsize=6)
            ax2.invert_yaxis(); ax2.set_xlim(0, 1)
        ax2.tick_params(labelsize=6)
        # panel 3: clearance telemetry
        ax3.set_title("min clearance to true traffic (1.0 = contact)", fontsize=8)
        if phase == 2:
            kk = np.arange(k + 1)
            for p, c, lb in ((newp, BLUE, "ManeuverFlow"), (oldp, RED, "old"), (gt, GREEN, "truth")):
                ax3.plot(kk * DT, [clearance(p, j) for j in kk], color=c, label=lb, lw=2)
            ax3.axhline(1.0, color="k", lw=0.8, ls=":"); ax3.legend(fontsize=6, loc="upper right")
            ax3.text(0.02, 0.05, f"PDMS new {pdms(f_new).item():.2f} | old {pdms(f_old).item():.2f}", transform=ax3.transAxes, fontsize=7)
        ax3.set_xlim(0, TF * DT); ax3.set_ylim(0, 4); ax3.tick_params(labelsize=6)
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
        return Image.open(buf).convert("P", palette=Image.ADAPTIVE)

    for k in range(nsteps_obs): frames.append((frame(0, k), 150))
    for k in range(nsteps_gen): frames.append((frame(1, k), 200 if k < nsteps_gen - 1 else 900))
    for k in range(TF): frames.append((frame(2, k), 380))
    frames.append((frame(2, TF - 1), 1800))
    frames[0][0].save(out, save_all=True, append_images=[f for f, _ in frames[1:]], duration=[d for _, d in frames], loop=0, optimize=False)
    return dict(pdms_new=pdms(f_new).item(), pdms_old=pdms(f_old).item(), nc_new=f_new["nc"].item(), nc_old=f_old["nc"].item(), frames=len(frames))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="assets/maneuverflow_simulation.gif"); a = ap.parse_args()
    mf, bl = load()
    s, i = pick_scene(mf, bl)
    if s is None: raise SystemExit("no suitable demo scene (old collides / new safe, with a real cut-in or hard-brake) found - widen the seed search")
    print("scene", i, render(s, mf, bl, a.out))
