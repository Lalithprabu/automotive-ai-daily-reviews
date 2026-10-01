"""Synthetic 'AI in action' simulation: same scene, OLD (regression) vs NEW (EMPlan) planner, overlaid on a BEV road with
ground-truth expert path, all K anchor candidates (faint), chosen path, and live telemetry (collision flag, reward, ADE).
Usage: python simulate.py [--out assets/emplan_simulation.gif] [--scenes 4]
"""
import argparse, torch, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from emplan.data import make_dataset
from emplan.model import EMPlan, RegressionPlanner
from emplan.reward import rule_terms, rule_reward, CAR_L, CAR_W

BG, ROAD, GT, OLD, NEW, TXT = "#0e1117", "#1b2230", "#3ddc84", "#ff5c5c", "#4aa8ff", "#e6e6e6"


def car(ax, x, y, c, lbl=None):
    ax.add_patch(Rectangle((x - CAR_L / 2, y - CAR_W / 2), CAR_L, CAR_W, color=c, zorder=5))
    if lbl: ax.text(x, y + 1.3, lbl, color=TXT, fontsize=7, ha="center")


def obstacles(s, t):
    out = [(s[1] + s[2] * t, 0.0, "lead")]
    if s[3] > 0: out.append((s[4] + s[5] * t, 3.5, ""))
    if s[6] > 0: out.append((s[7] + s[8] * t, -3.5, ""))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="assets/emplan_simulation.gif")
    ap.add_argument("--scenes", type=int, default=4); a = ap.parse_args()
    ck = torch.load("emplan_ckpt.pt"); cfg = ck["cfg"]; T, dt = cfg["data"]["horizon"], cfg["data"]["dt"]
    m = EMPlan(ck["anchors"], cfg["model"]["hidden"]); m.load_state_dict(ck["model"]); m.eval()
    r = RegressionPlanner(T, cfg["model"]["hidden"]); r.load_state_dict(torch.load("regression_ckpt.pt")["model"]); r.eval()
    scene, expert = make_dataset(600, 99, T, dt)
    # pick scenes where the old way collides but the new way does not (shown honestly + one case where both do fine)
    with torch.no_grad():
        po = r.plan(scene); pn, _ = m.plan(scene); logits, cands = m(scene)
        co = rule_terms(scene, po[:, None], dt)["collision"][:, 0]; cn = rule_terms(scene, pn[:, None], dt)["collision"][:, 0]
    good = torch.where((co > 0) & (cn == 0))[0][: a.scenes - 1].tolist()
    both_bad = torch.where((co > 0) & (cn > 0))[0][:1].tolist()    # honest failure case
    picks = (good + both_bad)[: a.scenes]
    frames = []; ts = torch.arange(1, T + 1) * dt
    for si in picks:
        s = scene[si].numpy(); gt, tn, to, cd = expert[si].numpy(), pn[si].numpy(), po[si].numpy(), cands[si].detach().numpy()
        for f in range(T + 1):
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.4, 1]}, facecolor=BG)
            ax.set_facecolor(ROAD); ax.set_xlim(-8, 75); ax.set_ylim(-6.5, 6.5)
            for y in (-5.25, 5.25): ax.axhline(y, color="#888", lw=2)
            for y in (-1.75, 1.75): ax.axhline(y, color="#555", lw=1, ls="--")
            for k in range(len(cd)): ax.plot(cd[k, :, 0], cd[k, :, 1], color="#888", alpha=0.12, lw=1)
            ax.plot(gt[:, 0], gt[:, 1], color=GT, lw=2, ls=":", label="expert (ground truth)")
            ax.plot(to[:, 0], to[:, 1], color=OLD, lw=2, label="OLD: regression")
            ax.plot(tn[:, 0], tn[:, 1], color=NEW, lw=2.5, label="NEW: EMPlan")
            t = 0.0 if f == 0 else float(ts[f - 1])
            for ox, oy, l in obstacles(s, t): car(ax, ox, oy, "#c9a227", l)
            idx = max(f - 1, 0); car(ax, (to[idx, 0] if f else 0), (to[idx, 1] if f else 0), OLD)
            car(ax, (tn[idx, 0] if f else 0), (tn[idx, 1] if f else 0), NEW)
            ax.legend(loc="upper right", fontsize=7, facecolor=BG, labelcolor=TXT); ax.tick_params(colors=TXT)
            ax.set_title(f"Scene {si} | t={t:.1f}s | ego v0={s[0]:.1f} m/s | lead gap={s[1]:.0f}m", color=TXT, fontsize=10)
            ax2.set_facecolor(BG); ax2.axis("off")
            def stats(p, name, c, y0):
                tm = rule_terms(torch.from_numpy(s)[None], torch.from_numpy(p)[None, None], dt)
                rw = rule_reward(torch.from_numpy(s)[None], torch.from_numpy(p)[None, None], dt).item()
                ade = np.linalg.norm(p - gt, axis=-1).mean()
                ax2.text(0, y0, name, color=c, fontsize=11, weight="bold", transform=ax2.transAxes)
                ax2.text(0, y0 - 0.1, f"collision: {'YES' if tm['collision'].item() else 'no'}", color=c, fontsize=10, transform=ax2.transAxes)
                ax2.text(0, y0 - 0.19, f"rule reward: {rw:+.2f}", color=TXT, fontsize=10, transform=ax2.transAxes)
                ax2.text(0, y0 - 0.28, f"ADE vs expert: {ade:.2f} m", color=TXT, fontsize=10, transform=ax2.transAxes)
            stats(to, "OLD  single-mode regression", OLD, 0.92); stats(tn, "NEW  EMPlan (anchors+offset+RL pref)", NEW, 0.55)
            ax2.text(0, 0.1, "Grey: all K sparse anchor candidates\nSynthetic scene - not NAVSIM", color="#999", fontsize=8, transform=ax2.transAxes)
            fig.tight_layout(); fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())); plt.close(fig)
        frames += [frames[-1]] * 2
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=350, loop=0)
    print("wrote", a.out, len(frames), "frames; scenes", picks)


if __name__ == "__main__":
    main()
