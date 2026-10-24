"""AI-in-action simulation: same frozen weights, guidance OFF (left) vs ON (right).
Overlay per frame: ground-truth expert plan (cyan), sampled model plan (red=baseline / green=guided),
obstacle footprint, ego footprint, and live telemetry (speed, gap-to-obstacle, hazard energy, intrusion flag).
Usage: python simulate.py [--out assets/simulation.gif] [--seed N]"""
import argparse, yaml, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from evaluate import load
from src.closed_loop import run_episode
from src.scene import demo_trajectory, EGO_FRONT, OBS_LEN, OBS_HALF_W, EGO_HALF_W, DT, box_sdf, inflated_box

def fig_to_img(fig):
    fig.canvas.draw(); w, h = fig.canvas.get_width_height()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

def pick_scene(net, cfg, n=40):
    """Search for a scene where baseline collides and guided does not (so the demo shows the effect honestly: it is
    a *selected* example; aggregate numbers live in assets/results.json)."""
    rng = np.random.default_rng(21)
    for i in range(n):
        v0 = float(rng.uniform(9, 12.5)); D = float(v0 * v0 / 10 + 5 + rng.uniform(6, 16)); seed = 1000 + i
        a = run_episode(net, v0, D, 0.0, seed, cfg["sampler_steps"], cfg["guide_margin"], record=True)
        b = run_episode(net, v0, D, cfg["guide_lambda"], seed, cfg["sampler_steps"], cfg["guide_margin"], record=True)
        if a["collided"] and not b["collided"]: return v0, D, a, b
    raise RuntimeError("no contrast scene found")

def draw_panel(ax, title, rec, D, color, v0, upto):
    ax.set_facecolor("#2b2f36"); ax.set_xlim(-8, D + 14); ax.set_ylim(-5, 5)
    ax.add_patch(Rectangle((-8, -3.6), D + 22, 7.2, color="#3a3f48", zorder=0))
    for x in np.arange(-8, D + 14, 6): ax.plot([x, x + 3], [0, 0], color="#8a8f98", lw=1, zorder=1)
    ax.add_patch(Rectangle((D, -OBS_HALF_W), OBS_LEN, 2 * OBS_HALF_W, color="#e8a33d", zorder=3))
    ax.text(D + OBS_LEN / 2, 1.9, "STALLED VEHICLE", color="#e8a33d", ha="center", fontsize=7)
    fr = rec["log"][min(upto, len(rec["log"]) - 1)]; p = fr["p"]
    gt = demo_trajectory(fr["v"], fr["d_rel"], True, np.random.default_rng(0))
    ax.plot(p + gt[:, 0], gt[:, 1] + 0.0, "--", color="#4fd1ff", lw=1.8, label="expert / ground truth", zorder=4)
    ax.plot(p + fr["plan"][:, 0], fr["plan"][:, 1], "-o", color=color, ms=3, lw=2, label="model plan", zorder=5)
    ax.add_patch(Rectangle((p - 1.5, -EGO_HALF_W), 1.5 + EGO_FRONT, 2 * EGO_HALF_W, color="#ffffff", alpha=.9, zorder=6))
    pl = torch.from_numpy(fr["plan"][:12].astype(np.float32))      # first 3 s of the plan (what the car will act on)
    hit = bool((box_sdf(pl[:, 0] + p, pl[:, 1], inflated_box(torch.tensor(float(D)))) < 0).any())
    ax.set_title(title, color="white", fontsize=10); ax.tick_params(colors="#aaa", labelsize=7)
    ax.text(-7.5, 4.0, f"t={fr['t']:.1f}s  v={fr['v']:.1f} m/s  gap={fr['d_rel']-EGO_FRONT:.1f} m", color="white", fontsize=8)
    ax.text(-7.5, -4.6, "PLAN (next 3 s) INTRUDES ON OBSTACLE" if hit else "plan (next 3 s) clear of obstacle",
            color="#ff5d5d" if hit else "#6dd58c", fontsize=8, weight="bold")
    ax.legend(loc="upper right", fontsize=6, facecolor="#2b2f36", labelcolor="white", framealpha=.8)

def main(out, seed_unused=None):
    cfg = yaml.safe_load(open("config.yaml")); torch.manual_seed(0); net = load(cfg)
    v0, D, a, b = pick_scene(net, cfg)
    n = max(len(a["log"]), len(b["log"])); frames = []
    for i in range(n + 3):
        fig = plt.figure(figsize=(12, 9.5), facecolor="#14161a")
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.1])
        axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
        draw_panel(axa, "OLD WAY: frozen flow-matching sampler", a, D, "#ff5d5d", v0, i)
        draw_panel(axb, "NEW WAY: + hazard-energy guidance (same weights)", b, D, "#6dd58c", v0, i)
        axt = fig.add_subplot(gs[2, 0]); axt.set_facecolor("#1d2025")
        for rec, col, lab in [(a, "#ff5d5d", "baseline speed"), (b, "#6dd58c", "guided speed")]:
            k = min(i, len(rec["log"]) - 1) + 1
            axt.plot([f["t"] for f in rec["log"][:k]], [f["v"] for f in rec["log"][:k]], color=col, lw=2, label=lab)
        axt.set_xlim(0, max(a["log"][-1]["t"], b["log"][-1]["t"]) + 0.6); axt.set_ylim(0, v0 + 2)
        axt.set_xlabel("time (s)", color="#aaa"); axt.set_ylabel("ego speed (m/s)", color="#aaa"); axt.tick_params(colors="#aaa")
        axt.legend(facecolor="#1d2025", labelcolor="white", fontsize=8, loc="upper right")
        done_a = i >= len(a["log"]) - 1; done_b = i >= len(b["log"]) - 1
        axt.set_title(f"Outcome: baseline {'COLLISION' if done_a and a['collided'] else '...'}   |   guided "
                      f"{'stopped, gap %.1f m' % b['final_gap'] if done_b and not b['collided'] else '...'}   "
                      "[synthetic scene, selected example]", color="white", fontsize=9)
        fig.tight_layout(); frames.append(fig_to_img(fig)); plt.close(fig)
    frames += [frames[-1]] * 4
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=450, loop=0)
    print("wrote", out, len(frames), "frames; scene v0=%.1f D=%.1f" % (v0, D))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="assets/simulation.gif"); ap.add_argument("--seed")
    main(ap.parse_args().out)
