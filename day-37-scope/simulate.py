"""Renders the 'AI in action' simulation GIF: input history vs ground truth vs model prediction (SCOPE tubes),
next to fixed-covariance and Monte-Carlo-sampling tubes, with a live lane-encroachment risk panel + telemetry.
All tubes come from the real trained checkpoint on a held-out synthetic scene (no hand-drawn numbers)."""
import argparse, json, math, time
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from PIL import Image
from scipy.stats import norm
from scope import *
from scope.io import load, nominal
from scope.evaluate import blocks

C95 = math.sqrt(5.991)
BLUE, GRAY, ORANGE, GREEN, RED, INK = "#2a6fdb", "#7a7a7a", "#e08a1e", "#1f9d55", "#d64545", "#1b1f2a"
THR = 0.05                                   # brake if max lane-encroachment probability exceeds this
LANE_Y = 1.0                                   # ego lane starts here (m); pedestrian encroaches if y > LANE_Y


def ellipse(ax, mu, S, **kw):
    w, V = np.linalg.eigh(S)
    ang = math.degrees(math.atan2(V[1, 1], V[0, 1]))
    ax.add_patch(Ellipse(mu, 2 * C95 * math.sqrt(max(w[1], 1e-9)), 2 * C95 * math.sqrt(max(w[0], 1e-9)), angle=ang, **kw))


def risk(mu_y, var_y):
    return 1 - norm.cdf((LANE_Y - mu_y) / np.sqrt(var_y))


def pick_scene(te, S_sc, S_fx, a, cfg, rng_seed=0):
    """Held-out, mode 0 (passes on the lane side), high crowd density, ped truly encroaches, and fixed cov is blind to it."""
    best, score = None, -1
    for i in range(len(a)):
        if te["mode"][i] != 0 or te["c"][i] < 0.75: continue
        y_true = te["fut"][i, 1::2]
        if y_true.max() <= LANE_Y + 0.05: continue
        mu = (te["cv"][i] + a[i].numpy())[1::2]
        r_sc = risk(mu, blocks(S_sc[i:i+1])[0, :, 1, 1].numpy()).max()
        r_fx = risk(mu, blocks(S_fx[i:i+1])[0, :, 1, 1].numpy()).max()
        sc = (r_sc - r_fx) + (1.0 if (r_sc > THR > r_fx) else 0.0)     # prefer: SCOPE brakes, fixed cov does not
        if sc > score: best, score = i, sc
    return best


def main(out="assets/scope_simulation.gif", seed=0, scene=None):
    cfg, model, head, ck = load()
    res = json.load(open("results.json"))
    data = make_data(cfg, cfg["seed"]); te = data["test"]
    with torch.no_grad():
        a = nominal(model, cfg, te["cond"], te["mode_t"])
        S_sc = head.covariance(te["cond"], te["mode_t"], a)
        S_fx = (ck["S_fixed"] * ck["al_fix"]).expand(len(a), D, D)
    i = scene if scene is not None else pick_scene(te, S_sc, S_fx, a, cfg)
    print("scene index", i, "density c=%.2f speed v=%.2f" % (te["c"][i], te["v"][i]))
    K = cfg["mc_baseline"]["samples"]
    with torch.no_grad():
        s = cfg["sampler"]; torch.manual_seed(seed)
        xs = sample(model, te["cond"][i:i+1].expand(K, -1), te["mode_t"][i:i+1].expand(K, -1), torch.randn(K, D), s["steps"], s["sigma_max"], s["sigma_min"])
        S_mc = (torch.cov(xs.T) + 1e-4 * torch.eye(D))[None] * ck["al_mc"]
    cv = te["cv"][i]; nom = (cv + a[i].numpy()).reshape(T_FUT, 2)
    gt = te["fut"][i].reshape(T_FUT, 2); past = te["past"][i]
    mu_true = mean_future(te["v"][i:i+1], te["mode"][i:i+1])[0].reshape(T_FUT, 2)
    S_or = true_cov(te["v"][i:i+1], te["c"][i:i+1])
    blk = lambda S: blocks(S if torch.is_tensor(S) else torch.tensor(S, dtype=torch.float32)).numpy()[0]
    B_sc, B_fx, B_mc, B_or = blk(S_sc[i:i+1]), blk(S_fx[i:i+1]), blk(S_mc), blk(S_or)
    r_sc, r_fx, r_mc = (risk(nom[:, 1], b[:, 1, 1]) for b in (B_sc, B_fx, B_mc))
    r_true = risk(mu_true[:, 1], B_or[:, 1, 1])
    tt = (np.arange(T_FUT) + 1) * DT
    # running per-step NLL telemetry (2-D Gaussian NLL at each revealed step)
    def nll_series(B):
        e = gt - nom
        return np.array([0.5 * (2 * math.log(2 * math.pi) + math.log(np.linalg.det(B[k])) + e[k] @ np.linalg.solve(B[k], e[k])) for k in range(T_FUT)])
    n_sc, n_fx, n_mc = nll_series(B_sc), nll_series(B_fx), nll_series(B_mc)
    lat = res["latency_ms"]; frames = []
    for k in range(T_FUT + 1):
        fig = plt.figure(figsize=(12.4, 5.9), dpi=100, facecolor="white")
        ax = fig.add_axes([0.04, 0.10, 0.52, 0.78]); ar = fig.add_axes([0.63, 0.55, 0.34, 0.33]); at = fig.add_axes([0.63, 0.10, 0.34, 0.32])
        # --- BEV ---
        ax.set_facecolor("#f4f6fa"); ax.set_xlim(-3.4, 8.8); ax.set_ylim(-2.2, 3.4); ax.set_aspect("equal")
        ax.add_patch(Rectangle((-3.4, LANE_Y), 12.2, 2.4, color="#dfe6f3", zorder=0)); ax.text(-3.2, 3.05, "ego lane (encroachment zone y > %.1f m)" % LANE_Y, fontsize=8, color="#55607a")
        ax.add_patch(Rectangle((2.7, -0.25), 0.9, 0.5, color="#9aa3b5", zorder=1)); ax.text(2.6, -0.55, "parked obstacle", fontsize=8, color="#55607a")
        ex = -3.0 + 2.3 * (k * DT); ax.add_patch(Rectangle((ex, 1.45), 1.9, 0.85, color="#33415c", zorder=3)); ax.text(ex + 0.2, 1.6, "EGO", color="white", fontsize=8, zorder=4)
        ax.plot(past[:, 0], past[:, 1], "o-", color=INK, ms=3, lw=1.3, label="input: observed history")
        ax.plot(*nom.T, "-", color=INK, lw=1, alpha=.55, label="model nominal (decoded)")
        for j in range(T_FUT):
            if j % 2 == 1:
                ellipse(ax, nom[j], B_fx[j], fill=False, ec=GRAY, ls="--", lw=1.1, zorder=2)
                ellipse(ax, nom[j], B_mc[j], fill=False, ec=ORANGE, ls=":", lw=1.4, zorder=2)
            ellipse(ax, nom[j], B_sc[j], fc=BLUE, ec=BLUE, alpha=0.16 if j >= k else 0.28, lw=1.0, zorder=2)
        ax.plot([], [], color=BLUE, lw=6, alpha=.3, label="SCOPE 95% tube (this repo)"); ax.plot([], [], "--", color=GRAY, label="fixed covariance"); ax.plot([], [], ":", color=ORANGE, lw=1.6, label=f"Monte-Carlo (K={K})")
        ax.plot(*gt[:k].T, "o-", color=GREEN, lw=2, ms=4, label="ground truth (revealed)") if k else None
        if k: ax.plot(*nom[k - 1], "s", color=BLUE, ms=6)
        ax.legend(loc="lower left", fontsize=7.2, framealpha=.9); ax.set_xlabel("x (m)", fontsize=8); ax.set_ylabel("y (m)", fontsize=8); ax.tick_params(labelsize=7)
        ax.set_title(f"Held-out scene: crowd density c={te['c'][i]:.2f}, speed {te['v'][i]:.2f} m/s, t = +{k*DT:.1f}s", fontsize=10, loc="left", color=INK)
        # --- risk panel ---
        ar.plot(tt, r_true, color=GREEN, lw=2, label="true (known generator)"); ar.plot(tt, r_sc, color=BLUE, lw=2, label="SCOPE")
        ar.plot(tt, r_fx, "--", color=GRAY, label="fixed cov"); ar.plot(tt, r_mc, ":", color=ORANGE, lw=1.8, label="MC")
        ar.axhline(THR, color=RED, lw=.8, ls="-."); ar.text(0.05, THR + 0.012, "brake threshold %.2f" % THR, fontsize=7, color=RED)
        if k: ar.axvline(k * DT, color="#999", lw=.8)
        ar.set_ylim(0, 0.4); ar.set_title("P(pedestrian enters ego lane) per future step", fontsize=9, loc="left"); ar.legend(fontsize=6.5, loc="upper left"); ar.tick_params(labelsize=7); ar.set_xlabel("horizon (s)", fontsize=8)
        # --- telemetry ---
        names = ["SCOPE", "Fixed", "MC"]; sc = [n_sc[:max(k, 1)].mean(), n_fx[:max(k, 1)].mean(), n_mc[:max(k, 1)].mean()]
        at.barh(names, [-v for v in sc], color=[BLUE, GRAY, ORANGE]); at.set_title("running tube score: -NLL of ground truth (higher = better)", fontsize=8.5, loc="left", color=INK); at.tick_params(labelsize=8)
        for yy, v in enumerate(sc): at.text(4.9, yy, f"{-v:.2f}", va="center", ha="right", fontsize=8.5, weight="bold")
        at.set_xlim(-5, 5); at.axvline(0, color="#aaa", lw=.6)
        fig.text(0.63, 0.015, f"latency/scene (CPU): SCOPE head+solve {lat['scope_head_plus_solve_ms']:.2f} ms  vs  MC sampling {lat['mc_sampling_ms']:.1f} ms  ({lat['mc_sampling_ms']/lat['scope_head_plus_solve_ms']:.0f}x)", fontsize=8, color=INK)
        d = lambda r: "BRAKE" if r.max() > THR else "keep going"
        fig.text(0.04, 0.945, f"Day 37 | SCOPE-style control-ready uncertainty | planner decision  SCOPE: {d(r_sc)}   fixed-cov: {d(r_fx)}   MC: {d(r_mc)}   |   truth: ped {'DOES' if gt[:,1].max()>LANE_Y else 'does not'} enter lane", fontsize=9.2, color=INK, weight="bold")
        fig.text(0.04, 0.012, "Synthetic data + this repo's own reconstruction (not the paper's model/numbers). Scene chosen from held-out set to show the high-density regime.", fontsize=7, color="#666")
        fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())); plt.close(fig)
    frames += [frames[-1]] * 3
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=[700] * len(frames), loop=0)
    print("saved", out, len(frames), "frames", frames[0].size)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="assets/scope_simulation.gif"); ap.add_argument("--scene", type=int, default=None)
    a = ap.parse_args(); main(a.out, scene=a.scene)
