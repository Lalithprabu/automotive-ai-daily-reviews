"""Real-time-style synthetic simulation: input vs ground truth vs model prediction overlay.

Renders a top-down off-road scene (friction map = mud patches, hill contours) and animates a
24-step (2.4 s) forecast: dark dotted trail = INPUT history, green = GROUND TRUTH, grey dashed =
physics-only, orange = point-conditioned model, blue = SpatialBEV. The cyan box is the BEV window
SpatialBEV queries at its own predicted state. Right: live position-error curves and the left/right
wheel-friction telemetry (the hidden cause of the yaw pull).

The scene is the MEDIAN hard test episode (rank by physics-baseline FDE within the hardest 20%),
not a hand-picked best case.   python simulate.py --out assets/spatialbev_demo
"""
import argparse, json
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Polygon

from spatialbev.models import SpatialBEVDynamics, physics_rollout, WIN_F, WIN_L
from spatialbev.terrain import E, make_terrain, sample_map, ego_to_world, _WHEELS_F, _WHEELS_L
from train import HIST, H, make_test, evaluate

COL = dict(truth="#1a9850", phys="#7f7f7f", point="#f28e2b", spatial="#1f5fd1", hist="#222222")


def car_poly(x, y, yaw, L=3.4, W=1.8):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2]])
    return pts @ np.array([[c, s], [-s, c]]) + [x, y]


def window_poly(x, y, yaw):
    (f0, f1), (l0, l1) = WIN_F, WIN_L
    pts = np.array([[f0, l0], [f1, l0], [f1, l1], [f0, l1]])
    c, s = np.cos(yaw), np.sin(yaw)
    return pts @ np.array([[c, s], [-s, c]]) + [x, y]


def main(a):
    torch.manual_seed(0)
    bank = make_terrain(48, 2)
    test = make_test(bank, 1000, 123)
    st, ac, idx = test
    _, perr = evaluate(None, bank, test)
    order = torch.argsort(perr[:, -1], descending=True)[:200]        # hardest 20%
    ep = int(order[len(order) // 2])                                  # median of the hard set
    models = {}
    for k in ("point", "spatial"):
        m = SpatialBEVDynamics(k); m.load_state_dict(torch.load(f"checkpoints/{k}.pt")); m.eval(); models[k] = m
    terr = bank[idx[ep]][None]
    s0, acts = st[ep:ep + 1, HIST], ac[ep:ep + 1, HIST:]
    gt = st[ep, HIST:].numpy()                                        # includes the start state
    hist = st[ep, :HIST + 1].numpy()
    with torch.no_grad():
        pred = {"phys": physics_rollout(s0, acts)[0].numpy(),
                "point": models["point"].rollout(s0, acts, terr)[0].numpy(),
                "spatial": models["spatial"].rollout(s0, acts, terr)[0].numpy()}
        mu_lr = []
        for t in range(H):
            w = sample_map(terr, ego_to_world(st[ep:ep + 1, HIST + 1 + t], _WHEELS_F, _WHEELS_L))[0, 1]
            mu_lr.append((float(w[[0, 2]].mean()), float(w[[1, 3]].mean())))
    mu_lr = np.array(mu_lr)
    P = {k: np.vstack([gt[:1], v]) for k, v in pred.items()}          # prepend start state
    err = {k: np.linalg.norm(v[1:, :2] - gt[1:, :2], axis=1) for k, v in P.items()}

    fig = plt.figure(figsize=(12.4, 5.9), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1.35, 1], hspace=0.38, wspace=0.16,
                          left=0.04, right=0.98, top=0.835, bottom=0.09)
    ax, axe, axm = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    mu_map = terr[0, 1].numpy(); h_map = terr[0, 0].numpy()
    ext = [-E, E, -E, E]
    ax.imshow(mu_map, origin="lower", extent=ext, cmap="YlOrBr_r", vmin=0.25, vmax=1.0, alpha=0.85)
    X = np.linspace(-E, E, mu_map.shape[1]); ax.contour(X, X, h_map, levels=10, colors="k", linewidths=0.3, alpha=0.35)
    allp = np.vstack([gt[:, :2]] + [v[:, :2] for v in P.values()] + [hist[:, :2]])
    cx, cy = allp.mean(0); hw = max(np.ptp(allp[:, 0]) / 2 + 9, (np.ptp(allp[:, 1]) / 2 + 9) * 1.4)
    hh = hw / 1.4
    hw, hh = min(hw, E - 0.5), min(hh, E - 0.5)
    cx, cy = np.clip(cx, -E + hw, E - hw), np.clip(cy, -E + hh, E - hh)
    ax.set_xlim(cx - hw, cx + hw); ax.set_ylim(cy - hh, cy + hh)                       # y up
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Top-down off-road scene  (brown = low-friction mud)", fontsize=10)
    ax.plot(hist[:, 0], hist[:, 1], ":", color=COL["hist"], lw=2.4, label="INPUT: observed history")
    lines = {}
    lines["truth"], = ax.plot([], [], "-", lw=7, color=COL["truth"], alpha=0.55, label="GROUND TRUTH", zorder=2)
    lines["phys"], = ax.plot([], [], "--", lw=1.8, color=COL["phys"], label="physics only")
    lines["point"], = ax.plot([], [], "-", lw=1.8, color=COL["point"], label="point-conditioned")
    lines["spatial"], = ax.plot([], [], "-o", lw=2.4, ms=3, color=COL["spatial"], label="SpatialBEV")
    tips = {k: ax.plot([], [], "o", ms=8, mfc=COL[k], mec="w", mew=1.2, zorder=7)[0] for k in ("phys", "point", "spatial")}
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    win = Polygon(window_poly(*P["spatial"][0, :3]), closed=True, fc="#00bcd4", ec="#00838f", alpha=0.25, lw=1.5)
    car = Polygon(car_poly(*gt[0, :3]), closed=True, fc="#111", ec="w", lw=1.2, zorder=5)
    ax.add_patch(win); ax.add_patch(car)
    axe.cla(); axe.set_xlim(0, H * 0.1); axe.set_ylim(0, max(1.0, 1.1 * max(e.max() for e in err.values())))
    axe.set_title("Position error vs ground truth (m)", fontsize=10); axe.set_xlabel("forecast time (s)", fontsize=8)
    axe.grid(alpha=0.3)
    el = {k: axe.plot([], [], color=COL[k], lw=2.2, ls="--" if k == "phys" else "-")[0] for k in err}
    tt = (np.arange(H) + 1) * 0.1
    axm.set_xlim(0, H * 0.1); axm.set_ylim(0.2, 1.05); axm.grid(alpha=0.3)
    axm.set_title("Wheel-contact friction under the car (hidden cause of the yaw pull)", fontsize=9)
    axm.set_xlabel("forecast time (s)", fontsize=8)
    ml = axm.plot([], [], color="#8e44ad", lw=2, label="left wheels")[0]
    mr = axm.plot([], [], color="#c0392b", lw=2, label="right wheels")[0]
    axm.legend(fontsize=7, loc="lower left")
    head = fig.text(0.04, 0.93, "", fontsize=12, weight="bold", va="center")
    sub = fig.text(0.04, 0.885, "", fontsize=9, va="center", family="monospace")
    fig.text(0.98, 0.935, "Synthetic demo - SpatialBEV-style re-implementation (arXiv 2610.22900), not the paper's data",
             ha="right", fontsize=8, color="#555")
    n_hold = 8

    def draw(f):
        t = min(f, H)
        for k in ("truth", "phys", "point", "spatial"):
            src = gt if k == "truth" else P[k]
            lines[k].set_data(src[:t + 1, 0], src[:t + 1, 1])
        for k, tp in tips.items():
            tp.set_data([P[k][t, 0]], [P[k][t, 1]])
        car.set_xy(car_poly(*gt[t, :3])); win.set_xy(window_poly(*P["spatial"][t, :3]))
        for k in el:
            el[k].set_data(tt[:t], err[k][:t])
        ml.set_data(tt[:t], mu_lr[:t, 0]); mr.set_data(tt[:t], mu_lr[:t, 1])
        head.set_text("AI in action: forecasting where an off-road vehicle will be in 2.4 s")
        if t > 0:
            sub.set_text(f"t=+{t * 0.1:.1f}s  error (m)  physics {err['phys'][t - 1]:.2f} | point {err['point'][t - 1]:.2f} | "
                         f"SpatialBEV {err['spatial'][t - 1]:.2f}")
        else:
            sub.set_text("t=+0.0s  input history shown; predictions reveal step by step")
        return []

    anim = FuncAnimation(fig, draw, frames=H + 1 + n_hold, interval=170, blit=False)
    anim.save(a.out + ".gif", writer=PillowWriter(fps=6))
    anim.save(a.out + ".mp4", writer=FFMpegWriter(fps=6, bitrate=2400))
    json.dump({"episode": ep, "final_error_m": {k: float(v[-1]) for k, v in err.items()}}, open(a.out + ".json", "w"))
    print("episode", ep, {k: round(float(v[-1]), 2) for k, v in err.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="assets/spatialbev_demo")
    main(ap.parse_args())
