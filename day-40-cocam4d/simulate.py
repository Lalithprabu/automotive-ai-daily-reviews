"""Real-time-style synthetic simulation: GT vs 'old way' (stale late fusion) vs CoCam4D fusion.
Renders a GIF with BEV, ego pseudo-camera overlay, and live telemetry (error, latency, C-V2X bytes).
Usage: python simulate.py --ckpt checkpoints/cocam4d.pt --out cocam4d_simulation.gif
"""
import argparse
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from PIL import Image
from src.cocam4d import MotionUncertaintyNet, ModelConfig, fuse_agents, DOP_BYTES
from src.cocam4d.data import simulate_scene, sample_agents
from src.cocam4d.evaluate import agent_dops, score

GT, OLD, NEW = "#2ca02c", "#d62728", "#1f77b4"


def project(pt, cam, yaw, f=520.0, W=640, H=360):
    d = pt - cam; c, s = np.cos(-yaw), np.sin(-yaw)
    x = c * d[0] - s * d[1]; y = s * d[0] + c * d[1]; z = d[2]
    if x < 3.0: return None
    return W / 2 - f * y / x, H / 2 - f * z / x, f / x


def run(a):
    ck = torch.load(a.ckpt, map_location="cpu"); cfg = ck["cfg"]; d = cfg["data"]
    model = MotionUncertaintyNet(ModelConfig(**cfg["model"])); model.load_state_dict(ck["model"]); model.eval()
    rng = np.random.default_rng(a.seed); T, dt = d["seq_len"], d["dt"]
    nf = a.frames
    P, V, _ = simulate_scene(rng, d["n_objects"], T + nf, dt, extent=35.0)
    A = sample_agents(rng, d["n_agents"], extent=25.0)
    yaw = np.arctan2(-A[0, 1], -A[0, 0])
    errs_old, errs_new, lats, bps, frames = [], [], [], [], []
    for f in range(nf):
        now = T - 1 + f
        lag = int(round(5 * f / max(nf - 1, 1)))              # latency ramps 0 -> 500 ms
        per, deltas, nbytes = [], [], 0
        for k, ag in enumerate(A):
            end = now if k == 0 else max(now - lag, T - 1)
            r2 = np.random.default_rng(a.seed * 1000 + f * 10 + k)
            dops, _ = agent_dops(model, r2, ag, P, V, end, d, k)
            per.append(dops); deltas.append(0.0 if k == 0 else lag * dt)
            nbytes += len(dops) * DOP_BYTES if k > 0 else 0
        r3 = np.random.default_rng(a.seed * 1000 + f * 10)    # old way: fixed-cov, no compensation
        per_old = [agent_dops(model, np.random.default_rng(a.seed * 1000 + f * 10 + k), ag,
                              P, V, now if k == 0 else max(now - lag, T - 1), d, k, learned=False)[0]
                   for k, ag in enumerate(A)]
        old = fuse_agents(per_old, deltas, compensate=False, bayesian=False, pose_sigma=d["pose_sigma"])
        new = fuse_agents(per, deltas, compensate=True, pose_sigma=d["pose_sigma"])
        eo, ro, _ = score(old, P[now]); en, rn, _ = score(new, P[now])
        errs_old.append(eo); errs_new.append(en); lats.append(lag * dt * 1000); bps.append(nbytes / dt)
        fig = plt.figure(figsize=(14, 5.2), dpi=80)
        ax = fig.add_axes([0.03, 0.08, 0.30, 0.82]); cx = fig.add_axes([0.36, 0.52, 0.34, 0.38])
        tx = fig.add_axes([0.38, 0.08, 0.28, 0.32]); lx = fig.add_axes([0.74, 0.08, 0.23, 0.82])
        ax.set_title("BEV: GT vs stale late-fusion vs CoCam4D", fontsize=10)
        for i in range(P.shape[1]):
            ax.plot(P[now - 5:now + 1, i, 0], P[now - 5:now + 1, i, 1], color=GT, lw=1, alpha=.5)
        ax.scatter(P[now, :, 0], P[now, :, 1], s=70, facecolors="none", edgecolors=GT, lw=2, label="Ground truth")
        ax.scatter([t.mu[0] for t in old], [t.mu[1] for t in old], marker="x", c=OLD, s=55, label="Old: stale late fusion")
        for t in new:
            sd = np.sqrt(t.pos_var[:2]); ax.add_patch(Ellipse(t.mu[:2], 4 * sd[0], 4 * sd[1], fill=False, ec=NEW, lw=1))
        ax.scatter([t.mu[0] for t in new], [t.mu[1] for t in new], marker="+", c=NEW, s=70, label="CoCam4D (+95% ellipse)")
        ax.scatter(A[:, 0], A[:, 1], marker="^", c="k", s=60); ax.annotate("ego", A[0, :2], fontsize=8)
        ax.set_xlim(-50, 50); ax.set_ylim(-50, 50); ax.set_aspect("equal"); ax.legend(fontsize=7, loc="lower right")
        cx.set_title("Ego camera view (overlay)", fontsize=10); cx.set_xlim(0, 640); cx.set_ylim(360, 0); cx.set_facecolor("#cfd8e3")
        cx.axhspan(180, 360, color="#8a8f98"); cx.set_xticks([]); cx.set_yticks([])
        for pts, col, ls in ((list(P[now]), GT, "-"), ([t.mu for t in old], OLD, "--"), ([t.mu for t in new], NEW, "-")):
            for p in pts:
                q = project(np.asarray(p), A[0], yaw)
                if q: u, v, sc = q; cx.add_patch(Rectangle((u - .9 * sc, v - .8 * sc), 1.8 * sc, 1.6 * sc, fill=False, ec=col, ls=ls, lw=2))
        tx.plot(errs_old, c=OLD, label="Old way"); tx.plot(errs_new, c=NEW, label="CoCam4D")
        tx.set_title("Position error vs GT (m)", fontsize=9); tx.set_xlim(0, nf); tx.set_ylim(0, max(4, np.nanmax(errs_old + [1]) * 1.1)); tx.legend(fontsize=7)
        lx.axis("off")
        lx.text(0, .95, f"Frame {f+1}/{nf}", fontsize=12, weight="bold", va="top")
        lx.text(0, .82, f"V2X latency:  {lag*dt*1000:.0f} ms", fontsize=11, va="top")
        lx.text(0, .70, f"C-V2X payload: {nbytes} B/msg-set\n  (35 B per DOP, {bps[-1]/1000:.2f} kB/s)", fontsize=10, va="top")
        lx.text(0, .50, f"Old way err:   {eo:.2f} m  (rec@2m {ro:.2f})", color=OLD, fontsize=11, va="top")
        lx.text(0, .38, f"CoCam4D err:  {en:.2f} m  (rec@2m {rn:.2f})", color=NEW, fontsize=11, va="top")
        lx.text(0, .22, "Synthetic scene — reconstruction,\nnot the paper's pipeline/numbers.", fontsize=8, va="top", style="italic")
        fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())); plt.close(fig)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=a.ms, loop=0, optimize=False)
    print(f"wrote {a.out}: {len(frames)} frames; mean err old {np.nanmean(errs_old):.2f} m vs CoCam4D {np.nanmean(errs_new):.2f} m; "
          f"at 500ms: old {errs_old[-1]:.2f} vs new {errs_new[-1]:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", default="checkpoints/cocam4d.pt")
    p.add_argument("--out", default="cocam4d_simulation.gif"); p.add_argument("--frames", type=int, default=30)
    p.add_argument("--seed", type=int, default=11); p.add_argument("--ms", type=int, default=250)
    run(p.parse_args())
