"""Synthetic 'AI in action' simulation: ground truth vs AR baseline vs VFlash drafter.
Usage: python simulate.py [--out assets/vflash_sim.mp4] [--frames 90]"""
import argparse, time, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from vflash.model import VFlashDrafter, ARBaseline
from vflash.train import Enc
from vflash.data import make_traj, ctx_from_hist, T

BG, FG = "#0b0f17", "#d7dde8"
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="assets/vflash_sim.mp4"); ap.add_argument("--frames", type=int, default=90)
a = ap.parse_args()

def load(cls, key):
    ck = torch.load(f"assets/{key}.pt"); m = cls(); m.load_state_dict(ck["m"]); e = Enc(); e.load_state_dict(ck["e"])
    return m.eval(), e.eval()
fm, fe = load(VFlashDrafter, "flash"); am, ae = load(ARBaseline, "ar")

rng = np.random.default_rng(7)
scenes = [make_traj(rng, k, 50)[0] for k in (0, 1, 2)]
per = a.frames // 3
fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.3], hspace=.25, left=.06, right=.97, top=.92, bottom=.08)
ax = fig.add_subplot(gs[0, :]); axe = fig.add_subplot(gs[1, 0]); axl = fig.add_subplot(gs[1, 1])
for x in (ax, axe, axl):
    x.set_facecolor(BG); x.tick_params(colors=FG); [s.set_color("#33415c") for s in x.spines.values()]
errs = {"ar": [], "vf": []}; lat = {"ar": [], "vf": []}; names = ["straight", "left curve", "hard braking"]

@torch.no_grad()
def step(f):
    si = min(f // per, 2); k = f % per
    path = scenes[si]; i0 = int(k / per * (len(path) - 8 - T)) 
    hist, gt = path[i0:i0 + 8], path[i0 + 8:i0 + 8 + T]
    c = torch.tensor(ctx_from_hist(hist))[None]
    fm(fe(c), "traj2d", T); am(ae(c), length=T)  # warm-up (untimed)
    t = time.perf_counter(); pv = fm(fe(c), "traj2d", T)[0].numpy() + hist[-1]; lv = time.perf_counter() - t
    t = time.perf_counter(); pa = am(ae(c), length=T)[0].numpy() + hist[-1]; la = time.perf_counter() - t
    return si, k, path, i0, hist, gt, pv, pa, lv, la

def draw(f):
    si, k, path, i0, hist, gt, pv, pa, lv, la = step(f)
    if k == 0: errs["ar"].clear(); errs["vf"].clear(); lat["ar"].clear(); lat["vf"].clear()
    errs["vf"].append(np.linalg.norm(pv - gt, axis=1).mean()); errs["ar"].append(np.linalg.norm(pa - gt, axis=1).mean())
    lat["vf"].append(lv * 1e3); lat["ar"].append(la * 1e3)
    ax.clear(); ax.set_facecolor(BG); ax.set_aspect("equal"); ax.grid(color="#1c2433")
    ax.plot(*path.T, color="#2a3550", lw=8, alpha=.6, solid_capstyle="round")
    ax.plot(*hist.T, "o-", color="#9aa5b8", ms=4, label="observed history")
    ax.plot(*gt.T, "-", color="#34d399", lw=3, label="ground truth")
    ax.plot(*pa.T, "--", color="#f87171", lw=2, label=f"AR baseline ({T} passes)")
    ax.plot(*pv.T, "-", color="#60a5fa", lw=2.5, label="VFlash drafter (1 pass)")
    ax.add_patch(plt.Rectangle((hist[-1, 0] - .05, hist[-1, 1] - .025), .1, .05, ec="white", fc="#a78bfa", zorder=5))
    m = .6; ax.set_xlim(hist[-1, 0] - m, hist[-1, 0] + 1.6); ax.set_ylim(hist[-1, 1] - m, hist[-1, 1] + m)
    ax.set_title(f"Scene: {names[si]}   |   ADE  AR {errs['ar'][-1]:.3f}   VFlash {errs['vf'][-1]:.3f}   |   latency  AR {la*1e3:.1f} ms   VFlash {lv*1e3:.1f} ms",
                 color=FG, fontsize=11, loc="left"); ax.tick_params(colors=FG)
    ax.legend(loc="upper right", facecolor=BG, labelcolor=FG, edgecolor="#33415c", fontsize=9)
    for x, d, ttl in ((axe, errs, "Trajectory error (ADE)"), (axl, lat, "Decode latency (ms)")):
        x.clear(); x.set_facecolor(BG); x.plot(d["ar"], color="#f87171"); x.plot(d["vf"], color="#60a5fa")
        x.set_title(ttl, color=FG, fontsize=10, loc="left"); x.tick_params(colors=FG); x.grid(color="#1c2433")
    fig.suptitle("VFlash-style Flash Primitive Decoding - synthetic toy (not the paper's model/data)", color=FG, x=.06, ha="left", fontsize=12)

anim = FuncAnimation(fig, draw, frames=per * 3, interval=83)
w = PillowWriter(fps=12) if a.out.endswith(".gif") else FFMpegWriter(fps=12, bitrate=2500)
anim.save(a.out, writer=w, dpi=100); print("saved", a.out)
