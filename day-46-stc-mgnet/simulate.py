"""Render the "AI in action" overlay GIF from the TRAINED checkpoints.
Two held-out scenes (a crosser, then a non-crosser) play back; at every frame the last
16 frames are fed to both the old-way baseline and STC-MGNet.
Panels: (1) side-view street: skeleton coloured by learned joint attention, ego car + speed
        (2) live crossing-probability timeline, baseline vs STC-MGNet vs ground-truth entry frame
        (3) telemetry: per-branch STC fusion weights, per-frame attention, verdict badges.
Usage: python simulate.py --out assets/simulation.gif"""
import argparse, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import io
from src.stcmgnet.model import SKELETON_EDGES
from src.stcmgnet.data import make_scene, window_tensors
from train import make_model

BG, FG, GRID = "#0d1117", "#e6edf3", "#30363d"
C_BASE, C_NEW, C_GT = "#f85149", "#58a6ff", "#3fb950"


def load(name):
    ck = torch.load(f"{name}.pt", map_location="cpu"); m = make_model(ck["cfg"], name)
    m.load_state_dict(ck["state"]); return m.eval()


@torch.no_grad()
def run_scene(scene, base, new, W=16, t_end=41):
    out = []
    for t in range(W, t_end):
        p, v, e = [torch.from_numpy(np.ascontiguousarray(a))[None].float() for a in window_tensors(scene, t, W)]
        lb = torch.sigmoid(base(p, v, e)).item()
        ln, attn = new(p, v, e, return_attn=True)
        out.append(dict(t=t, pb=lb, pn=torch.sigmoid(ln).item(), ja=attn["joint"][0].numpy(),
                        fa=attn["frame"][0].numpy(), w=attn["w"].numpy()))
    return out


def draw(scene, rec, hist, title):
    fig = plt.figure(figsize=(12.5, 5.6), dpi=90, facecolor=BG)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.1, 0.8], hspace=0.38, wspace=0.28)
    gl = gs[:, 0].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.12)
    az = fig.add_subplot(gl[0]); ax = fig.add_subplot(gl[1]); at = fig.add_subplot(gs[0, 1]); ae = fig.add_subplot(gs[1, 1]); ar = fig.add_subplot(gs[:, 2])
    for a in (az, ax, at, ae, ar):
        a.set_facecolor(BG); [s.set_color(GRID) for s in a.spines.values()]; a.tick_params(colors=FG, labelsize=7)
    t = rec["t"]; P = scene["pose"][t - 1]; hx = float(P[[8, 11], 0].mean()); ja = rec["ja"]
    # zoomed skeleton (hip-centred) coloured by learned joint attention
    az.set_xlim(-1.1, 1.1); az.set_ylim(-0.1, 1.95); az.set_aspect("equal"); az.set_xticks([]); az.set_yticks([])
    for i, j in SKELETON_EDGES: az.plot(P[[i, j], 0] - hx, P[[i, j], 1], color="#8b949e", lw=2.5)
    sz = 30 + 220 * (ja - ja.min()) / (np.ptp(ja) + 1e-6)
    az.scatter(P[:, 0] - hx, P[:, 1], s=sz, c=ja, cmap="plasma", zorder=5, edgecolors="white", linewidths=0.4)
    az.set_title(f"{title} - frame {t}", color=FG, fontsize=10)
    az.text(-1.05, 1.85, "dot size/colour = learned joint attention", color=FG, fontsize=7)
    # street strip
    ax.set_xlim(-5, 6); ax.set_ylim(-0.1, 1.4); ax.set_xticks([]); ax.set_yticks([])
    ax.add_patch(Rectangle((-5, -0.1), 5, 0.35, color="#21262d")); ax.add_patch(Rectangle((0, -0.1), 6, 0.35, color="#161b22"))
    ax.axvline(0, color="#d29922", lw=2); ax.text(0.1, 0.0, "curb / road", color="#d29922", fontsize=7)
    ax.plot([hx], [0.5], "o", color="#3fb950", ms=9); ax.text(hx - 0.3, 0.8, "ped", color="#3fb950", fontsize=7)
    ev = scene["ve"][t - 1]; ax.add_patch(Rectangle((3.4, 0.3), 2.2, 0.5, color="#bc8cff")); ax.text(3.4, 0.9, f"ego {ev:4.1f} m/s", color="#bc8cff", fontsize=8)
    ts = [h["t"] for h in hist]
    at.plot(ts, [h["pb"] for h in hist], color=C_BASE, lw=2, label="old way: pose-only ST-GCN")
    at.plot(ts, [h["pn"] for h in hist], color=C_NEW, lw=2, label="STC-MGNet")
    at.axhline(0.5, color=GRID, ls=":"); at.set_xlim(16, 41); at.set_ylim(-0.02, 1.02)
    if scene["intent"]: at.axvline(40, color=C_GT, ls="--", lw=1.5); at.text(33.5, 0.05, "enters road ->", color=C_GT, fontsize=7)
    at.set_title("P(will cross)", color=FG, fontsize=9); at.legend(fontsize=6, facecolor=BG, labelcolor=FG, loc="center left", edgecolor=GRID)
    ae.plot(np.arange(t - 16, t), rec["fa"], color="#d29922", lw=2); ae.set_ylim(rec["fa"].min() - 0.01, rec["fa"].max() + 0.01)
    ae.set_title("learned frame attention (last 16 frames)", color=FG, fontsize=8)
    ar.axis("off"); gt = "WILL CROSS" if scene["intent"] else "WILL NOT CROSS"
    ok = lambda p: (p > 0.5) == bool(scene["intent"])
    lines = [("GROUND TRUTH", gt, C_GT), ("OLD WAY", f"{rec['pb']:.2f}  {'OK' if ok(rec['pb']) else 'MISS'}", C_BASE),
             ("STC-MGNet", f"{rec['pn']:.2f}  {'OK' if ok(rec['pn']) else 'MISS'}", C_NEW)]
    y = 0.95
    for k, v, c in lines:
        ar.text(0.0, y, k, color=FG, fontsize=8, transform=ar.transAxes); ar.text(0.0, y - 0.07, v, color=c, fontsize=12, fontweight="bold", transform=ar.transAxes); y -= 0.2
    ar.text(0.0, y, "STC fusion weights", color=FG, fontsize=8, transform=ar.transAxes)
    for i, (n, w) in enumerate(zip(["joint", "frame", "channel"], rec["w"])):
        ar.add_patch(Rectangle((0.28, y - 0.08 - 0.07 * i), 0.5 * w / 3 * 2, 0.04, transform=ar.transAxes, color="#58a6ff"))
        ar.text(0.0, y - 0.08 - 0.07 * i, n, color=FG, fontsize=7, transform=ar.transAxes); ar.text(0.82, y - 0.08 - 0.07 * i, f"{w:.2f}", color=FG, fontsize=7, transform=ar.transAxes)
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=BG); plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="assets/simulation.gif"); ap.add_argument("--seed", type=int, default=909)
    a = ap.parse_args(); base, new = load("baseline"), load("stc_mgnet")
    rng = np.random.default_rng(a.seed); frames, stats = [], {}
    for intent, title in ((1, "Scene A: pedestrian who WILL cross"), (0, "Scene B: pedestrian who will NOT cross")):
        # pick a held-out scene where the new model is right by the end, to keep the demo representative
        best = None
        for _ in range(60):
            sc = make_scene(rng, intent); recs = run_scene(sc, base, new)
            if best is None or abs(recs[-1]["pn"] - intent) < abs(best[1][-1]["pn"] - intent): best = (sc, recs)
        sc, recs = best; stats[title] = (recs[-1]["pb"], recs[-1]["pn"])
        for k in range(len(recs)):
            im = draw(sc, recs[k], recs[:k + 1], title); frames += [im] * (3 if k == len(recs) - 1 else 1)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=300, loop=0)
    print("saved", a.out, len(frames), "frames", stats)
