"""Render the 'AI in action' overlay as a GIF from the REAL trained checkpoints.

Left   : INPUT — current LiDAR-like sweep (white), the previous two sweeps ghosted (memory ages),
         per-vehicle outline at time t.
Middle : GROUND TRUTH (green) vs OLD WAY K=1 (orange) vs MESSENGER K=3 (blue): where each moving point
         will be h frames (h x 0.5 s) ahead, with the displacement error segments drawn.
Right  : live telemetry — extrapolation EPE vs horizon for both models, and the mean reweighting
         gate per memory age on moving vs static points.
Usage: python simulate.py [--out assets/messenger_simulation.gif]"""
import argparse, io, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from messenger import rollout, SceneFlowDataset, collate
from train import Pipeline

BG, FG, GT, OLD, NEW = "#0d1117", "#c9d1d9", "#3fb950", "#f0883e", "#58a6ff"


def load(path):
    ck = torch.load(path, map_location="cpu")
    c = ck["cfg"]["model"]
    p = Pipeline(c["d"], ck["K"], c["topk"], ck["use_gate"]); p.load_state_dict(ck["state"]); p.eval()
    return p


def pick_scene(cfg):
    dc = cfg["data"]
    ds = SceneFlowDataset(dc["n_test"], cfg["seed"] + 3, T_obs=dc["T_obs"], H=dc["H"], N=dc["N"])
    for i, it in enumerate(ds.items):                      # non-outcome rule: first scene with 3 vehicles
        if it["M"] == 3: return i, ds
    return 0, ds


@torch.no_grad()
def predict(pipe, b, t, H=4):
    fl, hs = pipe.observe(b["pts"])
    outs = rollout(pipe.fc, b["pts"][:, t], pipe.entries_at(b["pts"], fl, hs, t), H)
    return [o[0][0].numpy() for o in outs], [o[2][:, 0].numpy() for o in outs]


def style(ax, title, lim):
    ax.set_facecolor(BG); ax.set_title(title, color=FG, fontsize=11, loc="left")
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal"); ax.tick_params(colors="#6e7681", labelsize=7)
    for s in ax.spines.values(): s.set_color("#30363d")
    ax.grid(color="#21262d", lw=0.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="assets/messenger_simulation.gif")
    ap.add_argument("--old", default="old_way_K1.pt"); ap.add_argument("--new", default="messenger_K3.pt")
    a = ap.parse_args()
    old, new = load(a.old), load(a.new)
    cfg = torch.load(a.new, map_location="cpu")["cfg"]
    i, ds = pick_scene(cfg)
    b = collate([ds[i]]); P = b["pts"][0].numpy(); D = b["disp"][0].numpy(); L = b["lab"][0].numpy()
    pred = {t: dict(old=predict(old, b, t)[0], new=predict(new, b, t)) for t in (3, 4, 5)}
    mov_all = np.concatenate([P[5][L[5] >= 0] + D[5][L[5] >= 0][:, 3]] + [P[5][L[5] >= 0]])
    lim = ((mov_all[:, 0].min() - 8, mov_all[:, 0].max() + 8), (mov_all[:, 1].min() - 8, mov_all[:, 1].max() + 8))
    epe = {t: {k: [] for k in ("old", "new")} for t in (3, 4, 5)}
    for t in (3, 4, 5):
        m = L[t] >= 0
        for h in range(4):
            gt = D[t][:, h]
            epe[t]["old"].append(np.linalg.norm(pred[t]["old"][h] - gt, axis=1)[m].mean())
            epe[t]["new"].append(np.linalg.norm(pred[t]["new"][0][h] - gt, axis=1)[m].mean())
    frames, durs = [], []
    for t in (3, 4, 5):
        m = L[t] >= 0
        for h in range(4):
            fig = plt.figure(figsize=(16, 5.6), dpi=70, facecolor=BG)
            ax1, ax2 = fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2)
            ax3, ax4 = fig.add_axes([0.70, 0.56, 0.27, 0.32]), fig.add_axes([0.70, 0.10, 0.27, 0.28])
            style(ax1, f"INPUT  sweep t={t}  (+ {min(3,t)} memory frames)", lim)
            for age, al in ((2, 0.18), (1, 0.35)):
                if t - age >= 0: ax1.scatter(*P[t - age].T, s=5, c=FG, alpha=al, lw=0)
            ax1.scatter(*P[t].T, s=9, c="white", lw=0); ax1.scatter(*P[t][m].T, s=12, c="#a371f7", lw=0)
            style(ax2, f"PREDICTION overlay  horizon +{(h+1)*0.5:.1f}s", lim)
            gtp, op, npd = P[t] + D[t][:, h], P[t] + pred[t]["old"][h], P[t] + pred[t]["new"][0][h]
            ax2.scatter(*P[t][m].T, s=5, c="#6e7681", lw=0)
            for q, c, lab in ((gtp, GT, "ground truth"), (op, OLD, "old way (K=1)"), (npd, NEW, "MESSENGER (K=3)")):
                ax2.scatter(*q[m].T, s=14, c=c, lw=0, label=lab, alpha=0.9)
            for k in np.where(m)[0][::2]:
                ax2.plot([gtp[k, 0], op[k, 0]], [gtp[k, 1], op[k, 1]], c=OLD, lw=0.5, alpha=0.5)
                ax2.plot([gtp[k, 0], npd[k, 0]], [gtp[k, 1], npd[k, 1]], c=NEW, lw=0.5, alpha=0.6)
            ax2.legend(loc="upper right", fontsize=8, facecolor=BG, labelcolor=FG, edgecolor="#30363d")
            style(ax3, "", ((0.8, 4.2), (0, max(epe[t]["old"]) * 1.15)))
            ax3.set_aspect("auto"); ax3.set_title("Extrapolation error (m), moving points", color=FG, fontsize=10, loc="left")
            hz = np.arange(1, 5)
            ax3.plot(hz, epe[t]["old"], c=OLD, marker="o", label="old way"); ax3.plot(hz, epe[t]["new"], c=NEW, marker="o", label="MESSENGER")
            ax3.axvline(h + 1, c="#8b949e", ls=":"); ax3.set_xticks(hz); ax3.legend(fontsize=8, facecolor=BG, labelcolor=FG, edgecolor="#30363d")
            gm = pred[t]["new"][1][0]                        # (K,N) gates at first rollout step
            ax4.set_facecolor(BG); ax4.set_title("Memory gate g (mean) by age", color=FG, fontsize=10, loc="left")
            xs = np.arange(3); w = 0.38
            ax4.bar(xs - w / 2, [gm[k][m].mean() for k in range(3)], w, color="#a371f7", label="moving")
            ax4.bar(xs + w / 2, [gm[k][~m].mean() for k in range(3)], w, color="#6e7681", label="static")
            ax4.set_xticks(xs); ax4.set_xticklabels(["age 1", "age 2", "age 3"], color=FG, fontsize=8); ax4.set_ylim(0, 1)
            ax4.tick_params(colors="#6e7681", labelsize=7); ax4.legend(fontsize=8, facecolor=BG, labelcolor=FG, edgecolor="#30363d")
            for s in ax4.spines.values(): s.set_color("#30363d")
            fig.suptitle("MESSENGER-style memory scene-flow forecasting — this repo's reconstruction on SYNTHETIC data (not the paper's)",
                         color=FG, fontsize=11, x=0.01, ha="left")
            buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=BG); plt.close(fig)
            frames.append(Image.open(io.BytesIO(buf.getvalue())).convert("P", palette=Image.ADAPTIVE)); durs.append(900 if h < 3 else 1600)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=False)
    print(a.out, len(frames), "frames; scene", i)
    print("EPE moving (old/new) at t=5:", np.round(epe[5]["old"], 2), np.round(epe[5]["new"], 2))


if __name__ == "__main__":
    main()
