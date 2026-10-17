"""Renders the 'AI in action' overlay GIF from the REAL trained checkpoints:
 left  : BEV  - map, history, GT future (green), frozen-backbone proposals (blue), future-graph nodes/edges,
         sequential(causal) refiner (orange dashed), FutureLift refined (red)
 middle: synthetic forward 'dashcam' (pinhole ground-plane projection) with tracking box on the lead vehicle
 right : live telemetry (minFDE bars per model, running scene error, test-set minADE)
"""
import argparse, json, yaml, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from futurelift.data import JunctionDataset, T_F, N_APP, N_BR
from futurelift.backbone import MapBlindBackbone
from futurelift.model import FutureLift

BLUE, RED, GRN, ORG, GRY = "#3b7ddd", "#e0245e", "#1faa59", "#f08c00", "#888888"


def proj(pts, f=150, cx=200, cy=100, h=1.4):
    """Ground-plane pinhole projection. pts (N,2): x forward, y left -> (u,v), valid mask."""
    x, y = pts[:, 0], pts[:, 1]; ok = x > 3.0
    xs = np.where(ok, x, 1.0)
    return np.stack([cx - f * y / xs, cy + f * h / xs], 1), ok


def draw_cam(ax, scene, polys, lead):
    ax.set_xlim(0, 400); ax.set_ylim(240, 0); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 400, 100, color="#cfe3f5")); ax.add_patch(plt.Rectangle((0, 100), 400, 140, color="#6b6f73"))
    for pts, col, lw, ls in polys:
        uv, ok = proj(pts)
        uv = uv.copy(); uv[~ok] = np.nan
        ax.plot(uv[:, 0], uv[:, 1], color=col, lw=lw, ls=ls)
    if lead is not None and lead[0] > 3:
        x, y = lead; u, v = proj(np.array([[x, y]]))[0][0]
        w = 150 * 0.95 / x; ht = 150 * 1.5 / x
        ax.add_patch(plt.Rectangle((u - w, v - ht), 2 * w, ht, fill=True, color="#444", alpha=0.85))
        ax.add_patch(plt.Rectangle((u - w - 3, v - ht - 3), 2 * w + 6, ht + 6, fill=False, ec="#ffd43b", lw=1.8))
        ax.text(u - w, v - ht - 8, f"TRK lead {x:4.1f} m", color="#ffd43b", fontsize=7, weight="bold")
    ax.set_title("Simulated forward camera", fontsize=9)


def main(cfg_path, out, n_scenes):
    cfg = yaml.safe_load(open(cfg_path)); rc = cfg["refiner"]
    bb = MapBlindBackbone(cfg["backbone"]["K"], cfg["backbone"]["hidden"]); bb.load_state_dict(torch.load("checkpoints/backbone.pt")); bb.eval()
    mods = {}
    for v in ("bidirectional", "causal"):
        m = FutureLift(rc["d"], rc["heads"], rc["layers"], rc["R"], v); m.load_state_dict(torch.load(f"checkpoints/{v}.pt")); m.eval(); mods[v] = m
    res = json.load(open("checkpoints/results.json"))
    ds = JunctionDataset(cfg["data"]["n_test"], 3)
    with torch.no_grad():
        P, L = bb(ds.batch(torch.arange(len(ds))))
        full = ds.batch(torch.arange(len(ds)))
        out_bi = mods["bidirectional"](full, P, L)[0]; out_ca = mods["causal"](full, P, L)[0]
    fde = lambda T: (T - ds.gt[:, None]).norm(dim=-1)[..., -1].min(1).values
    f0, f1, f2 = fde(P), fde(out_ca), fde(out_bi)
    meta = ds.meta.numpy(); gain = (f0 - f2).numpy()
    # demo selection: one scene per branch type (straight w/ lead, left, right), large-but-typical gain (median of top quartile)
    picks = []
    for br in (0, 1, 2)[:n_scenes]:
        idx = np.where((meta[:, 2] == br) & ((meta[:, 4] == 1) if br == 0 else True))[0]
        idx = idx[np.argsort(-gain[idx])]; picks.append(int(idx[len(idx) // 8]))
    frames, run_b, run_f = [], [], []
    phases = ["INPUT: frozen backbone proposals"] * 2 + ["LIFT: proposals -> future graph (R=3 nodes/mode)"] * 3 + \
             ["REFINE: one pass, all segments updated together"] * 6 + ["RESULT"] * 3
    for si, i in enumerate(picks):
        dj, R, br, v0, lp, lx, vl = meta[i]
        hist = ds.hist[i, 0, :, :2].numpy(); gt = ds.gt[i].numpy(); mp = ds.map[i].numpy()
        pr, rb, rcz = P[i].numpy(), out_bi[i].numpy(), out_ca[i].numpy()
        best = int(np.argmin(np.linalg.norm(rb[:, -1] - gt[-1], axis=1)))
        run_b.append(float(f0[i])); run_f.append(float(f2[i]))
        for fi, ph in enumerate(phases):
            fig = plt.figure(figsize=(14, 5.2), dpi=80)
            ax = fig.add_axes([0.03, 0.08, 0.40, 0.82]); axc = fig.add_axes([0.45, 0.30, 0.27, 0.50]); axt = fig.add_axes([0.77, 0.12, 0.21, 0.72])
            ax.set_aspect("equal"); ax.set_xlim(-25, 85); ax.set_ylim(-32, 32); ax.set_facecolor("#f4f4f4")
            for s in range(4):
                seg = mp[:N_APP] if s == 0 else mp[N_APP + (s - 1) * N_BR: N_APP + s * N_BR]
                ax.plot(seg[:, 0], seg[:, 1], color="#bbb", lw=7, solid_capstyle="round", zorder=1)
            ax.plot(hist[:, 0], hist[:, 1], "k-", lw=2.5, zorder=4); ax.plot(0, 0, "ks", ms=9, zorder=5)
            if lp:
                lxp = lx; ax.add_patch(plt.Rectangle((lxp - 2.2, -0.9), 4.4, 1.8, color="#555", zorder=4))
                ax.text(lxp, 2.5, "lead", fontsize=7, ha="center")
            ax.plot(gt[:, 0], gt[:, 1], color=GRN, lw=3, zorder=6, label="ground truth")
            alpha = 0.0 if fi < 5 else min(1.0, (fi - 4) / 5.0)
            cur = pr + alpha * (rb - pr)
            for k in range(pr.shape[0]):
                ax.plot(cur[k, :, 0], cur[k, :, 1], color=BLUE if alpha < 1 else RED, lw=2.4 if (alpha >= 1 and k == best) else 1.0,
                        alpha=0.9 if (alpha >= 1 and k == best) else 0.45, zorder=5)
            if 2 <= fi < 5:      # lifted future graph: R=3 nodes per mode + bidirectional temporal edges
                for k in range(pr.shape[0]):
                    nodes = pr[k, [3, 7, 11]]
                    ax.scatter(nodes[:, 0], nodes[:, 1], s=34, c="white", ec="#222", zorder=7)
                    for a, b_ in ((0, 1), (1, 2)):
                        ax.annotate("", xy=nodes[b_], xytext=nodes[a], arrowprops=dict(arrowstyle="<->", color="#222", lw=0.9), zorder=7)
                    j = int(np.argmin(np.linalg.norm(mp[:, :2] - nodes[2], axis=1)))
                    ax.plot([nodes[2, 0], mp[j, 0]], [nodes[2, 1], mp[j, 1]], ":", color="#999", lw=0.8, zorder=3)
            if fi >= 11:
                ax.plot(rcz[best, :, 0], rcz[best, :, 1], color=ORG, lw=2, ls="--", zorder=6)
            ax.set_title(f"Scene {si+1}/{len(picks)} | {['straight','left','right'][int(br)]} | {ph}", fontsize=9.5, loc="left")
            ax.text(-24, -30, "black=history  green=GT  blue=frozen proposals  red=FutureLift  orange(dashed)=sequential refiner", fontsize=6.8)
            lead_xy = (lx, 0.0) if lp else None
            polys = [(mp[:N_APP, :2], "white", 3, "-"), (mp[N_APP + (int(br)) * N_BR: N_APP + (int(br) + 1) * N_BR, :2], "white", 3, "-"),
                     (gt, GRN, 2.5, "-")] + [(cur[k], BLUE if alpha < 1 else RED, 1.2, "-") for k in range(pr.shape[0])]
            draw_cam(axc, None, polys, lead_xy)
            names = ["frozen\nbackbone", "sequential\nrefiner", "FutureLift"]; vals = [float(f0[i]), float(f1[i]), float(f2[i])]
            shown = vals if fi >= 11 else [vals[0], 0, 0]
            axt.barh(names, shown, color=[BLUE, ORG, RED]); axt.set_xlim(0, max(vals) * 1.25 + 0.5); axt.invert_yaxis()
            for y, v_ in enumerate(shown):
                if v_: axt.text(v_ + 0.05, y, f"{v_:.2f} m", va="center", fontsize=8)
            axt.set_title("this scene: minFDE@6", fontsize=9)
            axt.text(0.0, -0.32, f"running mean minFDE (scenes so far)\n backbone {np.mean(run_b):.2f} m -> FutureLift {np.mean(run_f):.2f} m" if fi >= 11 else
                     "running mean minFDE\n (updates at RESULT)", transform=axt.transAxes, fontsize=7.5)
            fig.suptitle(f"FutureLift (arXiv:2610.22215) - reconstruction on synthetic data | TEST-SET minADE: backbone {res['backbone_only']['minADE']:.2f} | "
                         f"sequential {res['causal']['minADE']:.2f} | FutureLift {res['bidirectional']['minADE']:.2f}", fontsize=9.5)
            fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())); plt.close(fig)
    dur = [650 if p.startswith("INPUT") else 500 if p.startswith("LIFT") else 380 if p.startswith("REFINE") else 1100 for _ in picks for p in phases]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=True)
    print(f"wrote {out}: {len(frames)} frames, scenes {picks}, per-scene minFDE backbone/seq/FL:",
          [(round(float(f0[i]), 2), round(float(f1[i]), 2), round(float(f2[i]), 2)) for i in picks])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="assets/simulation.gif"); ap.add_argument("--scenes", type=int, default=3)
    a = ap.parse_args(); main(a.config, a.out, a.scenes)
