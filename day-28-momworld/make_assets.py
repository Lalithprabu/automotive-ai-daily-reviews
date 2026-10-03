"""Render assets/architecture.png (this repo's own diagram, not the paper's figure) and assets/results.png from outputs/results.json."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BG, FG = "#0f1419", "#e6edf3"

def arch():
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=110, facecolor=BG); ax.set_facecolor(BG); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 5.2)
    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="white", lw=1)); ax.text(x + w / 2, y + h / 2, t, color="white", ha="center", va="center", fontsize=8.5)
    def arr(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=FG, lw=1.4))
    ax.text(0.2, 4.95, "MomWorld-style pipeline (this repo's reconstruction)", color=FG, fontsize=12, weight="bold")
    box(0.2, 2.2, 1.7, 1.0, "History\n(ego + lead)\n8 steps", "#30363d"); arr(1.9, 2.7, 2.3, 2.7)
    box(2.3, 2.2, 1.7, 1.0, "GRU encoder\nh_1..h_T", "#1f6feb"); arr(4.0, 2.7, 4.5, 3.4); arr(4.0, 2.7, 4.5, 2.0)
    box(4.5, 3.0, 2.0, 0.9, "Momentum extractor\nm0 = Σ softmax·Δh", "#9e6a03"); box(4.5, 1.5, 2.0, 0.9, "Scene context c\n(from h_T)", "#238636")
    arr(6.5, 3.4, 7.0, 2.9); arr(6.5, 1.95, 7.0, 2.5)
    box(7.0, 1.7, 2.6, 1.9, "Latent rollout k=1..H\nkeep_k = σ(W[m,z,c,e])  (reset gate)\nm_k = keep·α·m_{k-1} + g_k(z,c,e)\nz_k = z_{k-1} + m_k", "#8957e5")
    arr(9.6, 2.65, 10.1, 3.3); arr(9.6, 2.65, 10.1, 1.9)
    box(10.1, 3.0, 2.7, 0.8, "Base waypoints\n(+ future-scene aux head)", "#1f6feb"); box(10.1, 1.5, 2.7, 0.8, "MoFlow: flow-matching\nbase→plan, 4 Euler steps", "#da3633")
    arr(11.45, 3.0, 11.45, 2.3)
    box(10.1, 0.3, 2.7, 0.8, "plan = base + σ(φ_k)·(x₁ − base)\nhorizon-aware fusion", "#30363d"); arr(11.45, 1.5, 11.45, 1.1)
    box(2.3, 0.3, 7.3, 0.7, "Old way ablation: z_k = z_{k-1} + MLP([z,c,e])  — single latent state, no momentum, no gate", "#484f58")
    fig.savefig("assets/architecture.png", facecolor=BG); plt.close(fig)

def results():
    r = json.load(open("outputs/results.json")); s = r["summary"]; st = r["stale_momentum"]
    names = [("stale", "Stale momentum\n(non-learned)"), ("baseline", "Single-latent\nbaseline"), ("no_moflow", "MomWorld\nno MoFlow"), ("no_reset_gate", "MomWorld\nno reset gate"), ("momworld", "MomWorld-style\n(full)")]
    ade = [st["ade"]] + [s[k]["ade"][0] for k, _ in names[1:]]; err = [0] + [s[k]["ade"][1] for k, _ in names[1:]]
    cols = ["#f85149", "#e3a008", "#8b949e", "#8b949e", "#58a6ff"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), dpi=110, facecolor=BG)
    for a in axs: a.set_facecolor(BG); a.tick_params(colors=FG); [sp.set_color("#2b333d") for sp in a.spines.values()]
    axs[0].bar(range(5), ade, yerr=err, color=cols, ecolor="white", capsize=3); axs[0].set_xticks(range(5)); axs[0].set_xticklabels([n for _, n in names], fontsize=7, color=FG)
    axs[0].set_title("Held-out ADE (m), synthetic data, mean±std over 3 seeds", color=FG, fontsize=10)
    ce = [st["collision_event"]] + [s[k]["collision_event"][0] for k, _ in names[1:]]
    axs[1].bar(range(5), [c * 100 for c in ce], color=cols); axs[1].set_xticks(range(5)); axs[1].set_xticklabels([n for _, n in names], fontsize=7, color=FG)
    axs[1].set_title("Collision rate in EVENT scenes (%)", color=FG, fontsize=10)
    fig.text(0.5, 0.01, "Toy synthetic world — NOT the paper's NAVSIM / nuScenes / Bench2Drive results.", color="#8b949e", ha="center", fontsize=8)
    fig.savefig("assets/results.png", facecolor=BG); plt.close(fig)

if __name__ == "__main__":
    arch(); results(); print("assets written")
