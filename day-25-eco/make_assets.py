"""Renders assets/architecture.png (this repo's own diagram, NOT the paper's figure) and assets/results.png from results.json."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BG, TXT, GRID = "#0e1117", "#e6edf3", "#30363d"

fig, ax = plt.subplots(figsize=(12, 4.6), facecolor=BG); ax.set_facecolor(BG); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4.6)
def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=c, ec="white", lw=1)); ax.text(x + w/2, y + h/2, t, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
box(0.2, 2.9, 2.3, 1.1, "Camera / route\n(perception)", "#30363d"); box(3.0, 2.9, 2.6, 1.1, "End-to-end policy\n(waypoint regressor)", "#8250df")
box(6.3, 2.9, 2.2, 1.1, "raw waypoints\nw1 … wT", "#da3633"); box(9.4, 2.9, 2.4, 1.1, "Tracker /\ncontroller", "#30363d")
box(0.2, 0.5, 2.3, 1.1, "Executed history\n(last H poses)", "#238636"); box(5.4, 0.5, 4.4, 1.4, "ECO layer (training-free)\nmin |D2 s|² + |D3 s|² + λ|x−w|²\nendpoint w_T fixed", "#1f6feb")
for a, b in [((2.5, 3.45), (3.0, 3.45)), ((5.6, 3.45), (6.3, 3.45)), ((8.5, 3.45), (9.4, 3.45)), ((7.4, 2.9), (7.6, 1.95)), ((2.5, 1.05), (5.4, 1.15)), ((9.2, 1.2), (10.6, 2.9))]:
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color=TXT, lw=1.5))
ax.text(6, 4.4, "ECO = post-processing layer between policy and controller (this repo's reconstruction)", color=TXT, ha="center", fontsize=11)
fig.savefig("assets/architecture.png", dpi=110, facecolor=BG); plt.close(fig)

r = json.load(open("results.json"))
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=BG)
for ax, (k, t) in zip(axs, (("jerk_rms", "RMS jerk (m/s³) ↓"), ("steer_rate_rms", "RMS steer rate (rad/s) ↓"), ("lat_rms", "RMS lateral error (m) ↓"))):
    v = [r["raw"][k], r["eco"][k]]; ax.set_facecolor(BG)
    ax.bar(["raw", "+ECO"], v, color=["#f85149", "#58a6ff"])
    for i, x in enumerate(v): ax.text(i, x, f"{x:.2f}", ha="center", va="bottom", color=TXT)
    ax.set_title(t, color=TXT, fontsize=10); ax.tick_params(colors=TXT)
    for s in ax.spines.values(): s.set_color(GRID)
fig.suptitle("Synthetic closed-loop, 40 episodes, same weights (not paper numbers)", color=TXT, fontsize=10)
fig.savefig("assets/results.png", dpi=110, facecolor=BG); plt.close(fig)
print("ok")
