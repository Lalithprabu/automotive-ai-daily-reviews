"""Renders assets/architecture.png (this repo's own diagram, not the paper's figure) and assets/results.png (real run data)."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def box(ax, x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=c, ec="#222", lw=1.2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5)


def arrow(ax, a, b, t=None):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.4))
    if t: ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.2, t, fontsize=7, ha="center")


fig, ax = plt.subplots(figsize=(13, 5.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.6); ax.axis("off")
box(ax, 0.2, 3.9, 2.2, 1.1, "Frozen backbone\n(any: QCNet/DONUT/...)\nK proposals + logits", "#dbe8ff")
box(ax, 0.2, 0.5, 2.2, 1.6, "Scene context\nhistories + map\n(own encoder, trained\nwith the refiner)", "#e8f5e9")
box(ax, 3.2, 3.9, 2.6, 1.1, "LIFT\nproposal -> R temporal nodes\n(GRU over segment, local frame)", "#fff3cd")
box(ax, 6.5, 1.2, 4.0, 3.8, "FutureLift layer x L (graph built ONCE)\n\nhistory edges   map edges\nTEMPORAL edges (bidirectional)\nmodal edges  (social: joint only)\n\nrelation-specific MHA + edge features\nresidual + pre-LN FFN", "#fde2e4")
box(ax, 11.0, 3.2, 1.9, 1.6, "Heads\nper-interval\nresidual (x,y)\nLaplace scale\nconfidence", "#e6e0ff")
box(ax, 11.0, 0.8, 1.9, 1.3, "Refined K modes\n+ new scores", "#d4f5d4")
arrow(ax, (2.4, 4.45), (3.2, 4.45)); arrow(ax, (5.8, 4.45), (6.5, 4.3)); arrow(ax, (2.4, 1.3), (6.5, 2.0), "ctx")
arrow(ax, (10.5, 3.6), (11.0, 3.8)); arrow(ax, (11.95, 3.2), (11.95, 2.1))
ax.text(6.5, 0.4, "all R segments of a proposal updated simultaneously: later evidence can correct earlier segments in one pass", fontsize=8.5, style="italic")
ax.set_title("FutureLift refiner - this repo's reconstruction diagram (not the paper's figure)", fontsize=11)
fig.savefig("assets/architecture.png", dpi=110, bbox_inches="tight"); plt.close(fig)

r = json.load(open("checkpoints/results.json"))
names = ["backbone\nonly", "no temporal\nedges", "sequential\n(causal)", "FutureLift\n(bidirectional)"]
keys = ["backbone_only", "none", "causal", "bidirectional"]
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
for a, m, t in zip(axs, ["minADE", "minFDE", "MR"], ["minADE@6 (m)", "minFDE@6 (m)", "Miss rate @2 m"]):
    v = [r[k][m] for k in keys]; b = a.bar(names, v, color=["#3b7ddd", "#aaa", "#f08c00", "#e0245e"])
    a.set_title(t); a.tick_params(axis="x", labelsize=7.5)
    for x, y in zip(b, v): a.text(x.get_x() + x.get_width() / 2, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
fig.suptitle("Held-out SYNTHETIC test set (800 scenes) - not the paper's Argoverse 2 / WOMD numbers", fontsize=10)
fig.savefig("assets/results.png", dpi=110, bbox_inches="tight")
