"""Render assets/architecture.png (this repo's own diagram, NOT the paper's figure) and assets/results.png from results.json."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def arch():
    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=110); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4.6)
    def b(x, y, w, h, t, c="#dfe9fb"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=c, ec="#1f6fd6")); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8)
    def a(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color="#444"))
    b(0.1, 1.9, 1.5, 0.9, "Scene encoder\nego history +\nagent tokens"); a(1.6, 2.35, 2.0, 2.35)
    b(2.0, 3.2, 1.6, 0.8, "Maneuver router\ntop-k intents", "#fdf0c8"); b(2.0, 1.9, 1.6, 0.9, "Future world model\nagent futures +\nfuture tokens", "#e4f4e4")
    a(1.6, 2.6, 2.0, 3.5); a(3.6, 3.6, 4.1, 3.2)
    b(4.1, 2.7, 2.0, 1.2, "Gaussian-mixture prior\n(1 comp./maneuver)\n-> MoE MeanFlow\none-step: x = z - u(z,0,1)", "#fdf0c8")
    a(6.1, 3.3, 6.6, 2.7)
    b(6.6, 1.5, 2.3, 1.2, "Evidence scorer\nwaypoints attend to\ncurrent + FUTURE tokens\n-> PDM factors", "#e4f4e4"); a(3.6, 2.2, 6.6, 2.0)
    a(8.9, 2.1, 9.3, 2.1)
    b(9.3, 1.5, 1.2, 1.2, "Frenet\nrefiner\n(ds, dd)"); a(10.5, 2.1, 10.9, 2.1)
    b(10.9, 1.5, 1.0, 1.2, "Joint re-score\nretained +\nrefined -> pick", "#fbd9d6")
    ax.text(0.1, 0.8, "Old way (multi-query regression): same encoder -> K learned queries -> K regressed paths + classifier score. No intent prior, no future evidence, no refinement.", fontsize=8, color="#a33")
    ax.text(0.1, 0.3, "Reconstruction diagram (this repo's own, from the abstract-level description of arXiv:2610.13409) - not the paper's figure.", fontsize=7, style="italic")
    fig.savefig("assets/architecture.png", bbox_inches="tight"); plt.close(fig)

def results():
    R = json.load(open("results.json"))
    names = [("old_way_multiquery", "Old: multi-query", "#d6403a"), ("new_generate_only", "Gen + current score\n(no refine)", "#9bbbe8"),
             ("new_evidence_current_only", "+ refine\n(current only)", "#5c97e0"), ("new_full_maneuverflow", "+ future evidence\n(full)", "#1f6fd6")]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), dpi=110)
    for ax, (key, title) in zip(axs, [("pdms", "Synthetic PDMS (higher better)"), ("nc", "No-collision rate"), ("minade", "minADE@6 vs. human path, m (lower better)")]):
        ax.bar(range(4), [R[n][key] for n, _, _ in names], color=[c for _, _, c in names]); ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l, _ in names], rotation=18, ha="right", fontsize=7)
        for i, (n, _, _) in enumerate(names): ax.text(i, R[n][key], f"{R[n][key]:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(title, fontsize=9)
    fig.suptitle("Held-out synthetic scenes (600) - this repo's reconstruction, NOT the paper's NAVSIM numbers", fontsize=9)
    fig.tight_layout(); fig.savefig("assets/results.png"); plt.close(fig)

if __name__ == "__main__": arch(); results()
