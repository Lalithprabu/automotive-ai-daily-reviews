"""Render assets/architecture.png (this repo's own diagram, not the paper's figure) and assets/results.png from results.json."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def box(ax, x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=c, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9)

fig, ax = plt.subplots(figsize=(14, 5.5)); ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 5.5)
ax.text(.1, 5.2, "CoCam4D-style pipeline (this repo's DOP-level reconstruction)", fontsize=13, weight="bold")
box(ax, .2, 3.6, 2.6, 1.0, "Agent k camera tracks\n(T=6 frames, noisy centroid,\ndepth noise ~ range²)", "#e8f1fb")
box(ax, 3.3, 3.6, 2.8, 1.0, "Causal motion encoder\n(window-3 attention,\ntemporal embedding)", "#fdf2d9")
box(ax, 6.6, 4.1, 2.6, .8, "Velocity v⁺ + log-var (NLL)", "#e5f5e0"); box(ax, 6.6, 3.2, 2.6, .8, "p_dyn  |  pos σ (L1 vs |err|)", "#e5f5e0")
box(ax, 9.7, 3.6, 2.0, 1.0, "35-byte DOP\nμ, v, θ, dims, σ", "#fbe3e3")
box(ax, 12.0, 3.6, 1.8, 1.0, "C-V2X\n10 Hz ~1.4 kB", "#eee")
box(ax, 2.2, 1.4, 3.0, 1.0, "Latency compensation\nμ' = μ + v·δ ;  σ'² = σ² + σ_v²δ²", "#fdf2d9")
box(ax, 5.7, 1.4, 3.0, 1.0, "Mahalanobis Hungarian\ngate χ²₃,₀.₉₅ = 7.815", "#e8f1fb")
box(ax, 9.2, 1.4, 3.2, 1.0, "Information-filter update\nΛ=ΣΛᵢ ; μ=Λ⁻¹ΣΛᵢμᵢ", "#e5f5e0")
for a, b in ((2.8, 3.3), (6.1, 6.6), (9.2, 9.7), (11.7, 12.0)): ax.annotate("", (b, 4.1), (a, 4.1), arrowprops=dict(arrowstyle="->"))
ax.plot([12.9, 12.9, 3.7], [3.6, 2.95, 2.95], c="k", lw=1.2); ax.annotate("", (3.7, 2.4), (3.7, 2.95), arrowprops=dict(arrowstyle="->"))
ax.text(8.0, 3.0, "received by ego (age δ)", fontsize=8, va="bottom", ha="center")
for a, b in ((5.2, 5.7), (8.7, 9.2)): ax.annotate("", (b, 1.9), (a, 1.9), arrowprops=dict(arrowstyle="->"))
ax.text(.2, .6, "Old way: average stale boxes with equal weights  →  error grows with message age.   "
        "New way: velocity-propagated, uncertainty-weighted Bayesian fusion.", fontsize=10)
fig.savefig("assets/architecture.png", dpi=130, bbox_inches="tight"); plt.close(fig)

R = json.load(open("results.json"))["fusion"]; lats = list(next(iter(R.values())).keys())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
sty = {"ego_only": ("#888", "Ego only"), "late_fusion_stale": ("#d62728", "Old: stale late fusion"),
       "bayes_learned_no_latcomp": ("#ff7f0e", "Bayes, no latency comp"), "bayes_fixed_cov_latcomp": ("#9467bd", "Latency comp, fixed cov"),
       "cocam4d_full": ("#1f77b4", "CoCam4D-style (full)")}
for m, (c, n) in sty.items():
    x = [int(l) for l in lats]
    a1.plot(x, [R[m][l]["mean_pos_err_m"] for l in lats], "-o", c=c, label=n)
    a2.plot(x, [R[m][l]["recall_2m"] for l in lats], "-o", c=c, label=n)
a1.set_title("Position error at t_now (m, lower better)"); a2.set_title("Recall @ 2 m (higher better)")
for a in (a1, a2): a.set_xlabel("V2X message age (ms)"); a.grid(alpha=.3)
a1.legend(fontsize=8); fig.suptitle("Synthetic benchmark (this repo's reconstruction — NOT the paper's OPV2V+/DAIR-V2X numbers)", fontsize=9)
fig.savefig("assets/results.png", dpi=130, bbox_inches="tight")
