"""Render assets/architecture.png (this repo's own diagram, NOT the paper's figure) and assets/results.png from checkpoints/eval.json."""
import json, pathlib, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
BG, FG, A, B, C = "#0e1117", "#e6edf3", "#58a6ff", "#3ddc97", "#f2cc60"
pathlib.Path("assets").mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 4.6), facecolor=BG); ax.set_facecolor(BG); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 4.6)
def box(x, y, w, h, t, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc="#161b22", ec=c, lw=1.8)); ax.text(x + w/2, y + h/2, t, color=FG, ha="center", va="center", fontsize=9)
def arrow(x0, y0, x1, y1): ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=FG, lw=1.4))
box(.2, 1.6, 2.1, 1.4, "7 sensor cues\nlat. velocity, lat. accel,\nTTC front / L-front /\nR-front / L-rear / R-rear", A)
box(2.8, 1.6, 2.0, 1.4, "Symbolic discretiser\nvy,ay → 5 bins each\nTTC → HIGH/MED/LOW\n(precursor: ≤4 s / <10 s)", A)
box(5.3, 2.6, 2.4, 1.3, "Knowledge graph (KG)\n(Maneuver, relation, value)\ntriples per sample", C)
box(5.3, 0.6, 2.4, 1.3, "TransE embeddings\nh + r ≈ t, dim 100\nself-adversarial loss", C)
box(8.2, 1.6, 2.1, 1.4, "Bayesian head\nP(h|e) ∝ P(h) Π P(e_k|h)\nsoftmax(−‖h+r−t‖/τ)", B)
box(10.5, 1.6, 1.3, 1.4, "LLC / LK / RLC\n+ per-cue\nexplanation", B)
for a in [(2.3, 2.3, 2.8, 2.3), (4.8, 2.3, 5.3, 3.2), (6.5, 2.6, 6.5, 1.9), (7.7, 1.25, 8.2, 2.1), (10.3, 2.3, 10.5, 2.3)]: arrow(*a)
ax.text(6, 4.3, "KG-embedding + Bayesian lane-change predictor — this repo's own reconstruction diagram", color=FG, ha="center", fontsize=11, weight="bold")
ax.text(6, .15, "Missing sensor → that factor is simply omitted from the product (no imputation needed).", color=FG, ha="center", fontsize=8.5)
fig.savefig("assets/architecture.png", dpi=130, facecolor=BG); plt.close(fig)

ev = json.load(open("checkpoints/eval.json")); H = sorted(ev["horizon_f1"], key=float)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=BG)
for a in (a1, a2): a.set_facecolor(BG); a.tick_params(colors=FG); [s.set_color("#30363d") for s in a.spines.values()]
x = range(len(H)); w = .27
a1.bar([i - w for i in x], [ev["horizon_f1"][h]["kg"] for h in H], w, color=B, label="KG-Bayes")
a1.bar(list(x), [ev["horizon_f1"][h]["kg_rear_sensors_dropped"] for h in H], w, color=C, label="KG-Bayes, rear TTC dropped")
a1.bar([i + w for i in x], [ev["horizon_f1"][h]["mlp"] for h in H], w, color=A, label="MLP baseline")
a1.set_xticks(list(x)); a1.set_xticklabels([f"{float(h):.0f} s" for h in H]); a1.set_ylim(0, 1.05); a1.axhline(1/3, color=FG, ls=":", lw=.8)
a1.set_title("Macro-F1 vs. seconds before lane-line crossing (synthetic)", color=FG, fontsize=10); a1.legend(facecolor=BG, labelcolor=FG, edgecolor="#30363d", fontsize=8)
gaps = sorted(ev["closed_loop"], key=float); xs = range(len(gaps))
a2.bar([i - .2 for i in xs], [ev["closed_loop"][g]["reactive"]["mean_peak_decel"] for g in gaps], .4, color="#ff5c5c", label="reactive (capped 9.0)")
a2.bar([i + .2 for i in xs], [ev["closed_loop"][g]["predictive"]["mean_peak_decel"] for g in gaps], .4, color=B, label="KG-Bayes predictive")
for i, g in enumerate(gaps): a2.text(i - .2, 9.2, f"coll {ev['closed_loop'][g]['reactive']['collision_rate']*100:.0f}%", color="#ff5c5c", ha="center", fontsize=8); a2.text(i + .2, ev["closed_loop"][g]["predictive"]["mean_peak_decel"] + .25, f"coll {ev['closed_loop'][g]['predictive']['collision_rate']*100:.0f}%", color=B, ha="center", fontsize=8)
a2.set_xticks(list(xs)); a2.set_xticklabels([f"{float(g):.0f} m gap" for g in gaps]); a2.set_ylim(0, 10.5); a2.set_ylabel("mean peak decel (m/s²)", color=FG)
a2.set_title("Closed-loop cut-in, 200 randomized runs per gap (own sim)", color=FG, fontsize=10); a2.legend(facecolor=BG, labelcolor=FG, edgecolor="#30363d", fontsize=8, loc="center right")
fig.suptitle("This repo's synthetic results — NOT the paper's numbers", color=FG, fontsize=11); fig.tight_layout(); fig.savefig("assets/results.png", dpi=130, facecolor=BG)
