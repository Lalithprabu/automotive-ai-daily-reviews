"""Render assets/architecture.png (this repo's own diagram, not the paper's figure) and assets/results.png from metrics.json."""
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
INK, MUTED, BLUE, RED, GREEN, AMB = "#1f2937", "#6b7280", "#2563eb", "#dc2626", "#16a34a", "#d97706"
def arch():
    fig, ax = plt.subplots(figsize=(13, 5.2)); ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 5.2)
    def box(x, y, w, h, t, c=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc="white", ec=c, lw=1.8)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9, color=INK)
    def arr(x0, y0, x1, y1, c=MUTED): ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=c, lw=1.5))
    box(0.2, 3.3, 1.9, 1.0, "Motion history\n[a_x, yaw rate]\ntarget + 4 neighbours\n(B,5,15,2)")
    box(2.5, 3.3, 1.8, 1.0, "LSTM encoder\n(xLSTM in paper)\nper vehicle")
    box(4.7, 3.3, 1.8, 1.0, "Graph attention\nover neighbours")
    box(6.9, 3.3, 1.9, 1.0, "LSTM decoder\n+ locked dropout")
    box(9.2, 3.3, 1.8, 1.0, "Head: mean + log-var\nof a_x, yaw rate\n(B,25,2) x2")
    box(11.3, 3.3, 1.5, 1.0, "Sample N=16\nu ~ N(mu, s^2)")
    for a, b in [(2.1, 2.5), (4.3, 4.7), (6.5, 6.9), (8.8, 9.2), (11.0, 11.3)]: arr(a, 3.8, b, 3.8)
    box(9.2, 1.8, 3.6, 0.9, "Kinematic bicycle layer (differentiable)\nv += a dt;  psi += w dt;  x,y += v (cos,sin psi) dt", GREEN)
    arr(12.0, 3.3, 12.0, 2.7, GREEN)
    box(4.2, 0.3, 4.0, 1.1, "K=5 hetero ensemble  or  M=20 MC-dropout\nSigma_total = Sigma_ale + Sigma_epi", AMB)
    arr(9.2, 2.2, 8.2, 1.0, AMB)
    box(9.2, 0.3, 3.6, 1.1, "Split conformal (per horizon step)\nscore r = sqrt(e' S^-1 e);  S_cp = q_t^2 S", RED)
    arr(8.2, 0.85, 9.2, 0.85, RED)
    ax.text(0.2, 1.4, "X-TRACK-UQ (reconstruction)", fontsize=12, color=INK, weight="bold"); ax.text(0.2, 1.0, "Uncertainty lives in motion space,\nis pushed through vehicle dynamics,\nthen conformally calibrated.", fontsize=9, color=MUTED, va="top")
    fig.savefig("assets/architecture.png", dpi=140, bbox_inches="tight", facecolor="white")
def results():
    m = json.load(open("metrics.json")); names = ["baseline", "mc_dropout", "deep_ensemble"]; lab = ["Baseline", "MC dropout", "Deep ensemble"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    a.bar(lab, [m[n]["ADE"] for n in names], color=[MUTED, AMB, BLUE]); a.set_title("ADE over 5 s (m, synthetic) - lower is better", fontsize=10, color=INK)
    for i, n in enumerate(names): a.text(i, m[n]["ADE"] + .03, f'{m[n]["ADE"]:.2f}', ha="center", fontsize=9)
    import numpy as np; x = np.arange(3)
    b.bar(x - .2, [100 * m[n]["coverage_raw"] for n in names], .4, color=RED, label="raw 95% region"); b.bar(x + .2, [100 * m[n]["coverage_cp"] for n in names], .4, color=BLUE, label="conformal 95%")
    b.axhline(95, color=INK, ls="--", lw=1); b.set_ylim(80, 100); b.set_xticks(x, lab); b.legend(fontsize=8, loc="upper left"); b.set_title("Empirical coverage (%) - target 95", fontsize=10, color=INK)
    for A in (a, b): A.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, -0.02, "Synthetic data from this repo - not highD, not the paper's numbers.", ha="center", fontsize=8, color=MUTED)
    fig.savefig("assets/results.png", dpi=140, bbox_inches="tight", facecolor="white")
if __name__ == "__main__": arch(); results()
