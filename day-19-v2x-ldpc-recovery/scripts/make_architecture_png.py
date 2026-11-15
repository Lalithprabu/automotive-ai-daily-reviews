"""
Generates assets/architecture_diagram.png: deterministic matplotlib boxes +
arrows for the prediction-aided V2X LDPC recovery pipeline.
Palette fixed per this project's visual style.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
YELLOW = "#eda100"
BG = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"

fig, ax = plt.subplots(figsize=(13, 7.5), dpi=150, facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.5)
ax.axis("off")

fig.suptitle("Prediction-Aided V2X BSM Recovery via Uncertainty-Aware LDPC Decoding",
             color=INK, fontsize=15, fontweight="bold", x=0.5, y=0.975)
ax.text(6.5, 6.95, "This repo's own reconstruction of the mechanism in arXiv:2609.25609 (abstract-only source)",
        color=MUTED, fontsize=9, ha="center", style="italic")


def box(x, y, w, h, text, color, text_color=INK, fontsize=9.3, lw=1.8):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=lw, edgecolor=color, facecolor="white", alpha=0.97, zorder=3)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=text_color,
            fontsize=fontsize, fontweight="medium", zorder=4, linespacing=1.35)
    return (x, y, w, h)


def arrow(b1, b2, color, connectionstyle="arc3,rad=0.0", side1="right", side2="left"):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    pts = {"right": (x1 + w1, y1 + h1 / 2), "left": (x1, y1 + h1 / 2),
           "top": (x1 + w1 / 2, y1 + h1), "bottom": (x1 + w1 / 2, y1)}
    start = pts[side1]
    pts2 = {"right": (x2 + w2, y2 + h2 / 2), "left": (x2, y2 + h2 / 2),
            "top": (x2 + w2 / 2, y2 + h2), "bottom": (x2 + w2 / 2, y2)}
    end = pts2[side2]
    a = FancyArrowPatch(start, end, connectionstyle=connectionstyle, arrowstyle="-|>",
                         mutation_scale=14, linewidth=2.0, color=color, zorder=2)
    ax.add_patch(a)


# ---- Row 1: prediction path (blue) ----
b_hist = box(0.3, 5.6, 2.0, 1.1, "Trajectory\nHistory\n(BSM stream)", BLUE)
b_pred = box(2.7, 5.6, 2.5, 1.1, "ProbabilisticMotion\nPredictor\n(GRU encoder-decoder)", BLUE)
b_gauss = box(5.65, 5.6, 2.3, 1.1, "Predicted Gaussian\nper field (mu, sigma)\nx,y,speed,heading,accel", BLUE)
b_bitprob = box(8.4, 5.6, 2.3, 1.1, "BSMBitProbabilityLayer\nquantize + serialize\n-> P(bit=1)", BLUE)
b_llr = box(11.05, 5.6, 1.65, 1.1, "LLRPriorHead\n->\nprior LLR", BLUE)

arrow(b_hist, b_pred, BLUE)
arrow(b_pred, b_gauss, BLUE)
arrow(b_gauss, b_bitprob, BLUE)
arrow(b_bitprob, b_llr, BLUE)

# ---- Row 2: channel path (orange) ----
b_bsm = box(0.3, 3.7, 2.0, 1.1, "Ground-truth\nBSM (real)", ORANGE)
b_ldpc_enc = box(2.7, 3.7, 2.0, 1.1, "LDPC Encoder\n(n=256, k~131)", ORANGE)
b_channel = box(5.15, 3.7, 2.0, 1.1, "AWGN Channel\n(BPSK, Eb/N0)", ORANGE)
b_chllr = box(7.55, 3.7, 2.0, 1.1, "Channel LLR\nfrom received\nsamples", ORANGE)

arrow(b_bsm, b_ldpc_enc, ORANGE)
arrow(b_ldpc_enc, b_channel, ORANGE)
arrow(b_channel, b_chllr, ORANGE)

# ---- Row 3: fused decoding path (aqua/green) ----
b_decoder = box(4.6, 1.6, 3.6, 1.3,
                "LDPC Belief-Propagation Decoder\n(min-sum, 2-pass, CRC-guided)\nattempt 1: channel LLR only\nattempt 2 (on CRC fail): + prior LLR",
                GREEN, fontsize=9.0)
b_out = box(9.1, 1.7, 2.6, 1.1, "Recovered BSM\n(decoded message)", GREEN)

arrow(b_decoder, b_out, GREEN)

# feed channel LLR down into decoder
arrow(b_chllr, b_decoder, ORANGE, connectionstyle="arc3,rad=0.15", side1="bottom", side2="top")
# feed prior LLR down into decoder
arrow(b_llr, b_decoder, BLUE, connectionstyle="arc3,rad=-0.25", side1="bottom", side2="top")

ax.text(6.4, 0.95, "CRC-guided: 2nd (prediction-aided) attempt only runs if the 1st decode's syndrome check fails",
        color=MUTED, fontsize=8.5, ha="center", style="italic")

# Legend
legend_elems = [
    Line2D([0], [0], color=BLUE, lw=3, label="Prediction / neural path"),
    Line2D([0], [0], color=ORANGE, lw=3, label="Channel / RF path"),
    Line2D([0], [0], color=GREEN, lw=3, label="Fused decoding path"),
]
ax.legend(handles=legend_elems, loc="lower left", bbox_to_anchor=(0.0, -0.02), fontsize=9,
          frameon=False, labelcolor=INK2, ncol=3)

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.savefig("assets/architecture_diagram.png", facecolor=BG, dpi=150)
print("saved assets/architecture_diagram.png")
