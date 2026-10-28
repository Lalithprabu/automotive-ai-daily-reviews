"""Synthetic 'AI in action' demo: a fleet of held-out cells is observed for n = 1..50 cycles.
Each frame: INPUT (voltage curves + compressed scalogram of the featured cell), GROUND TRUTH (green true cycle life),
MODEL PREDICTION (cyan predicted life + error), TELEMETRY (fleet MAPE vs cycles seen).
python simulate.py            -> assets/cwt_rul_simulation.gif   |   python simulate.py --live  -> real-time window
Requires checkpoint.pt (run train.py first)."""
import argparse, numpy as np, torch, matplotlib
from src.synthetic import make_cells, split_idx, pick_cycles
from src.model import CWTRULNet

ap = argparse.ArgumentParser(); ap.add_argument("--live", action="store_true"); ap.add_argument("--fleet", type=int, default=12)
ap.add_argument("--out", default="assets/cwt_rul_simulation.gif"); a = ap.parse_args()
if not a.live: matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ck = torch.load("checkpoint.pt"); cfg = ck["cfg"]; d, c, m = cfg["data"], cfg["cwt"], cfg["model"]
net = CWTRULNet(d["T"], 3, c["n_scales"], c["compress"], m["width"], m["emb"], m["dropout"], c["w0"], c["s_min"], c["s_max"])
net.load_state_dict(ck["model"]); net.eval()
X, life, _ = make_cells(d["n_cells"], d["n_cycles"], d["T"], d["seed"]); _, _, te = split_idx(len(X), d["val_frac"], d["test_frac"], d["seed"])
fleet = te[np.argsort(life[te])][np.linspace(0, len(te) - 1, a.fleet).astype(int)]        # spread over short..long life
N = np.arange(1, d["n_cycles"] + 1)
with torch.no_grad():
    PRED = np.stack([np.exp(net(torch.from_numpy(X[fleet][:, pick_cycles(n, cfg["train"]["max_cycles_in"])])).numpy() * ck["sd"] + ck["mu"]) for n in N])  # [50,F]
TRUE = life[fleet]; MAPE = (np.abs(PRED - TRUE) / TRUE).mean(1) * 100
feat = a.fleet // 2                                                                        # featured cell
BG, FG, GREEN, CYAN, RED = "#0b0f14", "#d8dee9", "#39ff14", "#00e5ff", "#ff4d4d"
plt.rcParams.update({"text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": "#445"})
fig = plt.figure(figsize=(12, 6.5), facecolor=BG); gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25])
axV, axS, axF, axT = [fig.add_subplot(g) for g in (gs[0, 0], gs[0, 1], gs[:, 2], gs[1, :2])]
for ax in (axV, axS, axF, axT): ax.set_facecolor(BG)
title = fig.suptitle("", fontsize=13, fontfamily="monospace")
scal = lambda x: net.comp(net.cwt(torch.from_numpy(x[None]))).numpy()[0, 0]                # voltage channel [S',T]
t_ax = np.arange(d["T"])


def draw(i):
    n = int(N[i]); title.set_text(f"CWT-RUL-Mini | cycles observed n={n:02d} | fleet MAPE {MAPE[i]:.0f}%  (synthetic demo)")
    axV.clear(); axV.set_title("INPUT: charge voltage, cycle 1 vs n", fontsize=9)
    axV.plot(t_ax, X[fleet[feat], 0, 0], color="#667", lw=1); axV.plot(t_ax, X[fleet[feat], n - 1, 0], color=FG, lw=1.2)
    axS.clear(); axS.imshow(scal(X[fleet[feat], n - 1]), aspect="auto", origin="lower", cmap="magma"); axS.set_title("INPUT: compressed scalogram (8 scales)", fontsize=9)
    axF.clear(); y = np.arange(a.fleet); order = np.argsort(TRUE)
    axF.barh(y, TRUE[order], color="none", edgecolor=GREEN, lw=1.2, label="GROUND TRUTH life")
    axF.scatter(PRED[i][order], y, color=CYAN, zorder=3, s=28, label="MODEL PREDICTION")
    for yy, o in zip(y, order):
        axF.plot([TRUE[o], PRED[i][o]], [yy, yy], color=RED if abs(PRED[i][o] - TRUE[o]) / TRUE[o] > 0.3 else "#556", lw=1)
    axF.set_xlim(0, 450); axF.set_xlabel("cycles to 80% capacity"); axF.legend(loc="lower right", fontsize=8, facecolor=BG, edgecolor="#445")
    axF.set_title("Fleet: true vs predicted life (red = >30% off)", fontsize=9)
    axT.clear(); axT.plot(N, MAPE, color="#667"); axT.plot(N[: i + 1], MAPE[: i + 1], color=CYAN); axT.axhline(58, color=RED, ls="--", lw=0.8)
    axT.text(N[-1], 60, "mean-life baseline", color=RED, ha="right", fontsize=8); axT.set_ylim(0, 90); axT.set_xlabel("cycles observed"); axT.set_ylabel("MAPE %")
    axT.set_title("TELEMETRY: error shrinks as more cycles are seen", fontsize=9)
    if i == len(N) - 1: fig.text(0.5, 0.01, "Synthetic demo - reconstruction of arXiv:2610.34159 idea - not the authors' model or numbers", ha="center", fontsize=8, color="#99a")


fig.tight_layout(rect=(0, 0.03, 1, 0.95))
if a.live:
    for i in range(len(N)): draw(i); plt.pause(0.15)
    plt.show()
else:
    FuncAnimation(fig, draw, frames=len(N), interval=150).save(a.out, writer=PillowWriter(fps=7), dpi=70); print("saved", a.out)
