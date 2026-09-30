"""Renders assets/walt_simulation.gif from the REAL trained checkpoint: input raster | ground truth vs OLD (raw waypoints)
vs NEW (WALT latent planner) overlay | live per-step error telemetry. Synthetic scenes only."""
import torch, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, io
from PIL import Image
from walt.data import Scenes, denorm_wp, X_MAX, Y_MAX, T
from walt.model import FrozenWorldModel, TrajAutoencoder, WALTPlanner, DirectPlanner

def load():
    ck = torch.load("checkpoint.pt"); wm = FrozenWorldModel(); wm.load_state_dict(ck["wm"]); wm.eval()
    ae = TrajAutoencoder(); ae.load_state_dict(ck["ae"]); ae.eval()
    new = WALTPlanner(wm, ae); new.planner.load_state_dict(ck["new"]); old = DirectPlanner(wm); old.planner.load_state_dict(ck["old"])
    return wm, old.eval(), new.eval()

def main(n_scenes=3, out="assets/walt_simulation.gif"):
    wm, old, new = load(); va = Scenes(500, 2)
    idx = [i for i, m in enumerate(va.meta) if m[0]][:n_scenes]     # scenes with a lead vehicle
    frames = []
    for si in idx:
        x, wp = va.x[si:si + 1], va.wp[si]
        with torch.no_grad(): po = denorm_wp(old(x))[0]; pn = denorm_wp(new(x)[0])[0]; pf = wm(x)[0]
        eo = (po - wp).norm(dim=-1).numpy(); en = (pn - wp).norm(dim=-1).numpy()
        for t in range(1, T + 1):
            fig, ax = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0b0f14")
            for a in ax: a.set_facecolor("#0b0f14"); a.tick_params(colors="#8b98a5")
            ax[0].imshow((va.x[si, 0] * 0.6 + va.x[si, 1]).numpy(), cmap="magma", origin="lower", extent=[-Y_MAX, Y_MAX, 0, X_MAX], aspect="auto")
            ax[0].set_title("INPUT: BEV raster (road + lead vehicle)", color="w")
            ax[1].imshow(pf[1].numpy() > 0.3, cmap="gray", origin="lower", extent=[-Y_MAX, Y_MAX, 0, X_MAX], aspect="auto", alpha=0.35)
            for P, c, lab in [(wp, "#3fb950", "Ground truth"), (po, "#f85149", "OLD raw-waypoint"), (pn, "#58a6ff", "NEW WALT latent")]:
                ax[1].plot(P[:t, 1], P[:t, 0], "o-", color=c, label=lab, ms=4)
            ax[1].set_xlim(-Y_MAX, Y_MAX); ax[1].set_ylim(0, X_MAX); ax[1].legend(facecolor="#161b22", labelcolor="w", fontsize=8, loc="lower left")
            ax[1].set_title(f"GT vs prediction overlay  (t = {t*0.5:.1f}s)", color="w")
            ax[2].plot(np.arange(1, t + 1) * 0.5, eo[:t], color="#f85149", label=f"OLD  err {eo[t-1]:.2f} m")
            ax[2].plot(np.arange(1, t + 1) * 0.5, en[:t], color="#58a6ff", label=f"NEW  err {en[t-1]:.2f} m")
            ax[2].set_xlim(0, 4.2); ax[2].set_ylim(0, max(eo.max(), en.max(), 1) * 1.1); ax[2].legend(facecolor="#161b22", labelcolor="w")
            ax[2].set_title(f"Telemetry: L2 error (ADE old {eo.mean():.2f} | new {en.mean():.2f})", color="w")
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=60, facecolor=fig.get_facecolor()); plt.close(fig)
            frames.append(Image.open(io.BytesIO(buf.getvalue())).convert("P", palette=Image.ADAPTIVE))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=350, loop=0); print("wrote", out, len(frames), "frames")

if __name__ == "__main__": main()
