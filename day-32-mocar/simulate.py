"""Render the 'AI in action' simulation GIF from the REAL trained checkpoints.

Per scene: (1) code-by-code autoregressive generation (6 motion codes, each drawn with the reference frame it
induces), (2) playback of the ground-truth agent with running error telemetry for one-shot vs MoCAR top-1.
Panels: BEV (map lanes, observed history, ground truth, MoCAR modes, one-shot baseline) | motion-code ribbon |
telemetry (error-vs-time + mode probabilities).  Scenes: first held-out test scene of each maneuver type
(no cherry-picking).
"""
import argparse, io, json, yaml, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from mocar.data import SyntheticDrivingDataset, KINDS
from mocar.model import MoCAR, OneShotBaseline, compose

BG, FG, GRID = "#0f1419", "#e6edf3", "#2b3440"
C_GT, C_HIST, C_MOC, C_OLD, C_LANE = "#3fb950", "#58a6ff", "#ff7b72", "#8b949e", "#4a5568"
MODE_COL = ["#ff7b72", "#ffa657", "#d2a8ff"]


def load(cfg_path):
    cfg = yaml.safe_load(open(cfg_path)); mc = cfg["model"]
    mo = MoCAR(mc["code_dim"], mc["d_model"], mc["n_heads"], mc["n_layers"], mc["n_modes"]); mo.load_state_dict(torch.load("checkpoint_mocar.pt")); mo.eval()
    ol = OneShotBaseline(mc["d_model"], mc["n_heads"], mc["n_layers"], mc["n_modes"]); ol.load_state_dict(torch.load("checkpoint_oneshot.pt")); ol.eval()
    return cfg, mo, ol


def style(ax, title):
    ax.set_facecolor(BG); ax.set_title(title, color=FG, fontsize=10, loc="left")
    for s in ax.spines.values(): s.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=7); ax.grid(color=GRID, lw=0.4)


def frame(sc, mo_out, ol_out, codes, seg_shown, t_play, kind, fig_size=(12.6, 5.6)):
    gt = sc["fut_pts"][1:].numpy(); hist = sc["hist_pts"].numpy(); lanes = sc["lanes"].numpy(); lm = sc["lane_mask"].numpy()
    probs = torch.softmax(mo_out["scores"][0], 0).numpy(); top = int(probs.argmax())
    fig = plt.figure(figsize=fig_size, dpi=90, facecolor=BG); gs = fig.add_gridspec(2, 3, width_ratios=[2.6, 1, 1.2], height_ratios=[1, 1], wspace=0.28, hspace=0.45)
    ax = fig.add_subplot(gs[:, 0]); style(ax, f"BEV  |  scene: {kind}  |  agent frame (m)")
    for i in range(len(lanes)):
        if lm[i]: ax.plot(lanes[i, :, 0], lanes[i, :, 1], color=C_LANE, lw=6, alpha=0.5, solid_capstyle="round")
    ax.plot(hist[:, 0], hist[:, 1], color=C_HIST, lw=2.5, label="observed history")
    ax.plot(gt[:, 0], gt[:, 1], color=C_GT, lw=2.5, ls="--", label="ground truth (revealed)")
    if t_play is None: ax.plot([], [], color=C_GT)
    n_pts = seg_shown * 5
    for m in range(3):
        p = mo_out["pts"][0, m].detach().numpy()[:n_pts]
        if n_pts: ax.plot(p[:, 0], p[:, 1], color=MODE_COL[m], lw=3.5 if m == top else 1.6, alpha=1 if m == top else 0.6,
                          label=f"MoCAR mode {m+1}  p={probs[m]:.2f}")
        if n_pts and seg_shown <= 6:  # draw the reference frame induced by each generated code (coordinate-aware chain)
            for j in range(seg_shown):
                e = mo_out["ends"][0, m, j].detach().numpy(); L = 1.6
                if m == top:
                    ax.arrow(e[0], e[1], L * np.cos(e[2]), L * np.sin(e[2]), color=FG, width=0.15, head_width=0.7, length_includes_head=True, zorder=5)
    op = ol_out["pts"][0, int(ol_out["scores"][0].argmax())].detach().numpy()
    ax.plot(op[:, 0], op[:, 1], color=C_OLD, lw=2, ls=":", label="one-shot baseline (top-1)")
    if t_play is not None:
        ax.scatter(*gt[min(t_play, len(gt) - 1)], s=140, c=C_GT, marker="s", zorder=6, edgecolor="white")
    ax.scatter(*hist[-1], s=80, c=C_HIST, marker="s", zorder=6, edgecolor="white")
    ax.set_aspect("equal"); ax.set_xlim(-12, 46); ax.set_ylim(-16, 30)
    ax.legend(loc="upper left", fontsize=6.5, facecolor=BG, edgecolor=GRID, labelcolor=FG)
    # code ribbon
    axr = fig.add_subplot(gs[0, 1]); style(axr, "Motion codes (top mode)"); axr.grid(False)
    rib = codes[top].detach().numpy().T.copy(); mask = np.zeros_like(rib); mask[:, seg_shown:] = 1
    axr.imshow(np.ma.array(rib, mask=mask), cmap="coolwarm", aspect="auto", vmin=-3, vmax=3)
    axr.set_xticks(range(6)); axr.set_xticklabels([f"c{j+1}" for j in range(6)]); axr.set_ylabel("latent dim", color=FG, fontsize=8)
    axr.set_xlabel(f"{seg_shown}/6 generated" + ("  → each code fed back, never re-tokenized" if seg_shown < 6 else ""), color=FG, fontsize=7)
    # probs
    axp = fig.add_subplot(gs[1, 1]); style(axp, "Mode probabilities")
    axp.bar(range(3), probs, color=MODE_COL); axp.set_ylim(0, 1); axp.set_xticks(range(3)); axp.set_xticklabels(["m1", "m2", "m3"])
    # telemetry
    axt = fig.add_subplot(gs[:, 2]); style(axt, "Displacement error vs time (m)")
    tt = np.arange(1, 31) * 0.1
    e_new = np.linalg.norm(mo_out["pts"][0, top].detach().numpy() - gt, axis=1); e_old = np.linalg.norm(op - gt, axis=1)
    k = 30 if t_play is None else t_play + 1
    if t_play is not None:
        axt.plot(tt[:k], e_old[:k], color=C_OLD, ls=":", lw=2, label=f"one-shot  now {e_old[k-1]:.2f}")
        axt.plot(tt[:k], e_new[:k], color=C_MOC, lw=2.5, label=f"MoCAR top-1  now {e_new[k-1]:.2f}")
        axt.legend(fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=FG, loc="upper left")
    else:
        axt.text(0.5, 0.5, "generating…\n(error revealed\nduring playback)", color=FG, ha="center", va="center", transform=axt.transAxes, fontsize=9)
    axt.set_xlim(0, 3); axt.set_ylim(0, max(2.0, float(max(e_new.max(), e_old.max())) * 1.1)); axt.set_xlabel("time into future (s)", color=FG, fontsize=8)
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=BG); plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--out", default="assets/mocar_simulation.gif")
    a = ap.parse_args(); cfg, mo, ol = load(a.config)
    te = SyntheticDrivingDataset(cfg["data"]["n_test"], seed=3); frames, durs, summary = [], [], []
    for ki, kind in enumerate(KINDS):
        i = int((te.d["kind"] == ki).nonzero()[0]); sc = {k: v[i] for k, v in te.d.items()}; b = {k: v[None] for k, v in sc.items()}
        with torch.no_grad():
            out = mo(b); olo = ol(b)
            _, _, ends = compose(*mo.tok.decode(out["pred_codes"][0]), torch.zeros(3, 3))
            out["ends"] = ends[None]; codes = out["pred_codes"][0]
        gt = sc["fut_pts"][1:]; en = (out["pts"][0, out["scores"][0].argmax()] - gt).norm(dim=-1).mean().item(); eo = (olo["pts"][0, olo["scores"][0].argmax()] - gt).norm(dim=-1).mean().item()
        summary.append(dict(scene=kind, test_index=i, mocar_top1_ADE=en, oneshot_top1_ADE=eo))
        for sgn in range(0, 7): frames.append(frame(sc, out, olo, codes, sgn, None, kind)); durs.append(500 if sgn < 6 else 900)
        for t in range(0, 30, 3): frames.append(frame(sc, out, olo, codes, 6, t, kind)); durs.append(180)
        frames.append(frame(sc, out, olo, codes, 6, 29, kind)); durs.append(1500)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)
    json.dump(summary, open("assets/simulation_scenes.json", "w"), indent=1); print(json.dumps(summary, indent=1)); print("frames", len(frames), a.out)
