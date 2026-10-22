"""Render the 'AI in action' simulation as a GIF from the REAL trained run.

Three panels per frame (one frame per federated round):
  1. BEV: observed path (grey), ground truth (green), frozen generic base (red),
     LoRA-FedFT global model (blue)  -- for three held-out pedestrians.
  2. Radio: 20 vehicles upload to the edge server; packet area is proportional
     to payload bytes (tiny blue LoRA packets vs. huge purple full-FT packets).
  3. Telemetry: validation ADE (m) vs cumulative uplink MB, log-x.

Usage:  python train.py && python simulate.py [--out assets/simulation.gif]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from src.lora_fedft.data import HOR, OBS, to_world

C = dict(obs="#6b7280", gt="#16a34a", base="#dc2626", lora="#2563eb", full="#7c3aed", bg="#0f172a", fg="#e2e8f0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="assets/simulation.gif")
    ap.add_argument("--run", default="lora_F5", help="which LoRA run to show in blue")
    ap.add_argument("--fps", type=int, default=4)
    a = ap.parse_args()
    res = json.load(open(Path(a.results) / "results.json"))
    tr = np.load(Path(a.results) / "demo_trace.npz")
    lora = res["runs"][a.run]
    full = res["runs"]["full_ft_F5" if a.run.endswith("F5") else "full_ft"]   # match local-epoch budget
    scratch = res["runs"]["fed_scratch"]
    n_veh = res["config"]["data"]["num_clients"]
    rounds = len(lora["history"]["round"]) - 1
    names = {1: "curving", 2: "stop-and-go", 3: "crossing"}
    n_scene = len(tr["idx"])

    plt.rcParams.update({"text.color": C["fg"], "axes.labelcolor": C["fg"], "xtick.color": C["fg"], "ytick.color": C["fg"]})
    fig = plt.figure(figsize=(15, 6.2), facecolor=C["bg"])
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1.15, 1.6], wspace=0.35, hspace=0.3, left=0.03, right=0.98, top=0.88, bottom=0.1)
    scene_ax = [fig.add_subplot(gs[:, i]) for i in range(3)]
    rad = fig.add_subplot(gs[:, 3]); tel = fig.add_subplot(gs[:, 4])
    for ax in scene_ax + [rad, tel]:
        ax.set_facecolor("#111827")
        for s in ax.spines.values(): s.set_color("#334155")

    ang = np.linspace(0, 2 * np.pi, n_veh, endpoint=False)
    veh = np.stack([np.cos(ang), np.sin(ang)], 1) * 0.85
    mb = lambda b: b / 1e6
    lp, fp = lora["payload_bytes_per_upload"], full["payload_bytes_per_upload"]

    def world_pred(arr, k):  # arr: (n_scene, HOR, 2) canonical
        return to_world(arr, tr["R"], tr["origin"])[k]

    def draw(frame):
        r = min(frame, rounds)
        for k, ax in enumerate(scene_ax):
            ax.clear(); ax.set_facecolor("#111827"); ax.set_aspect("equal")
            w = tr["world"][k]
            ax.plot(*w[:OBS].T, "o-", c=C["obs"], ms=3, lw=2, label="observed (4 s)")
            ax.plot(*w[OBS - 1:].T, "-", c=C["gt"], lw=2.5, label="ground truth (6 s)")
            p0, pr = world_pred(tr["pred_round0"], k), world_pred(tr[f"pred_{a.run}"][r], k)
            ax.plot(*np.vstack([w[OBS - 1], p0]).T, "--", c=C["base"], lw=2, label="frozen base")
            ax.plot(*np.vstack([w[OBS - 1], pr]).T, "-", c=C["lora"], lw=2.5, label=f"LoRA-FedFT r{r}")
            ax.scatter(*w[OBS - 1], c="white", s=30, zorder=5)
            err0 = np.linalg.norm(p0 - w[OBS:], axis=1).mean(); errr = np.linalg.norm(pr - w[OBS:], axis=1).mean()
            ax.set_title(f"{names.get(int(tr['style'][k]), '?')}   ADE base {err0:.1f} m -> {errr:.1f} m", fontsize=9, color=C["fg"])
            ax.set_xticks([]); ax.set_yticks([])
            allp = np.vstack([w, p0, pr]); c = allp.mean(0); span = max(np.ptp(allp, 0).max() / 2 + 1, 4)
            ax.set_xlim(c[0] - span, c[0] + span); ax.set_ylim(c[1] - span, c[1] + span)
            if k == 0: ax.legend(loc="lower left", fontsize=7, facecolor="#1e293b", edgecolor="#334155", labelcolor=C["fg"])

        rad.clear(); rad.set_facecolor("#111827"); rad.set_xlim(-1.5, 1.5); rad.set_ylim(-1.9, 1.7); rad.set_aspect("equal")
        rad.set_xticks([]); rad.set_yticks([])
        for v in veh:
            rad.plot([v[0], 0], [v[1], 0], c="#334155", lw=0.8, zorder=1)
            rad.scatter(*(v * 0.6), marker="s", s=max(lp / fp, 0.003) * 700, c=C["lora"], zorder=3)   # packet area ~ bytes
        rad.scatter(*veh.T, c=C["fg"], s=18, zorder=4)
        rad.scatter(0, 0, marker="s", s=200, c="#f59e0b", zorder=5)
        # reference packets drawn at the SAME area scale
        rad.scatter(-0.9, 1.45, marker="s", s=700, c=C["full"]); rad.text(-0.62, 1.45, f"full-FT packet\n{fp/1e3:.0f} KB", va="center", fontsize=7.5, color=C["fg"])
        rad.scatter(0.55, 1.45, marker="s", s=max(lp / fp, 0.003) * 700, c=C["lora"]); rad.text(0.7, 1.45, f"LoRA\n{lp/1e3:.0f} KB", va="center", fontsize=7.5, color=C["fg"])
        rad.set_title(f"Round {r}/{rounds}: {n_veh} vehicles -> edge server", fontsize=9, color=C["fg"])
        rad.text(0, -1.45, f"cumulative uplink\nLoRA {mb(lora['history']['uplink_bytes'][r]):.1f} MB  |  full FT {mb(full['history']['uplink_bytes'][r]):.1f} MB",
                 ha="center", va="center", fontsize=8, color=C["fg"])

        tel.clear(); tel.set_facecolor("#111827")
        for h, col, lab in ((scratch, "#94a3b8", "fed from scratch"), (full, C["full"], "federated full FT (same F)"), (lora, C["lora"], f"LoRA-FedFT ({a.run[-2:]})")):
            x = [max(mb(b), 0.01) for b in h["history"]["uplink_bytes"][: r + 1]]
            tel.plot(x, h["history"]["ADE"][: r + 1], "-o", c=col, ms=3, lw=2, label=lab)
        tel.axhline(res["centralized"]["ADE"], c=C["gt"], ls=":", lw=1.2, label="centralized")
        tel.axhline(res["zero_shot"]["ADE"], c=C["base"], ls=":", lw=1.2, label="frozen base")
        tel.set_xscale("log"); tel.set_xlim(0.01, mb(scratch["total_uplink_bytes"]) * 1.2); tel.set_ylim(0, res["zero_shot"]["ADE"] * 1.1)
        tel.set_xlabel("cumulative uplink (MB, log)"); tel.set_ylabel("validation ADE (m)")
        tel.set_title("Accuracy per uplink byte", fontsize=9, color=C["fg"])
        tel.legend(fontsize=7, facecolor="#1e293b", edgecolor="#334155", labelcolor=C["fg"], loc="upper right")
        fig.suptitle("LoRA federated fine-tuning for pedestrian trajectory prediction  |  SYNTHETIC data, this repo's reconstruction", color=C["fg"], fontsize=11)

    frames = list(range(rounds + 1)) + [rounds] * 3
    Path(a.out).parent.mkdir(exist_ok=True, parents=True)
    FuncAnimation(fig, draw, frames=frames).save(a.out, writer=PillowWriter(fps=a.fps), dpi=70, savefig_kwargs={"facecolor": C["bg"]})
    print("wrote", a.out)


if __name__ == "__main__":
    main()
