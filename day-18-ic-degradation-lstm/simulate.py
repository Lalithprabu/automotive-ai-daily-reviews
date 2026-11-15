"""
simulate.py
==========================================================================
Real-time synthetic BMS-style simulation, built from the REAL trained
ChargeIC-LSTM checkpoint (not mocked). Picks one held-out TEST cell
(genuinely unseen during training) and animates it cycle-by-cycle over
`simulate.cycles_to_show` cycles:

  Panel 1 (top-left):  Live charging-feed -- the V/I/T charging-phase
                        signal for the CURRENT cycle, as if streamed live
                        off the BMS for that fast-charge window.
  Panel 2 (top-right): IC-curve overlay -- ground-truth discharge IC curve
                        vs the model's PREDICTED curve for the current
                        cycle, redrawn every cycle.
  Panel 3 (bottom):    Degradation-indicator telemetry strip -- running
                        true-vs-predicted SOH trend across all cycles seen
                        so far (a BMS dashboard-style trend line), plus
                        true-vs-predicted peak-IC-height.

Output: battery_ic_simulation.gif

DISCLOSURE: "real-time" here means the model performs one real forward
pass per cycle shown (no lookahead, no cheating), and the animation
advances cycle-by-cycle the way a BMS would see completed fast-charge
cycles roll in. It is NOT a sub-cycle, sample-by-sample live stream of a
physical BMS -- that level of fidelity is out of scope for this
reconstruction. This is this project's own simplification, disclosed here
and in the README.
==========================================================================
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import load_config, load_checkpoint, build_model  # noqa: E402
from data.synthetic_battery import generate_cells, split_cells  # noqa: E402


def main():
    cfg = load_config()
    ckpt_path = os.path.join(_ROOT, cfg["train"]["checkpoint_path"])
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}. Run `python src/train.py` first."

    ckpt = load_checkpoint(ckpt_path)
    train_cfg = ckpt["config"]  # config as it was AT TRAIN TIME (source of truth for norm stats)
    mean, std = ckpt["norm_mean"], ckpt["norm_std"]

    model = build_model(train_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from {ckpt_path} (trained epoch {ckpt.get('epoch', '?')})")

    # --- Rebuild the exact same synthetic split used at train time ----------
    seed = train_cfg.get("seed", 42)
    cells = generate_cells(train_cfg, seed=seed)
    _, _, test_cells = split_cells(cells, train_cfg, seed=seed)
    assert len(test_cells) > 0, "No held-out test cells available for simulation."

    # Pick the test cell with the most cycles for a richer animation.
    sim_cell = max(test_cells, key=lambda c: len(c["cycles"]))
    print(
        f"Simulating on held-out TEST cell_id={sim_cell['cell_id']} "
        f"(c_rate={sim_cell['c_rate']:.2f}C, {len(sim_cell['cycles'])} cycles available)"
    )

    scfg = cfg["simulate"]
    n_show = min(scfg["cycles_to_show"], len(sim_cell["cycles"]))
    v_grid = sim_cell["v_grid"]
    t_axis = np.arange(train_cfg["data"]["charge_len_T"])

    # --- Run the REAL model forward pass for every cycle up front -----------
    ic_preds, soh_preds = [], []
    with torch.no_grad():
        for k in range(n_show):
            x_raw = sim_cell["charge"][k]              # (T, 3)
            x_norm = (x_raw - mean) / std                # normalize w/ TRAIN stats
            x_t = torch.from_numpy(x_norm).float().unsqueeze(0)  # (1, T, 3)
            ic_pred, soh_pred = model(x_t)                # (1, M), (1,)
            ic_preds.append(ic_pred.squeeze(0).numpy())
            soh_preds.append(float(soh_pred.item()))
    ic_preds = np.stack(ic_preds, axis=0)   # (n_show, M)
    soh_preds = np.array(soh_preds)          # (n_show,)

    true_soh = sim_cell["soh"][:n_show]
    true_ic = sim_cell["ic"][:n_show]
    true_peak_height = true_ic.max(axis=1)
    pred_peak_height = ic_preds.max(axis=1)

    # --- Figure layout: 2 rows -> [feed | ic overlay] on top, telemetry below
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 0.9], hspace=0.38, wspace=0.32)
    ax_feed = fig.add_subplot(gs[0, 0])
    ax_ic = fig.add_subplot(gs[0, 1])
    ax_telem = fig.add_subplot(gs[1, :])

    fig.suptitle(
        f"ChargeIC-LSTM -- Held-out Test Cell {sim_cell['cell_id']} "
        f"({sim_cell['c_rate']:.2f}C fast-charge)  |  synthetic reconstruction, not paper data",
        fontsize=11,
    )

    # Panel 1: live charging feed (V on left axis, I & T on right axis)
    ax_feed.set_title("Live Charging Feed (this cycle)")
    ax_feed.set_xlabel("timestep (10s each)")
    ax_feed.set_ylabel("Voltage (V)", color="tab:blue")
    line_v, = ax_feed.plot([], [], color="tab:blue", label="Voltage")
    ax_feed.set_xlim(0, len(t_axis))
    ax_feed.set_ylim(2.9, 4.4)
    ax_feed.tick_params(axis="y", labelcolor="tab:blue")

    ax_feed2 = ax_feed.twinx()
    line_i, = ax_feed2.plot([], [], color="tab:red", alpha=0.8, label="Current")
    line_t, = ax_feed2.plot([], [], color="tab:orange", alpha=0.8, label="Temperature")
    ax_feed2.set_ylabel("Current (A-like) / Temp (deg C offset)")
    all_i = sim_cell["charge"][:n_show, :, 1]
    all_t = sim_cell["charge"][:n_show, :, 2]
    ax_feed2.set_ylim(min(all_i.min(), all_t.min()) - 1, max(all_i.max(), all_t.max()) + 1)
    lines_feed = [line_v, line_i, line_t]
    labels_feed = ["Voltage", "Current", "Temperature"]
    ax_feed.legend(lines_feed, labels_feed, loc="lower right", fontsize=8)

    # Panel 2: IC curve overlay (ground truth vs predicted)
    ax_ic.set_title("Discharge IC Curve: Ground Truth vs Predicted")
    ax_ic.set_xlabel("Voltage (V)")
    ax_ic.set_ylabel("dQ/dV")
    line_ic_true, = ax_ic.plot([], [], color="black", lw=2, label="Ground truth (synthetic)")
    line_ic_pred, = ax_ic.plot([], [], color="tab:green", lw=2, ls="--", label="Model prediction")
    ax_ic.set_xlim(v_grid.min(), v_grid.max())
    ax_ic.set_ylim(0, max(true_ic.max(), ic_preds.max()) * 1.15)
    ax_ic.legend(loc="upper left", fontsize=8)

    # Panel 3: degradation telemetry strip (true vs pred SOH, true vs pred peak height)
    ax_telem.set_title("Degradation Telemetry: True vs Predicted SOH & Peak IC Height (cycles seen so far)")
    ax_telem.set_xlabel("Cycle index")
    ax_telem.set_ylabel("SOH (fraction)")
    line_soh_true, = ax_telem.plot([], [], color="black", lw=1.8, label="True SOH")
    line_soh_pred, = ax_telem.plot([], [], color="tab:green", lw=1.8, ls="--", label="Predicted SOH")
    ax_telem.set_xlim(0, n_show)
    ax_telem.set_ylim(min(true_soh.min(), soh_preds.min()) - 0.03, max(true_soh.max(), soh_preds.max()) + 0.03)

    ax_telem2 = ax_telem.twinx()
    ax_telem2.set_ylabel("Peak IC height")
    line_peak_true, = ax_telem2.plot([], [], color="tab:blue", lw=1.2, alpha=0.7, label="True peak height")
    line_peak_pred, = ax_telem2.plot([], [], color="tab:purple", lw=1.2, alpha=0.7, ls=":", label="Pred peak height")
    ax_telem2.set_ylim(
        min(true_peak_height.min(), pred_peak_height.min()) - 0.1,
        max(true_peak_height.max(), pred_peak_height.max()) + 0.1,
    )
    lines_telem = [line_soh_true, line_soh_pred, line_peak_true, line_peak_pred]
    labels_telem = ["True SOH", "Pred SOH", "True peak IC", "Pred peak IC"]
    ax_telem.legend(lines_telem, labels_telem, loc="upper right", fontsize=8)

    cycle_text = ax_ic.text(
        0.98, 0.95, "", transform=ax_ic.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85),
    )

    def update(frame_idx: int):
        k = frame_idx
        # Panel 1
        line_v.set_data(t_axis, sim_cell["charge"][k, :, 0])
        line_i.set_data(t_axis, sim_cell["charge"][k, :, 1])
        line_t.set_data(t_axis, sim_cell["charge"][k, :, 2])

        # Panel 2
        line_ic_true.set_data(v_grid, true_ic[k])
        line_ic_pred.set_data(v_grid, ic_preds[k])

        # Panel 3 (cumulative up to k)
        xs = np.arange(k + 1)
        line_soh_true.set_data(xs, true_soh[: k + 1])
        line_soh_pred.set_data(xs, soh_preds[: k + 1])
        line_peak_true.set_data(xs, true_peak_height[: k + 1])
        line_peak_pred.set_data(xs, pred_peak_height[: k + 1])

        cycle_text.set_text(
            f"Cycle {k}/{n_show - 1}\n"
            f"True SOH: {true_soh[k]:.3f}  Pred SOH: {soh_preds[k]:.3f}\n"
            f"True peak V: {v_grid[np.argmax(true_ic[k])]:.3f}  "
            f"Pred peak V: {v_grid[np.argmax(ic_preds[k])]:.3f}"
        )
        return (line_v, line_i, line_t, line_ic_true, line_ic_pred,
                line_soh_true, line_soh_pred, line_peak_true, line_peak_pred, cycle_text)

    anim = FuncAnimation(fig, update, frames=n_show, interval=1000 / scfg["fps"], blit=False)

    out_path = os.path.join(_ROOT, scfg["output_path"])
    writer = PillowWriter(fps=scfg["fps"])
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"Saved simulation GIF to {out_path} ({n_show} frames @ {scfg['fps']} fps)")


if __name__ == "__main__":
    main()
