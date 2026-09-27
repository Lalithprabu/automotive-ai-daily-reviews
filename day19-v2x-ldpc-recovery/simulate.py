"""
simulate.py

End-to-end pipeline demo on a held-out synthetic scenario:

  trajectory history -> ProbabilisticMotionPredictor -> predicted Gaussian
  -> BSMBitProbabilityLayer (quantize/serialize) -> bit probabilities
  -> LLRPriorHead -> prior LLR
  -> AWGN-corrupt a REAL encoded BSM codeword -> channel LLR
  -> LDPC BP decode attempt #1 (channel LLR only)
  -> if syndrome check fails (our CRC-fail surrogate): decode attempt #2
     (channel LLR + prior LLR)

Renders a real matplotlib animation to assets/v2x_ldpc_recovery_simulation.gif.

Run:
    python simulate.py
"""
from __future__ import annotations

import os
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter

from src.bsm import BSMState, encode_bsm, FIELD_ORDER, DEFAULT_FIELD_SPECS, field_bit_spans, MESSAGE_BITS
from src.dataset import generate_trajectory
from src.ldpc import build_ldpc_code, bp_decode
from src.channel import awgn_transmit, received_to_llr
from models.motion_predictor import ProbabilisticMotionPredictor
from models.bit_llr import BSMBitProbabilityLayer, LLRPriorHead

# ---- palette (this project's fixed visual style) ----
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
YELLOW = "#eda100"
BG = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GREY = "#d8d6d1"
RED = "#d64545"


def run_pipeline(cfg):
    pred_cfg = cfg["predictor"]
    sim_cfg = cfg["simulation"]
    ldpc_cfg = cfg["ldpc"]

    torch.manual_seed(sim_cfg["seed"])
    rng = np.random.default_rng(sim_cfg["seed"])

    # ---- load trained predictor ----
    ckpt = torch.load("checkpoints/motion_predictor.pt", map_location="cpu")
    model = ProbabilisticMotionPredictor(
        num_fields=pred_cfg["num_fields"], hidden_dim=pred_cfg["hidden_dim"],
        future_len=pred_cfg["future_len"], dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    bit_layer = BSMBitProbabilityLayer(max_exact_blocks=cfg["bit_llr"]["max_exact_blocks"])
    llr_head = LLRPriorHead()

    # ---- LDPC code (must match the one used elsewhere in this repo) ----
    code = build_ldpc_code(n=ldpc_cfg["n"], k_target=ldpc_cfg["k"], wc=ldpc_cfg["wc"],
                            wr=ldpc_cfg["wr"], seed=ldpc_cfg["seed"])

    spans = field_bit_spans()
    payload_bits = sum(b for _, _, b in DEFAULT_FIELD_SPECS.values())

    # ---- held-out synthetic scenario: one continuous trajectory, never seen in training ----
    history_len = pred_cfg["history_len"]
    num_messages = sim_cfg["num_messages"]
    traj = generate_trajectory(rng, length=history_len + num_messages, dt=0.1)

    eb_n0_db = sim_cfg["eb_n0_db"]

    frames = []
    n_seen = 0
    n_first_fail = 0
    n_second_recovered = 0
    gt_path = []

    for i in range(num_messages):
        history = traj[i:i + history_len]                 # [history_len, 5]
        target = traj[i + history_len]                     # [5] ground-truth next BSM state
        gt_path.append((target[0], target[1]))

        hist_t = torch.from_numpy(history).float().unsqueeze(0)  # [1, T_hist, 5]
        with torch.no_grad():
            out = model(hist_t)
        mu_phys = out["mu_phys"][0, 0].numpy()    # [5] first future step, physical units
        sigma_phys = out["sigma_phys"][0, 0].numpy()

        # ---- encode the REAL (ground-truth) BSM that is about to be transmitted ----
        state = BSMState(x=target[0], y=target[1], speed=target[2], heading=target[3], accel=target[4])
        payload_bits_arr = encode_bsm(state)  # [64] uint8
        message_bits = np.zeros(code.k, dtype=np.uint8)
        message_bits[:MESSAGE_BITS] = payload_bits_arr  # remaining (code.k - 64) bits stay 0 (reserved)
        cw = code.encode(message_bits)

        # ---- channel ----
        r = awgn_transmit(cw, eb_n0_db=eb_n0_db, code_rate=code.rate, rng=rng)
        llr = received_to_llr(r, eb_n0_db=eb_n0_db, code_rate=code.rate)
        initial_hard = (llr < 0).astype(np.uint8)

        # ---- attempt 1: channel LLR only ----
        decoded1, ok1, iters1 = bp_decode(code, llr, max_iters=ldpc_cfg["max_bp_iters"],
                                           alpha=ldpc_cfg["min_sum_alpha"])
        n_seen += 1

        status1 = np.where(initial_hard == cw, 0, 3)          # 0=grey (untouched), 3=red(still wrong)
        status1 = np.where((initial_hard != cw) & (decoded1 == cw), 2, status1)  # 2=green (BP-only fixed it)

        decoded2 = decoded1
        ok2 = ok1
        prior_llr_used = None
        if not ok1:
            n_first_fail += 1
            # ---- prediction -> bit probabilities -> prior LLR, for the 60 real payload bits ----
            prior = np.zeros(code.n)
            for f_idx, name in enumerate(FIELD_ORDER):
                start, nbits = spans[name]
                mu_f = torch.tensor([mu_phys[f_idx]], dtype=torch.float32)
                sigma_f = torch.tensor([sigma_phys[f_idx]], dtype=torch.float32)
                p1 = bit_layer(mu_f, sigma_f, name)[0]     # [nbits], P(bit=1), LSB first
                field_llr = llr_head(p1).numpy()            # [nbits]
                prior[start:start + nbits] = field_llr

            decoded2, ok2, iters2 = bp_decode(code, llr, prior_llr=prior,
                                               max_iters=ldpc_cfg["max_bp_iters"],
                                               alpha=ldpc_cfg["min_sum_alpha"])
            if ok2:
                n_second_recovered += 1
            prior_llr_used = prior

        status2 = status1.copy()
        if not ok1:
            fixed_by_prior = (status1 == 3) & (decoded2 == cw)
            status2 = np.where(fixed_by_prior, 1, status2)          # 1=orange (prediction-aided fix)
            still_wrong = (status1 == 3) & (decoded2 != cw)
            status2 = np.where(still_wrong, 3, status2)               # remains red

        live_rate = (n_second_recovered / n_first_fail) if n_first_fail > 0 else 0.0

        frames.append({
            "gt_path": list(gt_path),
            "mu_xy": (mu_phys[0], mu_phys[1]),
            "sigma_xy": (max(sigma_phys[0], 0.05), max(sigma_phys[1], 0.05)),
            "status1": status1,
            "status2": status2,
            "ok1": ok1, "ok2": ok2,
            "n_seen": n_seen, "n_first_fail": n_first_fail,
            "n_second_recovered": n_second_recovered, "live_rate": live_rate,
        })

    summary = {
        "num_messages": num_messages,
        "first_attempt_failures": n_first_fail,
        "second_attempt_recoveries": n_second_recovered,
        "final_recovery_rate": (n_second_recovered / n_first_fail) if n_first_fail > 0 else 0.0,
        "eb_n0_db": eb_n0_db,
    }
    return frames, summary, code


def render_gif(frames, summary, out_path):
    cmap = ListedColormap([GREY, ORANGE, GREEN, RED])  # status codes 0,1,2,3

    fig = plt.figure(figsize=(11, 6), dpi=110, facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1], width_ratios=[1.1, 1])
    ax_bev = fig.add_subplot(gs[0, 0])
    ax_bits = fig.add_subplot(gs[0, 1])
    ax_tel = fig.add_subplot(gs[1, :])

    for ax in (ax_bev, ax_bits, ax_tel):
        ax.set_facecolor(BG)

    fig.suptitle("Prediction-Aided V2X BSM Recovery -- synthetic reconstruction demo",
                 color=INK, fontsize=13, fontweight="bold", y=0.98)

    # ---- Panel A: BEV trajectory ----
    ax_bev.set_title("Panel A -- Ego trajectory: ground truth vs. predicted", color=INK2, fontsize=10)
    ax_bev.set_xlabel("x (m)", color=MUTED, fontsize=8)
    ax_bev.set_ylabel("y (m)", color=MUTED, fontsize=8)
    ax_bev.tick_params(colors=MUTED, labelsize=7)
    for spine in ax_bev.spines.values():
        spine.set_color(GREY)
    gt_line, = ax_bev.plot([], [], color=GREEN, lw=2.0, label="Ground truth path")
    pred_dot, = ax_bev.plot([], [], "o", color=BLUE, markersize=6, label="Predicted mean (next BSM)")
    unc_ellipse = Ellipse((0, 0), 0, 0, facecolor=BLUE, alpha=0.18, edgecolor=BLUE, lw=1.0)
    ax_bev.add_patch(unc_ellipse)
    ax_bev.legend(loc="upper left", fontsize=7, frameon=False, labelcolor=INK2)

    all_x = [p[0] for fr in frames for p in fr["gt_path"]]
    all_y = [p[1] for fr in frames for p in fr["gt_path"]]
    pad = 5
    ax_bev.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax_bev.set_ylim(min(all_y) - pad, max(all_y) + pad)

    # ---- Panel B: bitstream grid (16x16 = 256 codeword bits) ----
    ax_bits.set_title("Panel B -- Transmitted codeword bitstream (256 bits)", color=INK2, fontsize=10)
    grid_side = 16
    im = ax_bits.imshow(np.zeros((grid_side, grid_side)), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax_bits.set_xticks([]); ax_bits.set_yticks([])
    for spine in ax_bits.spines.values():
        spine.set_color(GREY)
    legend_elems = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=GREY, markersize=10, label="Transmitted OK"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=RED, markersize=10, label="Still wrong"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=ORANGE, markersize=10,
                    label="Fixed by prediction-aided pass"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=GREEN, markersize=10, label="Corrected by BP"),
    ]
    ax_bits.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.04),
                    fontsize=6.5, frameon=False, labelcolor=INK2, ncol=2)

    # ---- Panel C: telemetry ----
    ax_tel.axis("off")
    tel_text = ax_tel.text(0.02, 0.55, "", color=INK, fontsize=11, family="monospace", va="center")
    frame_label = ax_tel.text(0.98, 0.55, "", color=MUTED, fontsize=9, va="center", ha="right")

    n_msg = len(frames)
    total_subframes = n_msg * 2  # (post-attempt-1, post-attempt-2) per message

    def get_subframe(idx):
        msg_idx = idx // 2
        sub = idx % 2  # 0 = after attempt 1, 1 = after attempt 2 (final)
        return frames[msg_idx], sub, msg_idx

    def update(idx):
        fr, sub, msg_idx = get_subframe(idx)
        xs = [p[0] for p in fr["gt_path"]]
        ys = [p[1] for p in fr["gt_path"]]
        gt_line.set_data(xs, ys)
        mx, my = fr["mu_xy"]
        pred_dot.set_data([mx], [my])
        sx, sy = fr["sigma_xy"]
        unc_ellipse.center = (mx, my)
        unc_ellipse.width = 4 * sx   # ~2-sigma band
        unc_ellipse.height = 4 * sy

        status = fr["status1"] if sub == 0 else fr["status2"]
        grid = np.zeros(grid_side * grid_side)
        grid[:len(status)] = status
        im.set_data(grid.reshape(grid_side, grid_side))

        phase = "attempt 1 (channel LLR only)" if sub == 0 else "attempt 2 (prediction-aided)" if not fr["ok1"] else "attempt 1 succeeded -- no 2nd pass needed"
        result = "CRC OK" if (fr["ok1"] if sub == 0 else fr["ok2"]) else "CRC FAIL"
        tel_text.set_text(
            f"Messages seen:            {fr['n_seen']:3d}\n"
            f"First-attempt CRC fails:  {fr['n_first_fail']:3d}\n"
            f"2nd-attempt recoveries:   {fr['n_second_recovered']:3d}\n"
            f"Live recovery rate:       {fr['live_rate']*100:5.1f}%   (of first-attempt failures)"
        )
        frame_label.set_text(f"message {msg_idx+1}/{n_msg}  |  {phase}  |  {result}")
        return gt_line, pred_dot, unc_ellipse, im, tel_text, frame_label

    anim = FuncAnimation(fig, update, frames=total_subframes, blit=False)
    writer = PillowWriter(fps=cfg_fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    return total_subframes


if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg_fps = cfg["simulation"]["fps"]

    print("=" * 70)
    print("Day 19 -- end-to-end pipeline simulation on a held-out scenario")
    print("=" * 70)

    frames, summary, code = run_pipeline(cfg)

    print(f"LDPC code: n={code.n}, k={code.k}, rate={code.rate:.4f}")
    print(f"Eb/N0 = {summary['eb_n0_db']} dB, messages = {summary['num_messages']}")
    print(f"First-attempt CRC (syndrome) failures: {summary['first_attempt_failures']}")
    print(f"Second-attempt (prediction-aided) recoveries: {summary['second_attempt_recoveries']}")
    print(f"Final recovery rate (of first-attempt failures): {summary['final_recovery_rate']*100:.1f}%")

    os.makedirs("assets", exist_ok=True)
    out_path = "assets/v2x_ldpc_recovery_simulation.gif"
    n_frames = render_gif(frames, summary, out_path)

    # ---- verify the GIF is valid ----
    from PIL import Image
    im = Image.open(out_path)
    n_pil_frames = 0
    try:
        while True:
            im.seek(n_pil_frames)
            n_pil_frames += 1
    except EOFError:
        pass
    print("-" * 70)
    print(f"GIF saved to {out_path}")
    print(f"GIF frames written by animator: {n_frames}, frames verified by Pillow: {n_pil_frames}")
    print(f"GIF dimensions: {im.size}, file size: {os.path.getsize(out_path)/1024:.1f} KB")
