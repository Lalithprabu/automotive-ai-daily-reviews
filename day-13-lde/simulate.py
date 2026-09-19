"""
simulate.py -- renders a 3-panel BEV + telemetry-strip GIF from REAL model
checkpoints captured by train.py (never from a random-init model).

Panels (left to right), all in the ego's own BEV grid:
  1. Ego Alone (native FoV)       -- static: what the ego's own onboard
     sensing radius can ever see, independent of training.
  2. Collaborator -> Gated V2X    -- per-frame REAL teacher pseudo-label
     confidence, after AdaptiveFeatureGate (bandwidth budget) + FoVAligner
     (warp + range mask), at that checkpoint's curriculum threshold.
  3. Ego After LDE Adaptation     -- per-frame REAL student sigmoid
     confidence on the (fixed) demo ego_bev, at that checkpoint.

Bottom telemetry strip: curriculum threshold (declining) and
supervised-cell count (growing) over the full phase-2 training history,
with a marker tracking the current frame's training step.
"""

import argparse
import glob
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.detector import BEVDetector
from models.lde import LDEModel
from src.utils.data import generate_scene

# ---- palette (matches every other day in this series) ----
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def load_checkpoints(ckpt_dir: str):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")))
    return [torch.load(p, map_location="cpu") for p in paths]


def build_model_from_ckpt(ckpt, model_cfg, curriculum_steps):
    model = LDEModel(
        in_channels=model_cfg["in_channels"],
        feat_channels=model_cfg["feat_channels"],
        budget_ratio=model_cfg["budget_ratio"],
        fov_radius_cells=model_cfg["fov_radius_cells"],
        ema_momentum=model_cfg["ema_momentum"],
        curriculum_start=model_cfg["curriculum_start"],
        curriculum_end=model_cfg["curriculum_end"],
        curriculum_steps=curriculum_steps,
    )
    model.student.load_state_dict(ckpt["student"])
    model.teacher.load_state_dict(ckpt["teacher"])
    model.eval()
    return model


def render_frame(ckpt, model_cfg, curriculum_steps, demo, history, frame_idx, total_frames):
    model = build_model_from_ckpt(ckpt, model_cfg, curriculum_steps)
    step = ckpt["step"]

    ego_bev = demo["ego_bev"].unsqueeze(0)
    collab_bev = demo["collab_bev"].unsqueeze(0)
    dx = torch.tensor([demo["dx"]])
    dy = torch.tensor([demo["dy"]])

    with torch.no_grad():
        student_conf = torch.sigmoid(model.student(ego_bev)["cls_logits"])[0, 0].numpy()
        pseudo = model.build_pseudo_labels(collab_bev, dx, dy, step=step)
        pseudo_conf = pseudo["pseudo_conf"][0, 0].numpy()
        confident_mask = pseudo["confident_mask"][0, 0].numpy()
        threshold = pseudo["threshold"]
        supervised_cells = pseudo["supervised_cells"]

    H, W = demo["world_labels"].shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    ego_range = model_cfg["_ego_range"]

    fig = plt.figure(figsize=(11.5, 5.3), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[3.0, 1.0], hspace=0.55, wspace=0.35,
                           left=0.05, right=0.98, top=0.86, bottom=0.12)

    # ---------------- Panel 1: Ego Alone ----------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(SURFACE)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(cx + ego_range * np.cos(theta), cy + ego_range * np.sin(theta), color=INK_MUTED, lw=1.2, ls="--")
    world_obj = demo["world_labels"].nonzero(as_tuple=False).numpy()
    ego_obj = demo["ego_labels"].nonzero(as_tuple=False).numpy()
    if len(world_obj):
        ax1.scatter(world_obj[:, 1], world_obj[:, 0], s=70, facecolors="none", edgecolors=INK_MUTED, lw=1.2)
    if len(ego_obj):
        ax1.scatter(ego_obj[:, 1], ego_obj[:, 0], s=70, c=BLUE, marker="s", label="detected (native)")
    ax1.scatter([cx], [cy], marker="^", s=90, c=INK, zorder=5)
    ax1.set_xlim(0, W - 1); ax1.set_ylim(H - 1, 0)
    ax1.set_title("1. Ego Alone (native FoV)", fontsize=10, color=INK, fontweight="bold")
    ax1.set_xticks([]); ax1.set_yticks([])
    for s in ax1.spines.values():
        s.set_color(INK_MUTED); s.set_linewidth(0.6)

    # ---------------- Panel 2: Collaborator -> Gated V2X ----------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(SURFACE)
    ax2.imshow(pseudo_conf, cmap="Greens", vmin=0, vmax=1, extent=(0, W - 1, H - 1, 0), alpha=0.85)
    conf_cells = np.argwhere(confident_mask > 0.5)
    if len(conf_cells):
        ax2.scatter(conf_cells[:, 1], conf_cells[:, 0], s=14, c=ORANGE, marker="o", lw=0,
                    label="passes curriculum")
    ax2.scatter([cx], [cy], marker="^", s=90, c=INK, zorder=5)
    ax2.set_xlim(0, W - 1); ax2.set_ylim(H - 1, 0)
    ax2.set_title(f"2. Collaborator -> Gated V2X\nthreshold={threshold:.2f}  shared={int(confident_mask.sum())}",
                  fontsize=9.5, color=INK, fontweight="bold")
    ax2.set_xticks([]); ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_color(INK_MUTED); s.set_linewidth(0.6)

    # ---------------- Panel 3: Ego After LDE Adaptation ----------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(SURFACE)
    ax3.imshow(student_conf, cmap="Blues", vmin=0, vmax=1, extent=(0, W - 1, H - 1, 0), alpha=0.85)
    if len(world_obj):
        ax3.scatter(world_obj[:, 1], world_obj[:, 0], s=70, facecolors="none", edgecolors=INK, lw=1.4)
    ax3.scatter([cx], [cy], marker="^", s=90, c=INK, zorder=5)
    ax3.set_xlim(0, W - 1); ax3.set_ylim(H - 1, 0)
    ax3.set_title(f"3. Ego After LDE Adaptation\nstep {step}", fontsize=10, color=INK, fontweight="bold")
    ax3.set_xticks([]); ax3.set_yticks([])
    for s in ax3.spines.values():
        s.set_color(INK_MUTED); s.set_linewidth(0.6)

    # ---------------- Telemetry strip ----------------
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_facecolor(SURFACE)
    steps = np.arange(len(history["phase2_threshold"]))
    ax4.plot(steps, history["phase2_threshold"], color=AQUA, lw=1.8, label="curriculum threshold")
    ax4.set_ylabel("threshold", color=AQUA, fontsize=8)
    ax4.tick_params(axis="y", labelcolor=AQUA, labelsize=7)
    ax4.set_xlabel("training step", fontsize=8, color=INK_SECONDARY)
    ax4.tick_params(axis="x", labelsize=7, colors=INK_SECONDARY)

    ax4b = ax4.twinx()
    ax4b.plot(steps, history["phase2_supervised_cells"], color=ORANGE, lw=1.8, label="supervised cells")
    ax4b.set_ylabel("supervised cells", color=ORANGE, fontsize=8)
    ax4b.tick_params(axis="y", labelcolor=ORANGE, labelsize=7)

    ax4.axvline(step, color=INK, lw=1.2, ls=":")
    for s in ax4.spines.values():
        s.set_color(INK_MUTED); s.set_linewidth(0.6)
    for s in ax4b.spines.values():
        s.set_visible(False)

    fig.suptitle("LDE: Collaborative-Perception Self-Training (bandwidth-gated V2X)",
                 fontsize=12, color=INK, fontweight="bold", y=0.97)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=SURFACE)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, args.config)) as f:
        cfg = yaml.safe_load(f)

    ckpt_dir = os.path.join(root, cfg["train"]["checkpoint_dir"])
    log_path = os.path.join(root, cfg["train"]["log_path"])
    out_path = os.path.join(root, cfg["simulate"]["output_path"])

    ckpts = load_checkpoints(ckpt_dir)
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir} -- run train.py first.")
    with open(log_path) as f:
        history = json.load(f)

    model_cfg = dict(cfg["model"])
    model_cfg["_ego_range"] = cfg["data"]["ego_range"]

    demo_gen = torch.Generator().manual_seed(cfg["seed"] + 999)
    demo = generate_scene(generator=demo_gen, **cfg["data"])

    num_frames = min(cfg["simulate"]["num_frames"], len(ckpts))
    idxs = np.linspace(0, len(ckpts) - 1, num_frames).round().astype(int)
    chosen = [ckpts[i] for i in idxs]

    print(f"Rendering {len(chosen)} frames from {len(ckpts)} checkpoints...")
    frames = []
    for i, ckpt in enumerate(chosen):
        frames.append(render_frame(ckpt, model_cfg, cfg["train"]["lde_steps"], demo, history, i, len(chosen)))
        if i % 10 == 0:
            print(f"  frame {i}/{len(chosen)} (step {ckpt['step']})")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    duration_ms = int(1000 / cfg["simulate"]["fps"])
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print(f"Saved GIF ({len(frames)} frames) to {out_path}")


if __name__ == "__main__":
    main()
