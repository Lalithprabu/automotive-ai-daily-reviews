"""
train.py -- two-phase LDE training.

Phase 1 ("teacher pretrain"): a fresh BEVDetector is trained with plain
supervised class_balanced_bce_loss on the collaborator's own labeled BEV
scenes. This represents the collaborator's already-good, source-domain
detector (standard practice in UDA setups: start self-training from a
source-pretrained checkpoint). Its weights seed LDEModel.teacher.

Phase 2 (main LDE self-training loop): LDEModel.student starts from a
randomly-perturbed copy of that checkpoint (representing the ego's own,
not-yet-adapted local detector) and is trained ONLY via teacher pseudo-
labels through models.lde.LDEModel.self_training_loss -- no ego ground
truth is ever used here, matching the paper's "V2X link itself is the
supervision signal" framing. The teacher is EMA-updated from the student
after every step (Mean-Teacher).

Checkpoints and scalar history are saved to `assets/` for simulate.py to
render real (not random-init) frames from.
"""

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.detector import BEVDetector
from models.lde import LDEModel
from src.utils.data import generate_batch
from src.utils.losses import class_balanced_bce_loss


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def evaluate_recall_proxy(student, batch) -> float:
    """Fraction of the ego's OWN in-range true-object cells (ego_labels)
    that the student currently fires on at a 0.5 sigmoid threshold. Only
    ego_labels are used here (never as a training signal -- only for this
    read-only diagnostic), since those are the only cells where the ego's
    raw input actually carries real object signal."""
    with torch.no_grad():
        logits = student(batch["ego_bev"])["cls_logits"].squeeze(1)
        pred = (torch.sigmoid(logits) > 0.5).float()
        pos = batch["ego_labels"] > 0.5
        if pos.sum().item() == 0:
            return float("nan")
        return (pred[pos] > 0.5).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, args.config)) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    ckpt_dir = os.path.join(root, train_cfg["checkpoint_dir"])
    os.makedirs(ckpt_dir, exist_ok=True)

    history = {"phase1_loss": [], "phase2_loss": [], "phase2_threshold": [],
               "phase2_supervised_cells": [], "phase2_recall_proxy": [],
               "phase2_recall_steps": []}

    # ---------------- Phase 1: supervised teacher pretraining ----------------
    print("=== Phase 1: teacher pretraining (supervised, collaborator domain) ===")
    pretrain_net = BEVDetector(in_channels=model_cfg["in_channels"], feat_channels=model_cfg["feat_channels"])
    opt1 = torch.optim.Adam(pretrain_net.parameters(), lr=train_cfg["lr"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 1)

    for step in range(train_cfg["pretrain_steps"]):
        batch = generate_batch(train_cfg["batch_size"], generator=gen, **data_cfg)
        opt1.zero_grad()
        logits = pretrain_net(batch["collab_bev"])["cls_logits"].squeeze(1)
        loss = class_balanced_bce_loss(logits, batch["collab_labels"])
        loss.backward()
        opt1.step()
        history["phase1_loss"].append(loss.item())
        if step % 25 == 0 or step == train_cfg["pretrain_steps"] - 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}")

    # ---------------- Phase 2: main LDE self-training loop ----------------
    print("=== Phase 2: LDE Mean-Teacher self-training loop ===")
    model = LDEModel(
        in_channels=model_cfg["in_channels"],
        feat_channels=model_cfg["feat_channels"],
        budget_ratio=model_cfg["budget_ratio"],
        fov_radius_cells=model_cfg["fov_radius_cells"],
        ema_momentum=model_cfg["ema_momentum"],
        curriculum_start=model_cfg["curriculum_start"],
        curriculum_end=model_cfg["curriculum_end"],
        curriculum_steps=train_cfg["lde_steps"],
    )
    # Teacher = the pretrained, source-domain-competent detector.
    model.teacher.load_state_dict(pretrain_net.state_dict())
    for p in model.teacher.parameters():
        p.requires_grad_(False)

    # Student = ego's own, not-yet-adapted detector: start from the same
    # weights but perturbed, so there is real domain-adaptation work to do
    # and the loss starts high (mirrors phase 1's starting loss level).
    student_init = BEVDetector(in_channels=model_cfg["in_channels"], feat_channels=model_cfg["feat_channels"])
    student_init.load_state_dict(pretrain_net.state_dict())
    with torch.no_grad():
        for p in student_init.parameters():
            p.add_(torch.randn_like(p) * 0.9)
    model.student.load_state_dict(student_init.state_dict())

    opt2 = torch.optim.Adam(model.student.parameters(), lr=train_cfg["lr"])
    gen2 = torch.Generator().manual_seed(cfg["seed"] + 2)

    num_frames = cfg["simulate"]["num_frames"]
    ckpt_every = max(1, train_cfg["lde_steps"] // num_frames)

    for step in range(train_cfg["lde_steps"]):
        batch = generate_batch(train_cfg["batch_size"], generator=gen2, **data_cfg)

        ego_out = model.student(batch["ego_bev"])
        pseudo = model.build_pseudo_labels(batch["collab_bev"], batch["dx"], batch["dy"], step=step)

        opt2.zero_grad()
        loss = model.self_training_loss(ego_out["cls_logits"], pseudo["pseudo_labels"], pseudo["confident_mask"])
        loss.backward()
        opt2.step()
        model.update_teacher()

        history["phase2_loss"].append(loss.item())
        history["phase2_threshold"].append(pseudo["threshold"])
        history["phase2_supervised_cells"].append(pseudo["supervised_cells"])

        if step % train_cfg["eval_every"] == 0 or step == train_cfg["lde_steps"] - 1:
            recall = evaluate_recall_proxy(model.student, batch)
            history["phase2_recall_proxy"].append(recall)
            history["phase2_recall_steps"].append(step)
            print(f"  step {step:4d}  loss {loss.item():.4f}  thr {pseudo['threshold']:.3f}  "
                  f"sup_cells {pseudo['supervised_cells']:.0f}  recall~ {recall:.3f}")

        if step % ckpt_every == 0 or step == train_cfg["lde_steps"] - 1:
            torch.save({
                "step": step,
                "student": model.student.state_dict(),
                "teacher": model.teacher.state_dict(),
                "threshold": pseudo["threshold"],
                "supervised_cells": pseudo["supervised_cells"],
                "loss": loss.item(),
            }, os.path.join(ckpt_dir, f"ckpt_{step:04d}.pt"))

    log_path = os.path.join(root, train_cfg["log_path"])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved training history to {log_path}")
    print(f"Saved checkpoints to {ckpt_dir}")
    print(f"Phase 1 loss: {history['phase1_loss'][0]:.4f} -> {history['phase1_loss'][-1]:.4f}")
    print(f"Phase 2 loss: {history['phase2_loss'][0]:.4f} -> {history['phase2_loss'][-1]:.4f}")
    print(f"Phase 2 supervised_cells: {history['phase2_supervised_cells'][0]:.0f} -> "
          f"{history['phase2_supervised_cells'][-1]:.0f}")
    print(f"Phase 2 recall proxy: {history['phase2_recall_proxy'][0]:.3f} -> "
          f"{history['phase2_recall_proxy'][-1]:.3f}")


if __name__ == "__main__":
    main()
