#!/usr/bin/env python3
"""
train.py -- trains TWO predictor instances on the synthetic interactive-merge
dataset and reports real, measured metrics:

  1. `conditioned_model`  (AnchorConditionedPredictor, trained with cond sampled
     continuously) -- this is the "New Way (INTERACT)" predictor.
  2. `unconditional_model` (same architecture, but cond is forced to zero at both
     train AND eval time) -- this is the "Old Way A (non-interactive predict-then-
     plan)" baseline: it learns the best it can do while blind to ego intent, i.e.
     the marginal/average other-agent future.

Both are then evaluated on an anchor-bucketed validation set, split into an
"interactive" bucket (|assertiveness| > 0.3, i.e. anchors that meaningfully commit
one way or the other) and a "neutral" bucket (hold_lane_delay, |a| <= 0.3), to
measure the empirical claim this project can actually test: does anchor
conditioning help MOST where the scenario is most interactive? All numbers printed
and saved to checkpoints/metrics.json are real, measured outputs of this run --
never hand-picked.

Usage:
    python3 train.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.synthetic_interactive_scenario import InteractiveMergeDataset, collate_episodes
from src.models.anchor_conditioned_predictor import AnchorConditionedPredictor
from src.models.anchor_generator import AnchorGenerator


def set_seed(seed: int):
    torch.manual_seed(seed)


def ade_fde(pred: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    """Average / Final Displacement Error over a batch.
    pred, gt: [B, T, 2]
    """
    disp = torch.norm(pred - gt, dim=-1)     # [B, T]
    ade = disp.mean().item()
    fde = disp[:, -1].mean().item()
    return ade, fde


def run_epoch(model, loader, optimizer, force_zero_cond: bool, train: bool) -> float:
    model.train(mode=train)
    total_loss, n_batches = 0.0, 0
    loss_fn = nn.MSELoss()
    for batch in loader:
        cond = torch.zeros_like(batch["cond"]) if force_zero_cond else batch["cond"]
        with torch.set_grad_enabled(train):
            pred = model(batch["ego_hist"], batch["other_hist"], cond)
            loss = loss_fn(pred, batch["other_future"])
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_bucketed(model, dataset, force_zero_cond: bool, interactive_thresh: float = 0.3) -> dict:
    """Evaluates ADE/FDE split by |assertiveness| bucket over an entire (anchor-mode) dataset."""
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_episodes)
    batch = next(iter(loader))
    cond = torch.zeros_like(batch["cond"]) if force_zero_cond else batch["cond"]
    pred = model(batch["ego_hist"], batch["other_hist"], cond)   # [N, T, 2]
    gt = batch["other_future"]
    a = batch["assertiveness"]

    interactive_mask = a.abs() > interactive_thresh
    neutral_mask = ~interactive_mask

    results = {}
    for name, mask in (("interactive", interactive_mask), ("neutral", neutral_mask)):
        if mask.sum() == 0:
            continue
        ade, fde = ade_fde(pred[mask], gt[mask])
        results[name] = {"ade": ade, "fde": fde, "n": int(mask.sum())}
    ade_all, fde_all = ade_fde(pred, gt)
    results["all"] = {"ade": ade_all, "fde": fde_all, "n": len(dataset)}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    sc = cfg["scenario"]
    mc = cfg["model"]
    ac = cfg["anchors"]

    # --- datasets ---
    train_ds = InteractiveMergeDataset(
        n_samples=mc["train_samples"], seed=cfg["seed"], history_len=sc["history_len"],
        future_len=sc["future_len"], base_speed=sc["base_speed"], k_speed=sc["k_speed_reaction"],
        k_lateral=sc["k_lateral_reaction"], noise_std=sc["noise_std"], cond_mode="continuous",
    )
    val_ds = InteractiveMergeDataset(
        n_samples=mc["val_samples"], seed=cfg["seed"] + 1, history_len=sc["history_len"],
        future_len=sc["future_len"], base_speed=sc["base_speed"], k_speed=sc["k_speed_reaction"],
        k_lateral=sc["k_lateral_reaction"], noise_std=sc["noise_std"], cond_mode="continuous",
    )

    anchor_gen = AnchorGenerator()
    anchors = anchor_gen.generate(gap_to_merge=ac["default_gap_to_merge"], lane_width=ac["default_lane_width"])
    anchor_conds = [a.cond for a in anchors]
    anchor_names = [a.name for a in anchors]
    val_anchor_ds = InteractiveMergeDataset(
        n_samples=mc["val_anchor_samples_per_anchor"] * len(anchors), seed=cfg["seed"] + 2,
        history_len=sc["history_len"], future_len=sc["future_len"], base_speed=sc["base_speed"],
        k_speed=sc["k_speed_reaction"], k_lateral=sc["k_lateral_reaction"], noise_std=sc["noise_std"],
        cond_mode="anchors", anchor_conds=anchor_conds, anchor_names=anchor_names,
    )

    train_loader = DataLoader(train_ds, batch_size=mc["batch_size"], shuffle=True, collate_fn=collate_episodes)
    val_loader = DataLoader(val_ds, batch_size=mc["batch_size"], shuffle=False, collate_fn=collate_episodes)

    # --- models ---
    conditioned_model = AnchorConditionedPredictor(
        state_dim=sc["state_dim"], cond_dim=sc["cond_dim"], hidden_dim=mc["hidden_dim"],
        future_len=sc["future_len"], future_dim=sc["future_dim"],
    )
    unconditional_model = AnchorConditionedPredictor(
        state_dim=sc["state_dim"], cond_dim=sc["cond_dim"], hidden_dim=mc["hidden_dim"],
        future_len=sc["future_len"], future_dim=sc["future_dim"],
    )

    opt_cond = torch.optim.Adam(conditioned_model.parameters(), lr=mc["lr"])
    opt_uncond = torch.optim.Adam(unconditional_model.parameters(), lr=mc["lr"])

    history = {"conditioned": {"train_loss": [], "val_loss": []},
               "unconditional": {"train_loss": [], "val_loss": []}}

    t0 = time.time()
    print(f"=== Training conditioned_model (New Way / INTERACT predictor) and "
          f"unconditional_model (Old Way A baseline) for {mc['epochs']} epochs ===")
    for epoch in range(1, mc["epochs"] + 1):
        tr_loss_c = run_epoch(conditioned_model, train_loader, opt_cond, force_zero_cond=False, train=True)
        va_loss_c = run_epoch(conditioned_model, val_loader, opt_cond, force_zero_cond=False, train=False)
        tr_loss_u = run_epoch(unconditional_model, train_loader, opt_uncond, force_zero_cond=True, train=True)
        va_loss_u = run_epoch(unconditional_model, val_loader, opt_uncond, force_zero_cond=True, train=False)

        history["conditioned"]["train_loss"].append(tr_loss_c)
        history["conditioned"]["val_loss"].append(va_loss_c)
        history["unconditional"]["train_loss"].append(tr_loss_u)
        history["unconditional"]["val_loss"].append(va_loss_u)

        print(f"epoch {epoch:2d}/{mc['epochs']} | conditioned train={tr_loss_c:.5f} val={va_loss_c:.5f}"
              f" | unconditional train={tr_loss_u:.5f} val={va_loss_u:.5f}")

    train_wall_time = time.time() - t0
    print(f"\nTraining wall time: {train_wall_time:.2f}s")

    # --- bucketed held-out evaluation (the real empirical claim this repo can test) ---
    print("\n=== Held-out bucketed evaluation (anchor-sampled validation set) ===")
    cond_results = eval_bucketed(conditioned_model, val_anchor_ds, force_zero_cond=False)
    uncond_results = eval_bucketed(unconditional_model, val_anchor_ds, force_zero_cond=True)
    for bucket in ("interactive", "neutral", "all"):
        if bucket not in cond_results:
            continue
        c, u = cond_results[bucket], uncond_results[bucket]
        improvement_pct = 100.0 * (u["ade"] - c["ade"]) / max(u["ade"], 1e-9)
        print(f"[{bucket:11s}] n={c['n']:4d} | conditioned ADE={c['ade']:.4f} FDE={c['fde']:.4f} "
              f"| unconditional ADE={u['ade']:.4f} FDE={u['fde']:.4f} "
              f"| ADE improvement={improvement_pct:+.1f}%")

    # --- save checkpoints + metrics ---
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(conditioned_model.state_dict(), os.path.join(ckpt_dir, "conditioned_model.pt"))
    torch.save(unconditional_model.state_dict(), os.path.join(ckpt_dir, "unconditional_model.pt"))

    metrics = {
        "config_used": args.config,
        "train_wall_time_s": train_wall_time,
        "loss_history": history,
        "held_out_bucketed": {"conditioned": cond_results, "unconditional": uncond_results},
        "final_train_loss": {"conditioned": history["conditioned"]["train_loss"][-1],
                              "unconditional": history["unconditional"]["train_loss"][-1]},
        "final_val_loss": {"conditioned": history["conditioned"]["val_loss"][-1],
                            "unconditional": history["unconditional"]["val_loss"][-1]},
        "anchor_names": anchor_names,
    }
    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved checkpoints + metrics.json to {ckpt_dir}/")


if __name__ == "__main__":
    main()
