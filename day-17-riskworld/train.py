"""Trains RiskWorld's core forecasting model (BEV+actor encoding -> risk
field -> flow-guided occupancy evolution) on the synthetic dataset.

Trajectory-level planning (collision scoring + selective replacement) is NOT
trained here -- it has no learnable parameters and runs at planning time on
top of the trained model's outputs (see simulate.py).

Usage:
    python train.py                       # full run per config.yaml
    python train.py --epochs 2             # override epoch count
    python train.py --steps 20              # cap total optimizer steps (quick smoke test)
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import RiskWorldSyntheticDataset
from src.riskworld_model import RiskWorld
from src.utils import load_config, set_seed


def compute_losses(model_out: dict, batch: dict, occ_weight: float, risk_weight: float) -> dict:
    """MSE losses (not BCE) because `occupancy` can be exactly 0 or 1 after
    the flow-evolution module's clamp, which would make BCE's log(.) blow up
    without an extra epsilon fudge -- MSE stays well-behaved and still gives a
    clean, monotonically-informative training signal for this sanity-scale model."""
    occ_loss = F.mse_loss(model_out["forecast_occupancy"], batch["future_occupancy_gt"])
    risk_loss = F.mse_loss(model_out["risk_field"], batch["risk_target"])
    total = occ_weight * occ_loss + risk_weight * risk_loss
    return {"total": total, "occ_loss": occ_loss, "risk_loss": risk_loss}


def run_epoch(model, loader, optimizer, cfg, device, max_steps=None, step_counter=None) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device).float() if v.dtype != torch.bool else v.to(device) for k, v in batch.items()}

        model_out = model(
            bev_grid=batch["bev_grid"],
            agent_history=batch["agent_history"],
            agent_last_pos=batch["agent_last_pos"],
            agent_mask=batch["agent_mask"],
            prev_occupancy=batch["prev_occupancy"],
        )
        losses = compute_losses(
            model_out, batch, cfg["train"]["occ_loss_weight"], cfg["train"]["risk_loss_weight"]
        )

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
        optimizer.step()

        total_loss += losses["total"].item()
        n_batches += 1
        if step_counter is not None:
            step_counter[0] += 1
            if max_steps is not None and step_counter[0] >= max_steps:
                break

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help="cap total optimizer steps across all epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device("cpu")

    train_ds = RiskWorldSyntheticDataset(cfg, cfg["data"]["num_train_scenes"], base_seed=cfg["seed"])
    val_ds = RiskWorldSyntheticDataset(cfg, cfg["data"]["num_val_scenes"], base_seed=cfg["seed"] + 1)
    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False)

    model = RiskWorld(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RiskWorld reconstruction: {n_params:,} trainable parameters "
          f"(NOTE: the paper reports 90.81M params for the real model -- this "
          f"CPU-scale reconstruction is intentionally far smaller; see README).")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"]
    )

    epochs = args.epochs if args.epochs is not None else cfg["train"]["epochs"]
    step_counter = [0]
    loss_curve = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        avg_loss = run_epoch(model, train_loader, optimizer, cfg, device, max_steps=args.steps, step_counter=step_counter)
        loss_curve.append(avg_loss)
        print(f"epoch {epoch:02d}/{epochs}  train_loss={avg_loss:.6f}  steps={step_counter[0]}")
        if args.steps is not None and step_counter[0] >= args.steps:
            print(f"reached --steps cap ({args.steps}), stopping early")
            break

    elapsed = time.time() - t0
    print(f"training finished in {elapsed:.1f}s -- loss[0]={loss_curve[0]:.6f} -> loss[-1]={loss_curve[-1]:.6f}")

    # Quick validation: compare RiskWorld's forecast occupancy MSE against the
    # naive persistence baseline's MSE, split by whether the scene contains
    # the late-emerging cross-traffic hazard. These are THIS REPO'S OWN
    # synthetic-data numbers, not anything from the paper.
    model.eval()
    fw_err_hazard, pers_err_hazard, n_hazard = 0.0, 0.0, 0
    fw_err_normal, pers_err_normal, n_normal = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch_f = {k: v.to(device).float() if v.dtype != torch.bool else v.to(device) for k, v in batch.items()}
            out = model(
                bev_grid=batch_f["bev_grid"],
                agent_history=batch_f["agent_history"],
                agent_last_pos=batch_f["agent_last_pos"],
                agent_mask=batch_f["agent_mask"],
                prev_occupancy=batch_f["prev_occupancy"],
            )
            fw_err = ((out["forecast_occupancy"] - batch_f["future_occupancy_gt"]) ** 2).mean(dim=(1, 2, 3, 4))
            pers_err = ((out["persistence_occupancy"] - batch_f["future_occupancy_gt"]) ** 2).mean(dim=(1, 2, 3, 4))
            hazard_mask = batch["has_hazard"]
            for i in range(hazard_mask.shape[0]):
                if bool(hazard_mask[i]):
                    fw_err_hazard += fw_err[i].item()
                    pers_err_hazard += pers_err[i].item()
                    n_hazard += 1
                else:
                    fw_err_normal += fw_err[i].item()
                    pers_err_normal += pers_err[i].item()
                    n_normal += 1

    print("\n[this repo's own synthetic-data validation numbers -- NOT the paper's]")
    if n_hazard:
        print(f"  hazard scenes (n={n_hazard}): RiskWorld forecast MSE={fw_err_hazard/n_hazard:.5f}  "
              f"persistence-baseline MSE={pers_err_hazard/n_hazard:.5f}")
    if n_normal:
        print(f"  non-hazard scenes (n={n_normal}): RiskWorld forecast MSE={fw_err_normal/n_normal:.5f}  "
              f"persistence-baseline MSE={pers_err_normal/n_normal:.5f}")

    os.makedirs(os.path.dirname(cfg["paths"]["checkpoint"]), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "cfg": cfg}, cfg["paths"]["checkpoint"])
    print(f"\nsaved checkpoint to {cfg['paths']['checkpoint']}")


if __name__ == "__main__":
    main()
