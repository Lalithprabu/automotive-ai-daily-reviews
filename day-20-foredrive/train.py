"""
train.py -- end-to-end training of the ForeDrive reconstruction on the
synthetic dataset. Runs on CPU in a couple of minutes with the default
config.yaml (small model, few epochs) -- this is a DEMONSTRATION run to show
the whole pipeline works and produce a checkpoint for simulate.py, not a
from-scratch training run intended to fully converge (see
tests/test_sanity_signal.py for a longer, dedicated sanity-check training
loop, and SOURCING.md / README.md for what is and is not validated here).

Usage:
    python3 train.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import time

import torch
import yaml
from torch.utils.data import DataLoader

from models.foredrive import ForeDriveModel
from src.data.synthetic_dataset import SyntheticDrivingDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["data"]["seed"])
    device = torch.device(cfg["train"]["device"])

    print("=" * 70)
    print("ForeDrive reconstruction -- training on synthetic data")
    print("=" * 70)

    # --- Data ---------------------------------------------------------------
    context_t = cfg["data"]["context_len"] - 1
    train_ds = SyntheticDrivingDataset(
        num_scenes=cfg["data"]["train_scenes"], scene_len=cfg["data"]["scene_len"],
        dt=cfg["data"]["dt"], raster_size=cfg["data"]["raster_size"],
        horizons=cfg["data"]["horizons"], num_waypoints=cfg["data"]["num_waypoints"],
        context_t=context_t, num_agents_min=cfg["data"]["num_agents_min"],
        num_agents_max=cfg["data"]["num_agents_max"], seed=cfg["data"]["seed"],
    )
    val_ds = SyntheticDrivingDataset(
        num_scenes=cfg["data"]["val_scenes"], scene_len=cfg["data"]["scene_len"],
        dt=cfg["data"]["dt"], raster_size=cfg["data"]["raster_size"],
        horizons=cfg["data"]["horizons"], num_waypoints=cfg["data"]["num_waypoints"],
        context_t=context_t, num_agents_min=cfg["data"]["num_agents_min"],
        num_agents_max=cfg["data"]["num_agents_max"], seed=cfg["data"]["seed"] + 1,
    )
    train_dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    print(f"train scenes: {len(train_ds)}  val scenes: {len(val_ds)}")

    # --- Model ----------------------------------------------------------------
    model = ForeDriveModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model parameters: {n_params:,} (trainable: {n_trainable:,})")

    # Trajectory normalization stats (see models/foredrive.py for why this
    # matters for diffusion training stability), computed from the training
    # set and saved into the checkpoint via the model's buffers.
    all_train_wp = torch.stack([train_ds[i]["expert_waypoints"] for i in range(len(train_ds))])
    wp_mean = all_train_wp.mean(dim=(0, 1))
    wp_std = all_train_wp.std(dim=(0, 1))
    model.set_trajectory_normalization(wp_mean, wp_std)
    print(f"waypoint normalization: mean={wp_mean.tolist()} std={wp_std.tolist()}")

    opt = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"],
    )

    forecast_w = cfg["train"]["forecast_loss_weight"]
    planning_w = cfg["train"]["planning_loss_weight"]

    # --- Training loop --------------------------------------------------------
    history = {"forecast_loss": [], "planning_loss": []}
    t_start = time.time()
    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        epoch_forecast, epoch_planning, n_batches = 0.0, 0.0, 0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model.forward_train(
                batch["current_frame"], batch["future_frames"], batch["ego_context"], batch["expert_waypoints"],
            )
            loss = forecast_w * out["forecast_loss"] + planning_w * out["planning_loss"]

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
            opt.step()

            epoch_forecast += out["forecast_loss"].item()
            epoch_planning += out["planning_loss"].item()
            n_batches += 1
            step += 1

            if step % cfg["train"]["log_every"] == 0:
                print(f"  epoch {epoch} step {step}: forecast_loss={out['forecast_loss'].item():.4f} "
                      f"planning_loss={out['planning_loss'].item():.4f}")

        epoch_forecast /= max(n_batches, 1)
        epoch_planning /= max(n_batches, 1)
        history["forecast_loss"].append(epoch_forecast)
        history["planning_loss"].append(epoch_planning)
        print(f"[epoch {epoch}] avg forecast_loss={epoch_forecast:.4f} avg planning_loss={epoch_planning:.4f} "
              f"elapsed={time.time() - t_start:.1f}s")

    # --- Validation (sampled trajectory vs. expert / CV baseline) -------------
    model.eval()
    pred_errs, cv_errs = [], []
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            traj, _ = model.sample_trajectory(batch["current_frame"], batch["ego_context"])
            pred_errs.append((traj - batch["expert_waypoints"]).norm(dim=-1).mean().item())
            cv_errs.append((batch["cv_baseline_waypoints"] - batch["expert_waypoints"]).norm(dim=-1).mean().item())
    val_pred_err = sum(pred_errs) / len(pred_errs)
    val_cv_err = sum(cv_errs) / len(cv_errs)

    print("=" * 70)
    print(f"Final avg forecast_loss: {history['forecast_loss'][0]:.4f} -> {history['forecast_loss'][-1]:.4f}")
    print(f"Final avg planning_loss: {history['planning_loss'][0]:.4f} -> {history['planning_loss'][-1]:.4f}")
    print(f"Validation: model trajectory L2 err = {val_pred_err:.3f} m | CV baseline L2 err = {val_cv_err:.3f} m")
    print(f"NOTE: this quick demo run ({cfg['train']['epochs']} epochs) is not expected to fully "
          f"converge -- see tests/test_sanity_signal.py for a longer, dedicated sanity-check "
          f"training run and README.md / SOURCING.md for full context.")
    print(f"Total training time: {time.time() - t_start:.1f}s")

    torch.save({"model_state_dict": model.state_dict(), "config": cfg, "history": history},
               cfg["train"]["checkpoint_path"])
    print(f"Saved checkpoint to {cfg['train']['checkpoint_path']}")


if __name__ == "__main__":
    main()
