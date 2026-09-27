"""Train S2Planner on the synthetic multi-camera driving dataset.

Usage: python3 train.py --config config.yaml
"""
import argparse
import json
import time
import yaml
import torch
from torch.utils.data import DataLoader

from src.dataset import SyntheticDrivingSceneDataset, collate_scenes
from src.model import S2Planner
from src.losses import coarse_to_fine_loss, ade_fde


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])

    train_ds = SyntheticDrivingSceneDataset(n_samples=cfg["data"]["train_samples"], seed=1, image_size=cfg["data"]["image_size"])
    val_ds = SyntheticDrivingSceneDataset(n_samples=cfg["data"]["val_samples"], seed=2, image_size=cfg["data"]["image_size"])
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_scenes)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_scenes)

    model = S2Planner(d_feat=cfg["model"]["d_feat"], n_scales=cfg["model"]["n_scales"])
    n_params = model.count_parameters()
    print(f"S2Planner reconstruction: {n_params:,} trainable parameters")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    history = {"train_loss": [], "val_loss": [], "val_ade": [], "val_fde": [], "init_val_ade": []}
    t0 = time.time()
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            opt.zero_grad()
            out = model(batch["cam_features"], batch["ego_history"], batch["command"])
            loss, _ = coarse_to_fine_loss(out["stage_trajectories"], batch["future_gt"])
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / n_batches

        model.eval()
        val_loss_total, val_ade_total, val_fde_total, init_ade_total, n_val_batches = 0.0, 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch["cam_features"], batch["ego_history"], batch["command"])
                loss, _ = coarse_to_fine_loss(out["stage_trajectories"], batch["future_gt"])
                ade, fde = ade_fde(out["trajectory"], batch["future_gt"])
                init_ade, _ = ade_fde(out["stage_trajectories"][0], batch["future_gt"])
                val_loss_total += loss.item()
                val_ade_total += ade
                val_fde_total += fde
                init_ade_total += init_ade
                n_val_batches += 1
        val_loss = val_loss_total / n_val_batches
        val_ade = val_ade_total / n_val_batches
        val_fde = val_fde_total / n_val_batches
        init_ade = init_ade_total / n_val_batches

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ade"].append(val_ade)
        history["val_fde"].append(val_fde)
        history["init_val_ade"].append(init_ade)

        if epoch % cfg["train"]["log_every"] == 0 or epoch == cfg["train"]["epochs"] - 1:
            print(
                f"epoch {epoch+1:02d}/{cfg['train']['epochs']}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"val_ADE={val_ade:.3f}m  val_FDE={val_fde:.3f}m  "
                f"(coarse-init ADE={init_ade:.3f}m)"
            )

    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.1f}s")

    reduction = 100.0 * (1 - history["val_ade"][-1] / history["init_val_ade"][-1])
    print(
        f"Final-stage val ADE {history['val_ade'][-1]:.3f}m vs. coarse-init val ADE "
        f"{history['init_val_ade'][-1]:.3f}m -> {reduction:.1f}% reduction from cross-attention refinement "
        f"(synthetic data, this project's own reconstruction -- NOT the paper's 88.03 PDMS NAVSIM number, "
        f"which is a different metric on a different benchmark)."
    )

    import os
    os.makedirs("outputs", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "n_params": n_params,
            "final_val_ade": history["val_ade"][-1],
            "final_val_fde": history["val_fde"][-1],
            "final_init_ade": history["init_val_ade"][-1],
        },
        cfg["output"]["checkpoint"],
    )
    with open(cfg["output"]["history"], "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved checkpoint to {cfg['output']['checkpoint']}")


if __name__ == "__main__":
    main()
