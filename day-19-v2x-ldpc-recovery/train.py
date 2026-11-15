"""
train.py

Trains ProbabilisticMotionPredictor on synthetic CTRV-style vehicle
trajectories (src/dataset.py) with a real Gaussian negative-log-likelihood
loss, on CPU. Saves a checkpoint to checkpoints/motion_predictor.pt.

Run:
    python train.py
"""
from __future__ import annotations

import time
import yaml
import torch
from torch.utils.data import DataLoader, random_split

from src.dataset import TrajectoryDataset
from models.motion_predictor import ProbabilisticMotionPredictor, gaussian_nll_loss


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["training"]["seed"])

    pred_cfg = cfg["predictor"]
    train_cfg = cfg["training"]

    print("=" * 70)
    print("Day 19 -- ProbabilisticMotionPredictor training (synthetic CTRV data)")
    print("=" * 70)

    dataset = TrajectoryDataset(
        num_trajectories=train_cfg["num_trajectories"],
        history_len=pred_cfg["history_len"],
        future_len=pred_cfg["future_len"],
        seed=train_cfg["seed"],
    )
    val_size = int(len(dataset) * train_cfg["val_fraction"])
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(train_cfg["seed"]),
    )
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False)

    print(f"Dataset: {len(dataset)} trajectories "
          f"({train_size} train / {val_size} val), "
          f"history_len={pred_cfg['history_len']}, future_len={pred_cfg['future_len']}")

    model = ProbabilisticMotionPredictor(
        num_fields=pred_cfg["num_fields"],
        hidden_dim=pred_cfg["hidden_dim"],
        future_len=pred_cfg["future_len"],
        dropout=pred_cfg["dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ProbabilisticMotionPredictor (GRU encoder-decoder), {n_params:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])

    loss_history = []
    t_start = time.time()
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        train_losses = []
        for hist, fut in train_loader:
            optimizer.zero_grad()
            out = model(hist)
            target_norm = model.normalize(fut)
            loss = gaussian_nll_loss(out["mu_norm"], out["logvar_norm"], target_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for hist, fut in val_loader:
                out = model(hist)
                target_norm = model.normalize(fut)
                loss = gaussian_nll_loss(out["mu_norm"], out["logvar_norm"], target_norm)
                val_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        loss_history.append({"epoch": epoch, "train_nll": train_loss, "val_nll": val_loss})
        print(f"Epoch {epoch:3d}/{train_cfg['epochs']}  "
              f"train_NLL={train_loss:.4f}  val_NLL={val_loss:.4f}")

    wall_time = time.time() - t_start

    import os
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/motion_predictor.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "loss_history": loss_history,
        "n_params": n_params,
        "wall_time_sec": wall_time,
    }, ckpt_path)

    print("-" * 70)
    print(f"Training complete in {wall_time:.1f}s on CPU.")
    print(f"Loss curve: epoch 1 train_NLL={loss_history[0]['train_nll']:.4f} "
          f"-> epoch {train_cfg['epochs']} train_NLL={loss_history[-1]['train_nll']:.4f}")
    print(f"Final val_NLL={loss_history[-1]['val_nll']:.4f}")
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
