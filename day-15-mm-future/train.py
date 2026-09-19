"""Train the MM-Future reconstruction on synthetic bimodal driving scenes.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --epochs 2   # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import MMFutureDataset
from src.model import MMFuture


def fit_action_prior(model: MMFuture, dataset: MMFutureDataset):
    all_deltas = np.stack([s["action_deltas"] for s in dataset.scenes])
    model.action_prior.fit(all_deltas)


def evaluate(model: MMFuture, loader: DataLoader, device) -> dict:
    model.eval()
    ades, top1s = [], []
    with torch.no_grad():
        for batch in loader:
            out = model.forward_train(batch, device)
            ades.append(out["ade"].item())
            top1s.append(out["top1_acc"].item())
    model.train()
    return {"ade": float(np.mean(ades)), "top1_acc": float(np.mean(top1s))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    torch.manual_seed(cfg["data"]["seed"])
    device = torch.device(cfg["train"]["device"])

    d = cfg["data"]
    train_ds = MMFutureDataset(
        num_scenes=d["num_train_scenes"], grid_size=d["grid_size"], num_agents=d["num_agents"],
        history_frames=d["history_frames"], future_frames=d["future_frames"],
        camera_crop=d["camera_crop"], seed=d["seed"],
    )
    val_ds = MMFutureDataset(
        num_scenes=d["num_val_scenes"], grid_size=d["grid_size"], num_agents=d["num_agents"],
        history_frames=d["history_frames"], future_frames=d["future_frames"],
        camera_crop=d["camera_crop"], seed=d["seed"] + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = MMFuture(cfg).to(device)
    fit_action_prior(model, train_ds)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    print(f"Training MM-Future reconstruction | train scenes={len(train_ds)} val scenes={len(val_ds)} "
          f"device={device} epochs={cfg['train']['epochs']}")

    step = 0
    history = []
    for epoch in range(cfg["train"]["epochs"]):
        for batch in train_loader:
            opt.zero_grad()
            out = model.forward_train(batch, device)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % cfg["train"]["log_every"] == 0:
                print(f"epoch {epoch} step {step} | loss {out['loss'].item():.4f} "
                      f"action {out['action_loss'].item():.4f} scene {out['scene_loss'].item():.4f} "
                      f"score {out['score_loss'].item():.4f} bev {out['bev_loss'].item():.4f} "
                      f"ade {out['ade'].item():.3f} top1_acc {out['top1_acc'].item():.3f}")
            history.append({"step": step, "loss": out["loss"].item()})
            step += 1

        val_metrics = evaluate(model, val_loader, device)
        print(f"== epoch {epoch} val | ADE {val_metrics['ade']:.3f}  top1_acc {val_metrics['top1_acc']:.3f} ==")

    ckpt_path = cfg["train"]["checkpoint_path"]
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "action_prior_means": model.action_prior.means,
        "action_prior_stds": model.action_prior.stds,
        "action_prior_k": model.action_prior.num_clusters,
        "cfg": cfg,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    final_val = evaluate(model, val_loader, device)
    with open("train_log.json", "w") as f:
        json.dump({"history": history, "final_val": final_val}, f, indent=2)
    print(f"Final validation: ADE={final_val['ade']:.3f}  top1_acc={final_val['top1_acc']:.3f}")


if __name__ == "__main__":
    main()
