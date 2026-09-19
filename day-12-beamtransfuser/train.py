"""
Train BeamTransFuser on the synthetic V2X drive-by beam-prediction task.

Usage:
    python train.py --config config.yaml --steps 300

This is a reconstruction training run on procedurally generated synthetic
data (see src/utils/synthetic_data.py) -- it reports THIS repo's own
synthetic-task numbers, not the original paper's results (see README's
"Sourcing note").
"""
import argparse
import time

import torch
import torch.nn.functional as F
import yaml

from src.models.beam_transfuser import build_model_from_config
from src.utils.synthetic_data import generate_batch


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    topk = logits.topk(k, dim=1).indices          # (B, k)
    hit = (topk == labels.view(-1, 1)).any(dim=1)
    return hit.float().mean().item()


def evaluate(model, cfg, device):
    model.eval()
    rng = __import__("numpy").random.default_rng(cfg["data"]["seed"] + 999)
    val_size = cfg["data"]["val_size"]
    batch_size = 64
    all_logits, all_labels = [], []
    with torch.no_grad():
        seen = 0
        while seen < val_size:
            bs = min(batch_size, val_size - seen)
            batch = generate_batch(
                bs,
                num_beams=cfg["model"]["num_beams"],
                modality_dropout_prob=cfg["data"]["modality_dropout_prob"],
                rng=rng,
            ).to(device)
            logits = model(batch.camera, batch.lidar, batch.radar, batch.gps, batch.presence)
            all_logits.append(logits)
            all_labels.append(batch.beam_label)
            seen += bs
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    loss = F.cross_entropy(logits, labels).item()
    top1 = topk_accuracy(logits, labels, 1)
    top3 = topk_accuracy(logits, labels, 3)
    model.train()
    return loss, top1, top3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps

    torch.manual_seed(cfg["train"]["seed"])
    device = torch.device("cpu")

    model = build_model_from_config(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BeamTransFuser | d_model={cfg['model']['d_model']} n_heads={cfg['model']['n_heads']} "
          f"n_fusion_blocks={cfg['model']['n_fusion_blocks']} num_beams={cfg['model']['num_beams']} "
          f"| {n_params:,} params")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
    )

    import numpy as np
    train_rng = np.random.default_rng(cfg["train"]["seed"])

    steps = cfg["train"]["steps"]
    batch_size = cfg["train"]["batch_size"]
    log_every = cfg["train"]["log_every"]

    losses = []
    t0 = time.time()
    for step in range(1, steps + 1):
        batch = generate_batch(
            batch_size,
            num_beams=cfg["model"]["num_beams"],
            modality_dropout_prob=cfg["data"]["modality_dropout_prob"],
            rng=train_rng,
        ).to(device)

        logits = model(batch.camera, batch.lidar, batch.radar, batch.gps, batch.presence)
        loss = F.cross_entropy(logits, batch.beam_label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        if step == 1 or step % log_every == 0 or step == steps:
            elapsed = time.time() - t0
            s_per_step = elapsed / step
            print(f"step {step:4d}/{steps} | loss {loss.item():.4f} | "
                  f"avg(last {min(log_every, step)}) {sum(losses[-log_every:]) / min(log_every, len(losses)):.4f} | "
                  f"{s_per_step:.3f}s/step")

    val_loss, top1, top3 = evaluate(model, cfg, device)
    total_time = time.time() - t0

    print("\n=== Training complete ===")
    print(f"loss: {losses[0]:.4f} -> {sum(losses[-10:]) / len(losses[-10:]):.4f} (final-10 avg), "
          f"last-step {losses[-1]:.4f}")
    print(f"validation (n={cfg['data']['val_size']}): loss={val_loss:.4f} "
          f"top1_acc={top1:.4f} top3_acc={top3:.4f}")
    print(f"total train time: {total_time:.1f}s ({total_time / steps:.3f}s/step avg)")

    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, "beam_transfuser_checkpoint.pt")
    print("saved checkpoint -> beam_transfuser_checkpoint.pt")


if __name__ == "__main__":
    main()
