"""
Trains ChronoFuseDetector on the synthetic latency-compensated event
detection task, and reports both the "ChronoFuse" (compensated) and
"naive / old-way" (uncompensated, same weights, chronofuse disabled)
center-distance error on a held-out validation split.

Usage:
    python train.py --config config.yaml
"""

import argparse
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.synthetic_events import SyntheticDrivingEventDataset, collate_samples
from src.models.model import ChronoFuseDetector
from src.utils.losses import DetectionLoss, build_targets, STRIDE
from src.utils.metrics import center_distance_error, stale_baseline_error


def run_epoch(model, loader, loss_fn, optimizer, num_classes, out_hw, device, train: bool):
    """Jointly trains BOTH forward paths through the same shared
    backbone/FPN/head each step:

      - enable_chronofuse=True  -> supervised against the FUTURE object
        positions (t_obs_last + latency_steps). This is ChronoFuse's job:
        predict for when the output becomes available.
      - enable_chronofuse=False -> supervised against the STALE
        (last-observed, t_obs_last) positions. This is what a standard
        single-frame detector with no temporal compensation is actually
        trained to do: report what it currently sees.

    Training both paths jointly (rather than only ever training the
    enabled path) matters for a fair 'old way vs. new way' comparison: if
    the shared head only ever saw ChronoFuse-fused features during
    training, disabling ChronoFuse at eval time would feed it an
    out-of-distribution input and the resulting 'old way' numbers would
    reflect that mismatch rather than the actual old-way detector's
    accuracy. (Caught by inspecting a first training run where the
    disabled path's error was worse than random -- see the daily-review
    doc's implementation notes.)
    """
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            event_seq = batch["event_seq"].to(device)
            latency_steps = batch["latency_steps"].to(device)

            future_hm, future_size, future_off, future_mask = build_targets(
                batch["future_centers"], batch["sizes"], batch["classes"], *out_hw, num_classes
            )
            stale_hm, stale_size, stale_off, stale_mask = build_targets(
                batch["stale_centers"], batch["sizes"], batch["classes"], *out_hw, num_classes
            )
            future_hm, future_size, future_off, future_mask = (
                future_hm.to(device), future_size.to(device), future_off.to(device), future_mask.to(device)
            )
            stale_hm, stale_size, stale_off, stale_mask = (
                stale_hm.to(device), stale_size.to(device), stale_off.to(device), stale_mask.to(device)
            )

            pred_hm_on, pred_size_on, pred_off_on = model(event_seq, latency_steps, enable_chronofuse=True)
            loss_on, _ = loss_fn(pred_hm_on, pred_size_on, pred_off_on,
                                  future_hm, future_size, future_off, future_mask)

            pred_hm_off, pred_size_off, pred_off_off = model(event_seq, latency_steps, enable_chronofuse=False)
            loss_off, _ = loss_fn(pred_hm_off, pred_size_off, pred_off_off,
                                   stale_hm, stale_size, stale_off, stale_mask)

            loss = loss_on + loss_off

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_compensation_gap(model, loader, device):
    """Compares, using the SAME trained weights: (a) ChronoFuse-compensated
    predictions vs (b) the same model with ChronoFuse disabled (the 'old
    way' ablation baseline) vs (c) the literal stale-position baseline."""
    model.eval()
    chronofuse_errs, disabled_errs, stale_errs = [], [], []

    for batch in loader:
        event_seq = batch["event_seq"].to(device)
        latency_steps = batch["latency_steps"].to(device)

        hm_on, _, off_on = model(event_seq, latency_steps, enable_chronofuse=True)
        hm_off, _, off_off = model(event_seq, latency_steps, enable_chronofuse=False)

        for i in range(event_seq.shape[0]):
            gt_centers = batch["future_centers"][i]
            gt_classes = batch["classes"][i]
            stale_centers = batch["stale_centers"][i]

            chronofuse_errs.append(center_distance_error(hm_on[i], off_on[i], gt_centers, gt_classes))
            disabled_errs.append(center_distance_error(hm_off[i], off_off[i], gt_centers, gt_classes))
            stale_errs.append(stale_baseline_error(stale_centers, gt_centers))

    import statistics
    return (
        statistics.mean(chronofuse_errs),
        statistics.mean(disabled_errs),
        statistics.mean(stale_errs),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["data"]["seed"])
    device = torch.device("cpu")

    train_ds = SyntheticDrivingEventDataset(
        num_samples=cfg["data"]["train_samples"],
        height=cfg["data"]["height"],
        width=cfg["data"]["width"],
        t_obs=cfg["data"]["t_obs"],
        max_agents=cfg["data"]["max_agents"],
        max_latency_steps=cfg["data"]["max_latency_steps"],
        seed=cfg["data"]["seed"],
    )
    val_ds = SyntheticDrivingEventDataset(
        num_samples=cfg["data"]["val_samples"],
        height=cfg["data"]["height"],
        width=cfg["data"]["width"],
        t_obs=cfg["data"]["t_obs"],
        max_agents=cfg["data"]["max_agents"],
        max_latency_steps=cfg["data"]["max_latency_steps"],
        seed=cfg["data"]["seed"] + 1,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                               collate_fn=collate_samples)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                             collate_fn=collate_samples)

    model = ChronoFuseDetector(
        in_channels=cfg["model"]["in_channels"],
        backbone_channels=tuple(cfg["model"]["backbone_channels"]),
        fpn_channels=cfg["model"]["fpn_channels"],
        num_classes=cfg["model"]["num_classes"],
        cache_len=cfg["model"]["cache_len"],
        latency_dim=cfg["model"]["latency_dim"],
        max_latency=cfg["model"]["max_latency"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    print(f"  of which ChronoFuse module: {model.chronofuse_param_count:,} "
          f"({100 * model.chronofuse_param_count / total_params:.1f}% of total)")

    loss_fn = DetectionLoss(cfg["train"]["size_weight"], cfg["train"]["offset_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])

    out_h = cfg["data"]["height"] // STRIDE
    out_w = cfg["data"]["width"] // STRIDE

    t0 = time.time()
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer,
                                cfg["model"]["num_classes"], (out_h, out_w), device, train=True)
        val_loss = run_epoch(model, val_loader, loss_fn, optimizer,
                              cfg["model"]["num_classes"], (out_h, out_w), device, train=False)
        if epoch % cfg["train"]["log_every"] == 0 or epoch == 1 or epoch == cfg["train"]["epochs"]:
            print(f"epoch {epoch:3d}/{cfg['train']['epochs']}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    elapsed = time.time() - t0
    print(f"Training complete in {elapsed:.1f}s")

    chronofuse_err, disabled_err, stale_err = evaluate_compensation_gap(model, val_loader, device)
    recovered_pct = 100.0 * (stale_err - chronofuse_err) / max(stale_err, 1e-6)
    disabled_recovered_pct = 100.0 * (stale_err - disabled_err) / max(stale_err, 1e-6)

    print("\n--- Held-out validation: center-distance error (px), synthetic data ---")
    print(f"  stale / no-compensation baseline (copy last observed position): {stale_err:.2f} px")
    print(f"  same weights, ChronoFuse DISABLED (raw-frame-only ablation):     {disabled_err:.2f} px "
          f"({disabled_recovered_pct:+.1f}% vs. stale)")
    print(f"  ChronoFuse ENABLED (this repo's reconstruction):                {chronofuse_err:.2f} px "
          f"({recovered_pct:+.1f}% of stale-baseline error recovered)")
    print("  NOTE: this is this repo's own simplified center-distance metric on synthetic data — "
          "NOT the paper's sAP metric or the paper's reported 71%/9.3x numbers. See SOURCING.md.")

    ckpt_path = Path(cfg["train"]["checkpoint_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
