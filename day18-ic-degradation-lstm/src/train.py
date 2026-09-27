"""
src/train.py
==========================================================================
Trains ChargeIC-LSTM on the synthetic 53-cell dataset, evaluates on a
held-out set of UNSEEN cells (val + test), saves a checkpoint, and runs a
sanity check against a trivial "predict the training-mean IC curve"
baseline to confirm the model is actually learning signal, not noise.

Run:  python src/train.py   (from the repo root)
==========================================================================
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import load_config, set_seed, build_datasets, build_model, save_checkpoint  # noqa: E402
from models.chargeic_lstm import chargeic_loss  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, soh_weight, device):
    model.eval()
    total_loss, total_ic_mse, total_soh_mae, n = 0.0, 0.0, 0.0, 0
    for x, y_ic, y_soh in loader:
        x, y_ic, y_soh = x.to(device), y_ic.to(device), y_soh.to(device)
        ic_pred, soh_pred = model(x)
        loss, ic_mse, _ = chargeic_loss(ic_pred, y_ic, soh_pred, y_soh, soh_weight)
        bsz = x.shape[0]
        total_loss += loss.item() * bsz
        total_ic_mse += ic_mse.item() * bsz
        total_soh_mae += torch.mean(torch.abs(soh_pred - y_soh)).item() * bsz
        n += bsz
    return {
        "loss": total_loss / n,
        "ic_mse": total_ic_mse / n,
        "soh_mae": total_soh_mae / n,
    }


@torch.no_grad()
def mean_curve_baseline_mse(train_ds, eval_ds):
    """
    Sanity-check baseline: predict the TRAIN-SET mean IC curve for every
    held-out sample, regardless of input. If the trained model cannot beat
    this trivial baseline's MSE on held-out cells, it has learned nothing
    useful from the charging signal (or the synthetic input-output link is
    too weak) and results should NOT be trusted.
    """
    mean_curve = train_ds.ic.mean(axis=0)  # (M,)
    diffs = eval_ds.ic - mean_curve[None, :]
    return float(np.mean(diffs ** 2))


def main():
    cfg = load_config()
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, test_ds, train_cells, val_cells, test_cells = build_datasets(cfg)
    print(
        f"Cells -> train:{len(train_cells)} val:{len(val_cells)} test:{len(test_cells)} "
        f"| Cycles -> train:{len(train_ds)} val:{len(val_ds)} test:{len(test_ds)}"
    )

    tcfg = cfg["train"]
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=tcfg["batch_size"], shuffle=False)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"]
    )

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_ic_mse": [], "val_soh_mae": []}
    best_val_loss = float("inf")
    ckpt_path = os.path.join(_ROOT, tcfg["checkpoint_path"])

    t0 = time.time()
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        for x, y_ic, y_soh in train_loader:
            x, y_ic, y_soh = x.to(device), y_ic.to(device), y_soh.to(device)

            optimizer.zero_grad()
            ic_pred, soh_pred = model(x)
            loss, ic_mse, soh_mse = chargeic_loss(ic_pred, y_ic, soh_pred, y_soh, tcfg["soh_loss_weight"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip_norm"])
            optimizer.step()

            running_loss += loss.item() * x.shape[0]
            n_seen += x.shape[0]

        train_loss = running_loss / n_seen

        if epoch % tcfg["val_check_every"] == 0 or epoch == tcfg["epochs"]:
            val_metrics = evaluate(model, val_loader, tcfg["soh_loss_weight"], device)
            print(
                f"Epoch {epoch:3d}/{tcfg['epochs']} | train_loss={train_loss:.5f} "
                f"| val_loss={val_metrics['loss']:.5f} val_ic_mse={val_metrics['ic_mse']:.5f} "
                f"val_soh_mae={val_metrics['soh_mae']:.5f}"
            )
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_ic_mse"].append(val_metrics["ic_mse"])
            history["val_soh_mae"].append(val_metrics["soh_mae"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(ckpt_path, model, cfg, train_ds.mean, train_ds.std, extra={"epoch": epoch, "val_metrics": val_metrics})

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s. Best val_loss={best_val_loss:.5f}. Checkpoint: {ckpt_path}")

    # --- Final test-set evaluation (fully unseen cells) ---------------------
    test_metrics = evaluate(model, test_loader, tcfg["soh_loss_weight"], device)
    print(f"Test (unseen cells) -> loss={test_metrics['loss']:.5f} ic_mse={test_metrics['ic_mse']:.5f} soh_mae={test_metrics['soh_mae']:.5f}")

    # --- Sanity check: beat the "predict training-mean IC curve" baseline --
    baseline_val_mse = mean_curve_baseline_mse(train_ds, val_ds)
    baseline_test_mse = mean_curve_baseline_mse(train_ds, test_ds)
    print(
        f"\nSanity check (predict-train-mean-curve baseline):\n"
        f"  val  -> baseline_ic_mse={baseline_val_mse:.5f}  model_ic_mse={history['val_ic_mse'][-1]:.5f}  "
        f"{'PASS (model beats baseline)' if history['val_ic_mse'][-1] < baseline_val_mse else 'FAIL (model does NOT beat baseline)'}\n"
        f"  test -> baseline_ic_mse={baseline_test_mse:.5f}  model_ic_mse={test_metrics['ic_mse']:.5f}  "
        f"{'PASS (model beats baseline)' if test_metrics['ic_mse'] < baseline_test_mse else 'FAIL (model does NOT beat baseline)'}"
    )

    metrics_path = os.path.join(_ROOT, "checkpoints", "metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "history": history,
                "test_metrics": test_metrics,
                "baseline_val_ic_mse": baseline_val_mse,
                "baseline_test_ic_mse": baseline_test_mse,
                "n_params": n_params,
                "elapsed_sec": elapsed,
            },
            f,
            indent=2,
        )
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
