"""
train.py -- Day 16: EKF + Neural Late Fusion

Training strategy (documented choice): the neural predictor is trained
FIRST and then FROZEN. Its inference-time (prior-sampled, deterministic
best-guess) predictions are cached, together with the deterministic EKF
candidate trajectories (EKF needs no training at all), and the fusion
network is trained on top of these fixed inputs against Smooth L1 loss.

Why train sequentially rather than jointly: the fusion network's job is to
learn to arbitrate between a FIXED neural predictor and FIXED classical
model outputs (that is what "late fusion" means, and how the paper's
abstract describes it: "fuses the output of Trajectron++ ... without
modifying the baseline architecture"). Training them jointly would let
gradients from the fusion loss reshape the neural predictor itself, which
would contradict the "late fusion of frozen/independent predictors" setup
the abstract describes.

Evaluates ADE/FDE for three methods, overall and broken out per motion
regime (straight / accel_decel / turning):
    - "EKF best-of-4" : per-sample best candidate among the 4 EKF models
                         (standard oracle/min-over-candidates baseline)
    - "Neural-only"   : the neural predictor's own best-guess trajectory
    - "Fused"         : the trained LateFusionNetwork's output
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import TrajectoryDataset, collate_scenes, REGIMES
from src.ekf_motion_models import build_ekf_bank, run_ekf_bank
from src.neural_predictor import NeuralTrajectoryPredictor
from src.fusion_network import LateFusionNetwork
from src.utils import set_seed, ade, fde


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_neural_predictor(model, train_loader, val_loader, epochs, lr, weight_decay, device, kl_weight=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    recon_loss_fn = nn.SmoothL1Loss()
    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_recon, total_kl, n_batches = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            history = batch["history"].to(device)
            future = batch["future"].to(device)
            optimizer.zero_grad()
            pred, kl = model(history, future=future, future_len=future.shape[1])
            recon = recon_loss_fn(pred, future)
            loss = recon + kl_weight * kl
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            n_batches += 1

        val_ade = evaluate_neural_ade(model, val_loader, device)
        print(f"  [predictor] epoch {epoch:2d}/{epochs} | train_loss={total_loss/n_batches:.4f} "
              f"(recon={total_recon/n_batches:.4f}, kl={total_kl/n_batches:.4f}) | val_ADE={val_ade:.4f}")
    return model


@torch.no_grad()
def evaluate_neural_ade(model, loader, device):
    model.eval()
    all_pred, all_gt = [], []
    for batch in loader:
        history = batch["history"].to(device)
        future = batch["future"].to(device)
        pred, _ = model(history, future=None, future_len=future.shape[1], sample_z=False)
        all_pred.append(pred.cpu().numpy())
        all_gt.append(future.cpu().numpy())
    pred = np.concatenate(all_pred, axis=0)
    gt = np.concatenate(all_gt, axis=0)
    return ade(pred, gt)


@torch.no_grad()
def precompute_split(dataset, predictor, ekf_models, dt, future_len, device):
    """Runs the frozen neural predictor + the deterministic EKF bank once
    over an entire split and caches everything needed for fusion training,
    avoiding recomputation every fusion epoch.
    """
    predictor.eval()
    N = len(dataset)
    history_t = torch.stack([dataset[i]["history"] for i in range(N)])   # [N, T_h, 2]
    future_t = torch.stack([dataset[i]["future"] for i in range(N)])     # [N, T_f, 2]
    regime_idx = torch.tensor([dataset[i]["regime_idx"] for i in range(N)], dtype=torch.long)

    neural_pred, _ = predictor(history_t.to(device), future=None, future_len=future_len, sample_z=False)
    neural_pred = neural_pred.cpu()  # [N, T_f, 2]

    ekf_cand = np.zeros((N, len(ekf_models), future_len, 2), dtype=np.float32)
    ekf_unc = np.zeros((N, len(ekf_models), future_len), dtype=np.float32)
    for i in range(N):
        history_np = dataset[i]["history_np"]
        cand, unc = run_ekf_bank(ekf_models, history_np, dt, future_len)
        ekf_cand[i] = cand
        ekf_unc[i] = unc

    return {
        "history": history_t,
        "future": future_t,
        "regime_idx": regime_idx,
        "neural_pred": neural_pred,
        "ekf_cand": torch.from_numpy(ekf_cand),
        "ekf_unc": torch.from_numpy(ekf_unc),
    }


def train_fusion_network(fusion, cache, epochs, batch_size, lr, weight_decay, device):
    ds = TensorDataset(cache["neural_pred"], cache["ekf_cand"], cache["ekf_unc"], cache["future"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(fusion.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()
    fusion.to(device)
    loss_history = []
    for epoch in range(1, epochs + 1):
        fusion.train()
        total_loss, n_batches = 0.0, 0
        for neural_pred, ekf_cand, ekf_unc, future in loader:
            neural_pred, ekf_cand, ekf_unc, future = (
                neural_pred.to(device), ekf_cand.to(device), ekf_unc.to(device), future.to(device)
            )
            optimizer.zero_grad()
            fused, weights = fusion(neural_pred, ekf_cand, ekf_unc)
            loss = loss_fn(fused, future)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        loss_history.append(avg_loss)
        print(f"  [fusion] epoch {epoch:2d}/{epochs} | train_loss={avg_loss:.4f}")
    return fusion, loss_history


def min_over_candidates_metric(cand: np.ndarray, gt: np.ndarray, metric_fn):
    """cand: [N, K, T, 2], gt: [N, T, 2]. Returns the per-sample best (min)
    metric value across the K candidates, then averages over N -- the
    standard "best-of-K" oracle evaluation used in multimodal trajectory
    prediction.
    """
    N, K = cand.shape[0], cand.shape[1]
    per_sample_best = np.full(N, np.inf)
    for k in range(K):
        vals = np.array([metric_fn(cand[i, k], gt[i]) for i in range(N)])
        per_sample_best = np.minimum(per_sample_best, vals)
    return float(per_sample_best.mean())


def evaluate_all(fusion, cache, device):
    fusion.eval()
    with torch.no_grad():
        fused, _ = fusion(cache["neural_pred"].to(device), cache["ekf_cand"].to(device), cache["ekf_unc"].to(device))
    fused = fused.cpu().numpy()
    neural = cache["neural_pred"].numpy()
    ekf_cand = cache["ekf_cand"].numpy()
    gt = cache["future"].numpy()
    regime_idx = cache["regime_idx"].numpy()

    def per_slice(mask=None):
        if mask is None:
            f, n, e, g = fused, neural, ekf_cand, gt
        else:
            f, n, e, g = fused[mask], neural[mask], ekf_cand[mask], gt[mask]
        return {
            "ekf_best": (min_over_candidates_metric(e, g, ade), min_over_candidates_metric(e, g, fde)),
            "neural": (ade(n, g), fde(n, g)),
            "fused": (ade(f, g), fde(f, g)),
        }

    results = {"overall": per_slice()}
    for r_idx, r_name in enumerate(REGIMES):
        mask = regime_idx == r_idx
        if mask.sum() > 0:
            results[r_name] = per_slice(mask)
    return results


def print_results_table(results: dict):
    header = f"{'Segment':<14}{'Method':<12}{'ADE':>8}{'FDE':>8}"
    print(header)
    print("-" * len(header))
    for segment, methods in results.items():
        for method, (a, f) in methods.items():
            print(f"{segment:<14}{method:<12}{a:>8.4f}{f:>8.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])
    device = torch.device(cfg["train"]["device"])

    d = cfg["data"]
    print("Building synthetic datasets (straight / accel_decel / turning regimes)...")
    train_ds = TrajectoryDataset(d["num_train"], d["history_len"], d["future_len"], d["dt"], d["noise_std"], d["seed"])
    val_ds = TrajectoryDataset(d["num_val"], d["history_len"], d["future_len"], d["dt"], d["noise_std"], d["seed"] + 1)
    test_ds = TrajectoryDataset(d["num_test"], d["history_len"], d["future_len"], d["dt"], d["noise_std"], d["seed"] + 2)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_scenes)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_scenes)

    print("\n=== Stage 1: training neural predictor (frozen afterwards) ===")
    np_cfg = cfg["neural_predictor"]
    predictor = NeuralTrajectoryPredictor(
        hidden_size=np_cfg["hidden_size"], latent_size=np_cfg["latent_size"],
        encoder_layers=np_cfg["encoder_layers"], decoder_layers=np_cfg["decoder_layers"],
    )
    predictor = train_neural_predictor(
        predictor, train_loader, val_loader,
        epochs=cfg["train"]["predictor_epochs"], lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"], device=device,
    )
    for p in predictor.parameters():
        p.requires_grad_(False)

    print("\n=== Stage 2: computing EKF candidates + caching neural predictions ===")
    ekf_models = build_ekf_bank(cfg["ekf"])
    t0 = time.time()
    train_cache = precompute_split(train_ds, predictor, ekf_models, d["dt"], d["future_len"], device)
    val_cache = precompute_split(val_ds, predictor, ekf_models, d["dt"], d["future_len"], device)
    test_cache = precompute_split(test_ds, predictor, ekf_models, d["dt"], d["future_len"], device)
    print(f"  precompute done in {time.time()-t0:.1f}s "
          f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)} scenes)")

    print("\n=== Stage 3: training fusion network ===")
    f_cfg = cfg["fusion"]
    fusion = LateFusionNetwork(
        future_len=d["future_len"], num_ekf=f_cfg["num_candidates"] - 1,
        embed_dim=f_cfg["embed_dim"], attn_hidden=f_cfg["attn_hidden"], residual_hidden=f_cfg["residual_hidden"],
    )
    fusion, loss_history = train_fusion_network(
        fusion, train_cache, epochs=cfg["train"]["fusion_epochs"], batch_size=cfg["train"]["batch_size"],
        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"], device=device,
    )

    # sanity check on the loss curve
    assert loss_history[-1] < loss_history[0], "fusion training loss did not decrease -- investigate before shipping"

    print("\n=== Stage 4: evaluation on held-out TEST split ===")
    results = evaluate_all(fusion, test_cache, device)
    print_results_table(results)

    ckpt_path = cfg["train"]["checkpoint_path"]
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        "predictor_state_dict": predictor.state_dict(),
        "fusion_state_dict": fusion.state_dict(),
        "config": cfg,
    }, ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
