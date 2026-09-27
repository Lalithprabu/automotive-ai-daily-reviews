"""Train PriorMapBEVNet on the synthetic prior-map/live-camera dataset and
evaluate the with-prior vs. without-prior ablation.

Usage:
    python train.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import SyntheticSceneDataset, collate_scenes
from src.geometry import BEVGrid, default_camera_rig
from src.model import PriorMapBEVNet


def focal_loss(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet-style penalty-reduced focal loss. `gt` must have an exact
    1.0 at true object centers (see dataset.py) -- a fuzzy gaussian target
    with no exact 1.0 would silently zero out `pos_mask` (Day-21 lesson)."""
    pos_mask = gt.eq(1).float()
    neg_mask = gt.lt(1).float()
    neg_weight = torch.pow(1 - gt, beta)
    pred_c = pred.clamp(min=1e-6, max=1 - 1e-6)
    pos_loss = torch.log(pred_c) * torch.pow(1 - pred_c, alpha) * pos_mask
    neg_loss = torch.log(1 - pred_c) * torch.pow(pred_c, alpha) * neg_weight * neg_mask
    num_pos = pos_mask.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def balanced_bce(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Average the positive-cell and negative-cell loss separately, then
    sum -- prevents the ~2% positive-pixel map task from being solved by
    predicting all-zero (Day-13 lesson)."""
    eps = 1e-6
    pos = gt > 0.5
    neg = ~pos
    loss = pred.new_tensor(0.0)
    if pos.any():
        loss = loss - torch.log(pred[pos].clamp(min=eps)).mean()
    if neg.any():
        loss = loss - torch.log((1 - pred[neg]).clamp(min=eps)).mean()
    return loss


def box_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.expand_as(pred_boxes)
    denom = mask.sum().clamp(min=1.0)
    return (torch.abs(pred_boxes - gt_boxes) * mask).sum() / denom


def compute_losses(out: dict, batch: dict, box_w: float, map_w: float) -> torch.Tensor:
    l_det = focal_loss(out["det_heatmap"], batch["det_heatmap"])
    l_box = box_loss(out["det_boxes"], batch["det_boxes"], batch["det_mask"])
    l_map = balanced_bce(out["map_heatmap"], batch["map_heatmap"])
    return l_det + box_w * l_box + map_w * l_map


def extract_peaks(heatmap: torch.Tensor, thresh: float = 0.5, k: int = 8):
    """Simple 3x3 local-max peak extraction -> list of (row, col, score)."""
    pooled = torch.nn.functional.max_pool2d(heatmap.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
    is_peak = (heatmap == pooled) & (heatmap > thresh)
    rows, cols = torch.where(is_peak[0])
    scores = heatmap[0, rows, cols]
    order = torch.argsort(scores, descending=True)[:k]
    return [(rows[i].item(), cols[i].item(), scores[i].item()) for i in order]


def detection_prf1(pred_heatmap: torch.Tensor, gt_mask: torch.Tensor, match_dist: float = 2.0):
    tp = fp = fn = 0
    for b in range(pred_heatmap.shape[0]):
        peaks = extract_peaks(pred_heatmap[b])
        gt_rc = torch.nonzero(gt_mask[b, 0] > 0.5).tolist()
        used = set()
        for r, c, _ in peaks:
            best, best_d = None, match_dist + 1
            for gi, (gr, gc) in enumerate(gt_rc):
                if gi in used:
                    continue
                d = ((gr - r) ** 2 + (gc - c) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = gi, d
            if best is not None:
                used.add(best)
                tp += 1
            else:
                fp += 1
        fn += len(gt_rc) - len(used)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


def map_iou(pred: torch.Tensor, gt: torch.Tensor, thresh: float = 0.5):
    pred_b = pred > thresh
    gt_b = gt > 0.5
    inter = (pred_b & gt_b).float().sum().item()
    union = (pred_b | gt_b).float().sum().item()
    return inter / union if union > 0 else 1.0


@torch.no_grad()
def evaluate(model, loader, prior_enabled: bool):
    model.eval()
    precisions, recalls, f1s, ious = [], [], [], []
    for batch in loader:
        out = model(batch["prior_points"], batch["camera_feats"], prior_enabled=prior_enabled)
        p, r, f1 = detection_prf1(out["det_heatmap"], batch["det_mask"])
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        ious.append(map_iou(out["map_heatmap"], batch["map_heatmap"]))
    model.train()
    n = max(len(precisions), 1)
    return {
        "det_precision": sum(precisions) / n,
        "det_recall": sum(recalls) / n,
        "det_f1": sum(f1s) / n,
        "map_iou": sum(ious) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])

    grid = BEVGrid(**cfg["grid"])
    cameras = default_camera_rig()
    train_ds = SyntheticSceneDataset(
        cfg["data"]["train_scenes"], grid, seed=1, occlusion=cfg["data"]["occlusion"]
    )
    val_ds = SyntheticSceneDataset(
        cfg["data"]["val_scenes"], grid, seed=2, occlusion=cfg["data"]["occlusion"]
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate_scenes
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_scenes
    )

    model = PriorMapBEVNet(cameras, grid=grid, channels=cfg["model"]["channels"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PriorMapBEVNet parameters: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    t0 = time.time()
    for epoch in range(cfg["train"]["epochs"]):
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            opt.zero_grad()
            # Train BOTH the prior-enabled and prior-disabled ("old way") paths
            # jointly each step (Day-20/21 lesson: an ablation flag needs both
            # paths trained, or the disabled path is evaluated out-of-distribution).
            out_on = model(batch["prior_points"], batch["camera_feats"], prior_enabled=True)
            out_off = model(batch["prior_points"], batch["camera_feats"], prior_enabled=False)
            loss = compute_losses(
                out_on, batch, cfg["train"]["box_loss_weight"], cfg["train"]["map_loss_weight"]
            ) + compute_losses(
                out_off, batch, cfg["train"]["box_loss_weight"], cfg["train"]["map_loss_weight"]
            )
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch + 1}/{cfg['train']['epochs']}  loss={epoch_loss / n_batches:.4f}")

    elapsed = time.time() - t0
    print(f"training complete in {elapsed:.1f}s")

    metrics_on = evaluate(model, val_loader, prior_enabled=True)
    metrics_off = evaluate(model, val_loader, prior_enabled=False)
    print("With prior map   :", metrics_on)
    print("Without prior map:", metrics_off)

    torch.save({"model_state": model.state_dict(), "config": cfg}, "checkpoint.pt")
    with open("metrics.json", "w") as f:
        json.dump(
            {"with_prior": metrics_on, "without_prior": metrics_off, "params": n_params, "train_seconds": elapsed},
            f,
            indent=2,
        )
    print("Saved checkpoint.pt and metrics.json")


if __name__ == "__main__":
    main()
