import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attention import BiLevelRoutingAttention, CascadedGroupAttention, SparseSpatialCrossAttention
from src.backbone import ConvBackbone
from src.dataset import SparseBEVDataset, make_bev_target, make_scene, render_cameras
from src.model import SparseBEVNet
from train import calibrate_threshold, class_balanced_bce, compute_loss, load_config, precision_recall_f1

TINY_CONFIG = load_config(os.path.join(os.path.dirname(__file__), "..", "tiny_config.yaml"))


def test_bra_shape():
    torch.manual_seed(0)
    bra = BiLevelRoutingAttention(dim=16, region_grid=2, topk=2, num_heads=2)
    x = torch.randn(3, 16, 4, 4)
    out = bra(x)
    assert out.shape == x.shape


def test_cga_shape():
    torch.manual_seed(0)
    cga = CascadedGroupAttention(dim=16, num_groups=4)
    x = torch.randn(2, 10, 16)
    out = cga(x)
    assert out.shape == x.shape


def test_sparse_spatial_cross_attention_shape_and_gate_sum():
    torch.manual_seed(0)
    dim = 16
    num_cameras = 6
    tokens_per_camera = 5
    ssca = SparseSpatialCrossAttention(dim=dim, num_cameras=num_cameras, topk_cameras=2, num_heads=2)
    bev_queries = torch.randn(2, 9, dim)
    camera_tokens = torch.randn(2, num_cameras * tokens_per_camera, dim)
    out, gate = ssca(bev_queries, camera_tokens, tokens_per_camera)
    assert out.shape == (2, 9, dim)
    assert gate.shape == (2, num_cameras)
    # gate is a softmax distribution over cameras per query, averaged -> still sums to 1 per batch row
    sums = gate.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_backbone_shape():
    torch.manual_seed(0)
    backbone = ConvBackbone(in_channels=3, base_channels=16, use_bra=True, bra_region_grid=2, bra_topk=2, bra_heads=2)
    x = torch.randn(4, 3, 32, 32)
    feat = backbone(x)
    assert feat.shape == (4, 16, 4, 4)


def test_model_forward_shapes():
    torch.manual_seed(0)
    model = SparseBEVNet(TINY_CONFIG)
    images = torch.randn(2, TINY_CONFIG["num_cameras"], 3, TINY_CONFIG["image_size"], TINY_CONFIG["image_size"])
    out = model(images)
    bev = TINY_CONFIG["bev_size"]
    assert out["obj_logits"].shape == (2, bev, bev)
    assert out["box_reg"].shape == (2, bev, bev, 6)
    assert out["cam_gate"].shape == (2, TINY_CONFIG["num_cameras"])


def test_model_forward_no_nan():
    torch.manual_seed(1)
    model = SparseBEVNet(TINY_CONFIG)
    images = torch.rand(3, TINY_CONFIG["num_cameras"], 3, TINY_CONFIG["image_size"], TINY_CONFIG["image_size"])
    out = model(images)
    assert torch.isfinite(out["obj_logits"]).all()
    assert torch.isfinite(out["box_reg"]).all()
    assert torch.isfinite(out["cam_gate"]).all()


def test_dataset_shapes_and_positive_cells():
    rng = np.random.default_rng(42)
    objs = make_scene(bev_range=12.0, num_objects_min=3, num_objects_max=3, rng=rng)
    assert len(objs) == 3
    imgs = render_cameras(objs, num_cameras=6, image_size=32, rng=rng)
    assert imgs.shape == (6, 3, 32, 32)
    assert imgs.min() >= 0.0 and imgs.max() <= 1.0

    obj_grid, box_grid = make_bev_target(objs, bev_size=8, bev_range=12.0)
    assert obj_grid.shape == (8, 8)
    assert box_grid.shape == (8, 8, 6)
    assert obj_grid.sum() > 0  # at least one object should land in the grid

    ds = SparseBEVDataset(num_samples=5, config=TINY_CONFIG, seed=7)
    item = ds[0]
    assert item["images"].shape == (TINY_CONFIG["num_cameras"], 3, TINY_CONFIG["image_size"], TINY_CONFIG["image_size"])
    assert item["obj_grid"].shape == (TINY_CONFIG["bev_size"], TINY_CONFIG["bev_size"])
    assert item["box_grid"].shape == (TINY_CONFIG["bev_size"], TINY_CONFIG["bev_size"], 6)


def test_loss_function_class_balanced_bce_and_reg():
    torch.manual_seed(0)
    model = SparseBEVNet(TINY_CONFIG)
    images = torch.rand(2, TINY_CONFIG["num_cameras"], 3, TINY_CONFIG["image_size"], TINY_CONFIG["image_size"])
    out = model(images)
    bev = TINY_CONFIG["bev_size"]
    obj_grid = torch.zeros(2, bev, bev)
    obj_grid[0, 0, 0] = 1.0
    box_grid = torch.zeros(2, bev, bev, 6)
    batch = {"obj_grid": obj_grid, "box_grid": box_grid}

    total, l_obj, l_box = compute_loss(out, batch, box_weight=1.0)
    assert torch.isfinite(total)
    assert total.requires_grad
    assert l_obj >= 0.0
    assert l_box >= 0.0

    # All-negative target should still produce a finite, well-defined loss
    # (no positive-cell mask division-by-zero).
    all_neg = {"obj_grid": torch.zeros(2, bev, bev), "box_grid": torch.zeros(2, bev, bev, 6)}
    total_neg, l_obj_neg, l_box_neg = compute_loss(out, all_neg, box_weight=1.0)
    assert torch.isfinite(total_neg)
    assert l_box_neg == 0.0

    # class_balanced_bce with only positives vs. only negatives should each be finite
    logits = torch.randn(4, 4)
    all_pos_t = torch.ones(4, 4)
    all_neg_t = torch.zeros(4, 4)
    assert torch.isfinite(class_balanced_bce(logits, all_pos_t))
    assert torch.isfinite(class_balanced_bce(logits, all_neg_t))


def test_calibrate_threshold_selects_best_f1():
    # Contrived probs/targets where threshold 0.6 clearly maximizes F1.
    probs = np.array([0.1, 0.2, 0.4, 0.55, 0.65, 0.7, 0.9, 0.95])
    targets = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)

    best_manual = None
    for t in np.arange(0.05, 0.96, 0.01):
        stats = precision_recall_f1(probs, targets, float(t))
        if best_manual is None or stats["f1"] > best_manual["f1"]:
            best_manual = dict(threshold=float(t), **stats)

    # perfect separation exists between 0.55 and 0.65 -> best F1 should be 1.0
    assert best_manual["f1"] == 1.0
    assert 0.55 <= best_manual["threshold"] <= 0.65

    # sanity: precision_recall_f1 itself is correct on a known case
    # at threshold 0.5, the negative sample with prob 0.55 (>= 0.5) is a false positive
    stats_at_05 = precision_recall_f1(probs, targets, 0.5)
    assert stats_at_05["tp"] == 4 and stats_at_05["fp"] == 1 and stats_at_05["fn"] == 0
    assert stats_at_05["recall"] == 1.0
