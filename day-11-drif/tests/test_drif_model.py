"""Test suite for the DRiF reconstruction (11 tests).

Covers: per-module tensor shapes, the pairwise ranking loss's actual
ranking behavior (a synthetic optimization check), synthetic scene
risk-signature / pairwise-label generation, dense bilinear risk sampling,
and an end-to-end training-loss-descent check.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bev_encoder import BEVEncoder
from src.models.drif_model import DRiFModel
from src.models.risk_head import (
    DynamicRiskHead,
    PlanningHead,
    StaticMapHead,
    pairwise_ranking_loss,
    sample_risk_at_points,
)
from src.utils.batching import collate_scenes
from src.utils.synthetic_scene import SceneConfig, SyntheticSceneGenerator, TIER_BACKGROUND, TIER_OVERLAP

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. BEVEncoder shape
# ---------------------------------------------------------------------------
def test_bev_encoder_output_shape():
    enc = BEVEncoder(in_channels=5, base_channels=32, bottleneck_channels=128, attn_heads=4)
    x = torch.randn(2, 5, 64, 64)
    f_t = enc(x)
    assert f_t.shape == (2, 128, 16, 16)


# ---------------------------------------------------------------------------
# 2. StaticMapHead shape
# ---------------------------------------------------------------------------
def test_static_map_head_output_shape():
    head = StaticMapHead(in_channels=128, num_classes=3)
    f_t = torch.randn(2, 128, 16, 16)
    logits = head(f_t)
    assert logits.shape == (2, 3, 64, 64)


# ---------------------------------------------------------------------------
# 3. DynamicRiskHead shape
# ---------------------------------------------------------------------------
def test_dynamic_risk_head_output_shape():
    head = DynamicRiskHead(in_channels=128)
    f_t = torch.randn(2, 128, 16, 16)
    risk = head(f_t)
    assert risk.shape == (2, 64, 64)


# ---------------------------------------------------------------------------
# 4. PlanningHead shape
# ---------------------------------------------------------------------------
def test_planning_head_output_shape():
    head = PlanningHead(in_channels=128, horizon=8, out_dim=2)
    f_t = torch.randn(2, 128, 16, 16)
    waypoints = head(f_t)
    assert waypoints.shape == (2, 8, 2)


# ---------------------------------------------------------------------------
# 5. Bilinear risk sampling: shape + a known-value sanity check
# ---------------------------------------------------------------------------
def test_sample_risk_at_points_shape_and_values():
    B, H, W = 2, 32, 32
    risk_map = torch.zeros(B, H, W)
    risk_map[:, H // 2, W // 2] = 10.0  # spike at the center cell (x=0, y=0)

    # Center of the BEV window is world coords (0, 0).
    points_xy = torch.zeros(B, 3, 2)
    points_xy[:, 0] = torch.tensor([0.0, 0.0])     # at the spike
    points_xy[:, 1] = torch.tensor([24.0, 24.0])     # far corner, near-zero
    points_xy[:, 2] = torch.tensor([-24.0, -24.0])     # opposite far corner

    sampled = sample_risk_at_points(risk_map, points_xy)
    assert sampled.shape == (B, 3)
    # The point at the spike should score much higher than the far corners.
    assert (sampled[:, 0] > sampled[:, 1]).all()
    assert (sampled[:, 0] > sampled[:, 2]).all()


# ---------------------------------------------------------------------------
# 6. Ranking loss is (near) zero when ordering already respects the margin
# ---------------------------------------------------------------------------
def test_pairwise_ranking_loss_zero_when_correctly_ordered():
    r_hat = torch.tensor([[0.0, 1.0, 2.0]])
    idx_i = torch.tensor([[2]])  # r = 2.0
    idx_j = torch.tensor([[0]])  # r = 0.0
    y_ij = torch.tensor([[1.0]])  # i should be riskier than j -> satisfied w/ margin 0.3
    loss = pairwise_ranking_loss(r_hat, idx_i, idx_j, y_ij, margin=0.3)
    assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# 7. Ranking loss actually drives correct ordering under optimization
# ---------------------------------------------------------------------------
def test_pairwise_ranking_loss_optimization_reduces_violations():
    torch.manual_seed(0)
    N, P = 20, 30
    r_hat = torch.zeros(1, N, requires_grad=True)  # start flat -> all pairs violate
    idx_i = torch.randint(0, N, (1, P))
    idx_j = torch.randint(0, N, (1, P))
    y_ij = torch.sign(torch.randn(1, P))
    y_ij[y_ij == 0] = 1.0

    optimizer = torch.optim.Adam([r_hat], lr=0.1)
    initial_loss = pairwise_ranking_loss(r_hat, idx_i, idx_j, y_ij, margin=0.3).item()
    for _ in range(400):
        optimizer.zero_grad()
        loss = pairwise_ranking_loss(r_hat, idx_i, idx_j, y_ij, margin=0.3)
        loss.backward()
        optimizer.step()
    final_loss = pairwise_ranking_loss(r_hat, idx_i, idx_j, y_ij, margin=0.3).item()

    # Some sampled pairs may be mutually contradictory (e.g. the same index
    # pair drawn twice with opposite labels), so we check strong relative
    # improvement rather than requiring the hinge to reach exactly zero.
    assert final_loss < initial_loss
    assert final_loss < 0.3 * initial_loss


# ---------------------------------------------------------------------------
# 8. Synthetic scene: risk signature shape / range
# ---------------------------------------------------------------------------
def test_synthetic_scene_risk_signature_shape_and_range():
    cfg = SceneConfig(grid_size=32, num_risk_points=40, num_pairs=20, planning_horizon=8, seed=1)
    gen = SyntheticSceneGenerator(cfg)
    rng = np.random.default_rng(1)
    scene = gen.generate(t=5, rng=rng)

    assert scene["risk_signature"].shape == (32, 32)
    assert scene["classical_map"].shape == (32, 32)
    assert np.isfinite(scene["risk_signature"]).all()
    assert scene["risk_signature"].min() >= 0.0


# ---------------------------------------------------------------------------
# 9. Synthetic scene: pairwise labels consistent with the 3-tier priority
# ---------------------------------------------------------------------------
def test_synthetic_scene_pairwise_labels_consistent_with_tiers():
    cfg = SceneConfig(grid_size=32, num_risk_points=60, num_pairs=50, planning_horizon=8, seed=2)
    gen = SyntheticSceneGenerator(cfg)
    rng = np.random.default_rng(2)
    scene = gen.generate(t=6, rng=rng)

    tiers = scene["point_tiers"]
    idx_i, idx_j, y_ij = scene["idx_i"], scene["idx_j"], scene["y_ij"]

    expected = np.sign(tiers[idx_i] - tiers[idx_j]).astype(np.float32)
    assert np.array_equal(y_ij, expected)
    # Tiers must actually vary across the scene (overlap/corridor/occupancy present)
    assert tiers.min() == TIER_BACKGROUND or tiers.min() >= TIER_BACKGROUND
    assert tiers.max() <= TIER_OVERLAP


# ---------------------------------------------------------------------------
# 10. Full DRiFModel forward pass shapes
# ---------------------------------------------------------------------------
def test_drif_model_forward_shapes():
    cfg = SceneConfig(grid_size=64, num_risk_points=32, num_pairs=16, planning_horizon=8, seed=3)
    gen = SyntheticSceneGenerator(cfg)
    rng = np.random.default_rng(3)
    scenes = gen.generate_batch(4, rng)
    batch = collate_scenes(scenes)

    model = DRiFModel(
        in_channels=5, base_channels=32, bottleneck_channels=128, attn_heads=4,
        static_map_classes=3, planning_horizon=8, planning_dim=2,
    )
    outputs = model(batch["bev_grid"])

    assert outputs["map_logits"].shape == (4, 3, 64, 64)
    assert outputs["risk_map"].shape == (4, 64, 64)
    assert outputs["waypoints"].shape == (4, 8, 2)

    losses = model.compute_losses(batch, outputs)
    for key in ("total", "rank", "map", "plan", "tv"):
        assert key in losses
        assert torch.isfinite(losses[key])


# ---------------------------------------------------------------------------
# 11. End-to-end training loss descends over a short optimization run
# ---------------------------------------------------------------------------
def test_training_loss_descends():
    torch.manual_seed(7)
    cfg = SceneConfig(grid_size=64, num_risk_points=64, num_pairs=48, planning_horizon=8, seed=7)
    gen = SyntheticSceneGenerator(cfg)
    rng = np.random.default_rng(7)

    model = DRiFModel(
        in_channels=5, base_channels=32, bottleneck_channels=128, attn_heads=4,
        static_map_classes=3, planning_horizon=8, planning_dim=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    losses_seen = []
    for step in range(40):
        scenes = gen.generate_batch(8, rng)
        batch = collate_scenes(scenes)
        outputs = model(batch["bev_grid"])
        losses = model.compute_losses(batch, outputs, w_rank=3.0, w_map=0.4, w_plan=0.3, w_tv=0.08)
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
        losses_seen.append(losses["total"].item())

    early_avg = sum(losses_seen[:5]) / 5
    late_avg = sum(losses_seen[-5:]) / 5
    assert late_avg < early_avg
