"""Shape + backward-pass smoke tests for PART."""
import torch

from src.models.part_model import (
    PARTConfig, PARTModel, UncertaintyAwareSupervision, part_loss,
)


def _make():
    cfg = PARTConfig(model_dim=32, point_feat_dim=16, n_heads=4, n_layers=2, n_queries=8)
    return cfg, PARTModel(cfg)


def test_forward_shapes():
    cfg, model = _make()
    B, P = 2, 128
    radar_points = torch.randn(B, P, 5)
    radar_points[..., 4] = radar_points[..., 4].abs()
    point_mask = torch.ones(B, P, dtype=torch.bool)

    out = model(radar_points, point_mask)
    assert out["existence_logits"].shape == (B, cfg.n_queries, 1)
    assert out["surface_points"].shape == (B, cfg.n_queries, 3)
    assert out["velocities"].shape == (B, cfg.n_queries, 2)


def test_masked_points_ignored():
    cfg, model = _make()
    B, P = 2, 64
    radar_points = torch.randn(B, P, 5)
    radar_points[..., 4] = radar_points[..., 4].abs()
    point_mask = torch.ones(B, P, dtype=torch.bool)
    point_mask[:, -20:] = False

    out = model(radar_points, point_mask)
    assert torch.isfinite(out["existence_logits"]).all()


def test_loss_and_backward():
    cfg, model = _make()
    uas = UncertaintyAwareSupervision()
    B, P = 2, 64
    radar_points = torch.randn(B, P, 5)
    radar_points[..., 4] = radar_points[..., 4].abs()
    point_mask = torch.ones(B, P, dtype=torch.bool)

    out = model(radar_points, point_mask)
    matched_gt_mask = torch.rand(B, cfg.n_queries) > 0.5
    existence_targets = uas(matched_gt_mask)
    surface_point_targets = torch.randn(B, cfg.n_queries, 3)
    velocity_targets = torch.randn(B, cfg.n_queries, 2)

    loss, logs = part_loss(out, existence_targets, surface_point_targets,
                            velocity_targets, matched_gt_mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(v)) for v in logs.values())


def test_uncertainty_aware_supervision_soft_targets():
    uas = UncertaintyAwareSupervision(mask_prob=1.0, soft_target=0.5)  # force all matches ambiguous
    matched = torch.ones(2, 5, dtype=torch.bool)
    targets = uas(matched)
    assert torch.allclose(targets, torch.full_like(targets, 0.5))
