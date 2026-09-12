"""Shape + gradient-flow sanity tests for MC-DeTra."""
import torch

from src.models.mc_detra import MCDeTraConfig, MCDeTra, MotionConsistencyLosses


def _setup():
    cfg = MCDeTraConfig(d_model=16, n_objects=4, n_future_steps=4, n_modes=3,
                         n_blocks=2, n_heads=2, lidar_channels=8, map_feat_dim=8,
                         occupancy_grid=5, knn_k=3)
    model = MCDeTra(cfg)
    mc_losses = MotionConsistencyLosses(cfg)
    return cfg, model, mc_losses


def test_forward_shapes():
    cfg, model, _ = _setup()
    B, H, W, M = 2, 16, 16, 10
    lidar_bev = torch.randn(B, cfg.lidar_channels, H, W)
    map_tokens = torch.randn(B, M, cfg.map_feat_dim)

    out = model(lidar_bev, map_tokens)
    assert out["detection_boxes"].shape == (B, cfg.n_objects, 5)
    assert out["forecast_offsets"].shape == (B, cfg.n_objects, cfg.n_future_steps - 1, cfg.n_modes, 2)
    assert out["headings"].shape == (B, cfg.n_objects, cfg.n_future_steps, cfg.n_modes)
    assert out["positions"].shape == (B, cfg.n_objects, cfg.n_future_steps, cfg.n_modes, 2)
    assert out["t0_query"].shape == (B, cfg.n_objects, cfg.d_model)


def test_positions_anchored_at_detection_center():
    cfg, model, _ = _setup()
    B, H, W, M = 2, 16, 16, 10
    lidar_bev = torch.randn(B, cfg.lidar_channels, H, W)
    map_tokens = torch.randn(B, M, cfg.map_feat_dim)
    out = model(lidar_bev, map_tokens)

    t0_pos_from_positions = out["positions"][:, :, 0, :, :]              # [B, N, K, 2]
    t0_pos_from_detection = out["detection_boxes"][..., :2].unsqueeze(2).expand(-1, -1, cfg.n_modes, -1)
    assert torch.allclose(t0_pos_from_positions, t0_pos_from_detection, atol=1e-5)


def test_motion_consistency_losses_finite():
    cfg, model, mc_losses = _setup()
    B, H, W, M = 2, 16, 16, 10
    lidar_bev = torch.randn(B, cfg.lidar_channels, H, W)
    map_tokens = torch.randn(B, M, cfg.map_feat_dim)
    out = model(lidar_bev, map_tokens)

    past_motion_gt = torch.randn(B, cfg.n_objects, cfg.n_future_steps, 2)
    occupancy_gt = (torch.rand(B, cfg.n_objects, cfg.occupancy_grid, cfg.occupancy_grid) > 0.5).float()
    valid_mask_obj = torch.ones(B, cfg.n_objects)
    valid_mask_traj = torch.ones(B, cfg.n_objects, cfg.n_future_steps, cfg.n_modes)

    mc_out = mc_losses(out["t0_query"], out["headings"], out["positions"],
                        past_motion_gt, occupancy_gt, valid_mask_obj, valid_mask_traj)
    for k, v in mc_out.items():
        assert torch.isfinite(v), f"{k} is not finite"


def test_combined_backward_pass_covers_all_parameters():
    cfg, model, mc_losses = _setup()
    B, H, W, M = 2, 16, 16, 10
    lidar_bev = torch.randn(B, cfg.lidar_channels, H, W)
    map_tokens = torch.randn(B, M, cfg.map_feat_dim)
    out = model(lidar_bev, map_tokens)

    past_motion_gt = torch.randn(B, cfg.n_objects, cfg.n_future_steps, 2)
    occupancy_gt = (torch.rand(B, cfg.n_objects, cfg.occupancy_grid, cfg.occupancy_grid) > 0.5).float()
    valid_mask_obj = torch.ones(B, cfg.n_objects)
    valid_mask_traj = torch.ones(B, cfg.n_objects, cfg.n_future_steps, cfg.n_modes)

    mc_out = mc_losses(out["t0_query"], out["headings"], out["positions"],
                        past_motion_gt, occupancy_gt, valid_mask_obj, valid_mask_traj)

    # Combine with the detection/forecast heads too, so the WHOLE backbone
    # (not just the path feeding MotionConsistencyLosses) gets a gradient
    # signal in this test, matching how the two would jointly train.
    total = mc_out["loss_mc_total"] + out["detection_boxes"].pow(2).mean() + out["positions"].pow(2).mean()
    total.backward()

    all_params = list(model.parameters()) + list(mc_losses.parameters())
    n_with_grad = sum(1 for p in all_params if p.grad is not None and torch.isfinite(p.grad).all())
    assert n_with_grad == len(all_params), (
        f"only {n_with_grad}/{len(all_params)} parameter tensors received finite gradients"
    )
