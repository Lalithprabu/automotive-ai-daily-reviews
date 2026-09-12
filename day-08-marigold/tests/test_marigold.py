"""Shape / gradient-flow / affine-alignment smoke tests for the Marigold
reconstruction. Run with: pytest tests/ -v
"""
import torch
from src.models.marigold import (
    MarigoldConfig,
    MarigoldUNet,
    TrailingDDIMScheduler,
    MarigoldDepthModel,
    ensemble_depth_predictions,
)


def test_unet_forward_shape():
    cfg = MarigoldConfig(base_channels=16, time_embed_dim=32)
    unet = MarigoldUNet(cfg)
    B, C, H, W = 2, cfg.latent_channels, 16, 16
    noisy = torch.randn(B, C, H, W)
    img_lat = torch.randn(B, C, H, W)
    t = torch.randint(0, cfg.num_train_timesteps, (B,))
    out = unet(noisy, img_lat, t)
    assert out.shape == (B, C, H, W)


def test_trailing_timesteps_include_final_step():
    cfg = MarigoldConfig(num_train_timesteps=1000, num_inference_steps=4)
    sched = TrailingDDIMScheduler(cfg)
    ts = sched.trailing_timesteps()
    assert len(ts) == 4
    assert ts[0].item() == cfg.num_train_timesteps - 1  # trails from the noisiest step
    assert (ts[:-1] > ts[1:]).all()  # strictly decreasing


def test_full_denoise_loop_shape():
    cfg = MarigoldConfig(base_channels=16, time_embed_dim=32, num_inference_steps=3)
    unet = MarigoldUNet(cfg)
    sched = TrailingDDIMScheduler(cfg)
    B, C, H, W = 2, cfg.latent_channels, 8, 8
    img_lat = torch.randn(B, C, H, W)
    depth_lat = sched.denoise(unet, img_lat)
    assert depth_lat.shape == (B, C, H, W)
    assert torch.isfinite(depth_lat).all()


def test_end_to_end_train_step_and_backward():
    cfg = MarigoldConfig(base_channels=16, time_embed_dim=32, num_inference_steps=2)
    model = MarigoldDepthModel(cfg)
    B, H, W = 2, 32, 32
    image = torch.rand(B, 3, H, W) * 2 - 1
    depth_latent_gt = torch.randn(B, cfg.latent_channels, H // 4, W // 4)

    loss = model.forward_train_step(image, depth_latent_gt)
    assert loss.dim() == 0 and torch.isfinite(loss)

    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g == g for g in grad_norms)  # no NaNs
    assert sum(grad_norms) > 0.0  # gradients actually flow


def test_predict_depth_shape_single_and_ensembled():
    cfg = MarigoldConfig(base_channels=16, time_embed_dim=32, num_inference_steps=2)
    model = MarigoldDepthModel(cfg)
    B, H, W = 1, 32, 32
    image = torch.rand(B, 3, H, W) * 2 - 1

    single = model.predict_depth(image, num_ensemble=1)
    assert single.shape == (B, 1, H, W)

    ensembled = model.predict_depth(image, num_ensemble=4)
    assert ensembled.shape == (B, 1, H, W)
    assert torch.isfinite(ensembled).all()


def test_ensemble_affine_alignment_recovers_scaled_shifted_copies():
    ref = torch.randn(1, 1, 8, 8)
    # Two copies with different (a, b) affine distortions of the same underlying map.
    p1 = 2.0 * ref + 5.0
    p2 = 0.5 * ref - 1.0
    result = ensemble_depth_predictions([ref, p1, p2])
    # After alignment, result should be very close to ref (the alignment reference).
    assert torch.allclose(result, ref, atol=1e-3)


def test_parameter_count_is_reasonable_for_smoke_scale():
    cfg = MarigoldConfig(base_channels=16, time_embed_dim=32)
    model = MarigoldDepthModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    assert 0 < n_params < 5_000_000  # far below the real ~866M SD-based checkpoint, by design
