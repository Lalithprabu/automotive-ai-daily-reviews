"""Shape + backward-pass + DAPSE smoke tests for SSDS + DAPSE."""
import torch

from src.models.ssds_dapse import (
    SSDSConfig, SSDSDenoiser, dapse_guided_step, comfort_energy, adversarial_proximity_energy,
)


def _make():
    cfg = SSDSConfig(traj_feat_dim=3, ctx_feat_dim=8, model_dim=32, cond_dim=32,
                      n_heads=4, n_dual_blocks=1, n_single_blocks=1)
    return cfg, SSDSDenoiser(cfg)


def test_forward_shape():
    cfg, model = _make()
    B, A, H, C = 2, 4, 8, 6
    x_t = torch.randn(B, A, H, 3)
    ctx_tokens = torch.randn(B, C, cfg.ctx_feat_dim)
    timesteps = torch.randint(0, 1000, (B,))
    nav_embed = torch.randn(B, cfg.cond_dim)

    out = model(x_t, ctx_tokens, timesteps, nav_embed)
    assert out.shape == (B, A, H, 3)


def test_backward_pass():
    cfg, model = _make()
    B, A, H, C = 2, 4, 8, 6
    x_t = torch.randn(B, A, H, 3)
    ctx_tokens = torch.randn(B, C, cfg.ctx_feat_dim)
    timesteps = torch.randint(0, 1000, (B,))
    nav_embed = torch.randn(B, cfg.cond_dim)

    out = model(x_t, ctx_tokens, timesteps, nav_embed)
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_config_mismatch_raises():
    cfg, model = _make()
    B, A, H = 2, 4, 8
    x_t = torch.randn(B, A, H, 3)
    bad_ctx = torch.randn(B, 6, cfg.ctx_feat_dim + 1)  # wrong feature width
    timesteps = torch.randint(0, 1000, (B,))
    nav_embed = torch.randn(B, cfg.cond_dim)
    try:
        model(x_t, bad_ctx, timesteps, nav_embed)
        assert False, "expected a shape error for mismatched ctx_feat_dim"
    except RuntimeError:
        pass


def test_dapse_planner_and_scenario_roles():
    cfg, model = _make()
    B, A, H, C = 2, 4, 8, 6
    x_t = torch.randn(B, A, H, 3)
    ctx_tokens = torch.randn(B, C, cfg.ctx_feat_dim)
    timesteps = torch.full((B,), 500)
    nav_embed = torch.randn(B, cfg.cond_dim)

    x0_planner = dapse_guided_step(x_t, model, ctx_tokens, timesteps, nav_embed,
                                    r_t=0.5, energy_fn=comfort_energy, n_inner_steps=2)
    x0_scenario = dapse_guided_step(x_t, model, ctx_tokens, timesteps, nav_embed,
                                     r_t=0.5, energy_fn=adversarial_proximity_energy, n_inner_steps=2)
    assert x0_planner.shape == (B, A, H, 3)
    assert x0_scenario.shape == (B, A, H, 3)
    assert torch.isfinite(x0_planner).all() and torch.isfinite(x0_scenario).all()
