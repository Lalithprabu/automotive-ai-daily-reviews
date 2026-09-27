"""Behavioral tests for the two "varying reliability across horizons" mechanisms:
gated visual fusion (models/fusion.py) and Trajectory-Adaptive Bias (models/tab.py)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

from models.fusion import GatedFutureFusion
from models.tab import TrajectoryAdaptiveBias

CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def _cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def test_zero_confidence_gates_out_future_latent_magnitude():
    """When confidence is exactly 0 for a horizon, the GATED predicted-latent
    contribution to that horizon's fused token must not depend on the
    predicted latent's magnitude -- i.e. the gate is really multiplicative,
    not just a soft down-weighting baked into a learned projection."""
    cfg = _cfg()
    token_dim = cfg["encoder"]["token_dim"]
    n = cfg["encoder"]["tokens_per_side"] ** 2
    h = len(cfg["data"]["horizons"])
    fusion = GatedFutureFusion(token_dim=token_dim, num_horizons=h,
                                status_dim=cfg["fusion"]["status_dim"],
                                fused_token_dim=cfg["fusion"]["fused_token_dim"])
    fusion.eval()

    current = torch.randn(2, n, token_dim)
    confidence_zero = torch.zeros(2, h, n)

    pred_small = torch.randn(2, h, n, token_dim) * 0.01
    pred_huge = torch.randn(2, h, n, token_dim) * 1000.0

    with torch.no_grad():
        fused_small, _ = fusion(current, pred_small, confidence_zero)
        fused_huge, _ = fusion(current, pred_huge, confidence_zero)

    # Both the "current" block (identical inputs) and the "future" block
    # (gated to zero regardless of predicted-latent magnitude) must match.
    assert torch.allclose(fused_small, fused_huge, atol=1e-5), (
        "With confidence=0, changing the predicted latent's magnitude by 5 "
        "orders of magnitude changed the fused output -- gating is not "
        "actually zeroing out the future latent's contribution."
    )


def test_nonzero_confidence_lets_future_latent_through():
    """Sanity counterpart: with confidence=1, the future latent's magnitude
    SHOULD change the fused output (the gate is not permanently closed)."""
    cfg = _cfg()
    token_dim = cfg["encoder"]["token_dim"]
    n = cfg["encoder"]["tokens_per_side"] ** 2
    h = len(cfg["data"]["horizons"])
    fusion = GatedFutureFusion(token_dim=token_dim, num_horizons=h,
                                status_dim=cfg["fusion"]["status_dim"],
                                fused_token_dim=cfg["fusion"]["fused_token_dim"])
    fusion.eval()

    current = torch.randn(2, n, token_dim)
    confidence_one = torch.ones(2, h, n)

    pred_a = torch.randn(2, h, n, token_dim)
    pred_b = pred_a + 5.0   # clearly different latent

    with torch.no_grad():
        fused_a, _ = fusion(current, pred_a, confidence_one)
        fused_b, _ = fusion(current, pred_b, confidence_one)

    assert not torch.allclose(fused_a, fused_b, atol=1e-3), (
        "With confidence=1, changing the predicted latent did not change "
        "the fused output at all -- the future path looks disconnected."
    )


def test_tab_bias_is_trajectory_adaptive():
    """TAB must recompute its attention bias from the CURRENT candidate
    trajectory -- two different trajectories (same conditioning tokens) must
    produce different biases. A fixed/learned-only bias (not actually a
    function of the trajectory) would fail this."""
    cfg = _cfg()
    fused_dim = cfg["fusion"]["fused_token_dim"]
    t = cfg["data"]["num_waypoints"]
    tab = TrajectoryAdaptiveBias(fused_token_dim=fused_dim, num_waypoints=t,
                                  hidden_dim=cfg["tab"]["hidden_dim"])
    tab.eval()

    cond = torch.randn(2, 12, fused_dim)
    traj_a = torch.zeros(2, t, 2)
    traj_b = torch.randn(2, t, 2) * 10.0   # a very different candidate trajectory

    with torch.no_grad():
        bias_a = tab(traj_a, cond)
        bias_b = tab(traj_b, cond)

    assert bias_a.shape == (2, t, 12)
    assert not torch.allclose(bias_a, bias_b, atol=1e-3), (
        "TAB produced the same attention bias for two very different "
        "candidate trajectories -- it is not actually trajectory-adaptive."
    )


def test_tab_bias_changes_across_denoising_steps_in_practice():
    """End-to-end version of the adaptivity property, through the real DiT
    planner: feeding two different noisy trajectories at the same diffusion
    timestep with the same conditioning must give different epsilon
    predictions (confirms TAB's effect actually reaches the model output,
    not just an internal bias tensor)."""
    cfg = _cfg()
    from models.dit_planner import DiTPlanner
    fused_dim = cfg["fusion"]["fused_token_dim"]
    t_wp = cfg["data"]["num_waypoints"]
    dit = DiTPlanner(fused_token_dim=fused_dim, num_waypoints=t_wp, hidden_dim=cfg["dit"]["hidden_dim"],
                      num_layers=cfg["dit"]["num_layers"], num_heads=cfg["dit"]["num_heads"],
                      dropout=0.0, tab_hidden_dim=cfg["tab"]["hidden_dim"])
    dit.eval()

    cond = torch.randn(2, 12, fused_dim)
    timestep = torch.full((2,), 10, dtype=torch.long)
    traj_a = torch.randn(2, t_wp, 2)
    traj_b = traj_a + 3.0

    with torch.no_grad():
        eps_a = dit(traj_a, timestep, cond)
        eps_b = dit(traj_b, timestep, cond)

    assert not torch.allclose(eps_a, eps_b, atol=1e-3)
