import torch

from models.motion_predictor import ProbabilisticMotionPredictor, gaussian_nll_loss


def test_forward_pass_shapes():
    B, T_hist, F, T_fut, H = 6, 10, 5, 5, 32
    model = ProbabilisticMotionPredictor(num_fields=F, hidden_dim=H, future_len=T_fut)
    history = torch.randn(B, T_hist, F)
    # Put history roughly in physical-ish ranges so normalize() doesn't blow up.
    history[..., 0] *= 50   # x
    history[..., 1] *= 50   # y
    history[..., 2] = history[..., 2].abs() * 5 + 10  # speed
    history[..., 3] = history[..., 3].clamp(-3, 3)     # heading
    history[..., 4] *= 2    # accel

    out = model(history)
    assert out["mu_norm"].shape == (B, T_fut, F)
    assert out["logvar_norm"].shape == (B, T_fut, F)
    assert out["mu_phys"].shape == (B, T_fut, F)
    assert out["sigma_phys"].shape == (B, T_fut, F)
    assert torch.all(out["sigma_phys"] > 0)
    assert torch.all(torch.isfinite(out["mu_phys"]))


def test_loss_is_finite_and_scalar():
    B, T_hist, F, T_fut, H = 4, 10, 5, 5, 16
    model = ProbabilisticMotionPredictor(num_fields=F, hidden_dim=H, future_len=T_fut)
    history = torch.randn(B, T_hist, F)
    future = torch.randn(B, T_fut, F)
    out = model(history)
    target_norm = model.normalize(future)
    loss = gaussian_nll_loss(out["mu_norm"], out["logvar_norm"], target_norm)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_gradients_flow_to_all_parameters():
    model = ProbabilisticMotionPredictor(num_fields=5, hidden_dim=16, future_len=3)
    history = torch.randn(2, 10, 5)
    future = torch.randn(2, 3, 5)
    out = model(history)
    target_norm = model.normalize(future)
    loss = gaussian_nll_loss(out["mu_norm"], out["logvar_norm"], target_norm)
    loss.backward()
    n_missing = sum(1 for p in model.parameters() if p.grad is None)
    assert n_missing == 0, f"{n_missing} parameters received no gradient"
