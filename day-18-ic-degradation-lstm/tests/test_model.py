"""
tests/test_model.py
Shape / sanity tests for the ChargeIC-LSTM model itself.
"""
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.chargeic_lstm import ChargeICLSTM, chargeic_loss  # noqa: E402


def make_model():
    return ChargeICLSTM(
        input_size=3, hidden_size=64, num_layers=2, dropout=0.2,
        ic_grid_size=50, ic_head_hidden=128, soh_head_hidden=32,
    )


def test_forward_shapes():
    model = make_model()
    x = torch.randn(8, 120, 3)
    ic_pred, soh_pred = model(x)
    assert ic_pred.shape == (8, 50)
    assert soh_pred.shape == (8,)


def test_forward_no_nan():
    model = make_model()
    x = torch.randn(16, 120, 3) * 5.0  # larger scale to stress-test
    ic_pred, soh_pred = model(x)
    assert not torch.isnan(ic_pred).any()
    assert not torch.isnan(soh_pred).any()
    assert not torch.isinf(ic_pred).any()
    assert not torch.isinf(soh_pred).any()


def test_ic_pred_nonnegative():
    """IC head uses softplus, so outputs must be strictly >= 0."""
    model = make_model()
    x = torch.randn(32, 120, 3)
    ic_pred, _ = model(x)
    assert (ic_pred >= 0).all()


def test_soh_pred_in_bounds():
    model = make_model()
    x = torch.randn(32, 120, 3)
    _, soh_pred = model(x)
    assert (soh_pred >= 0.55).all() and (soh_pred <= 1.02).all()


def test_single_sample_batch():
    """B=1 edge case should not break the LSTM/head pipeline."""
    model = make_model()
    x = torch.randn(1, 120, 3)
    ic_pred, soh_pred = model(x)
    assert ic_pred.shape == (1, 50)
    assert soh_pred.shape == (1,)


def test_loss_is_scalar_and_finite():
    model = make_model()
    x = torch.randn(4, 120, 3)
    ic_pred, soh_pred = model(x)
    ic_true = torch.rand(4, 50)
    soh_true = torch.rand(4) * 0.4 + 0.6
    total, ic_mse, soh_mse = chargeic_loss(ic_pred, ic_true, soh_pred, soh_true, soh_weight=0.5)
    assert total.dim() == 0
    assert torch.isfinite(total)
    assert torch.isfinite(ic_mse)
    assert torch.isfinite(soh_mse)


def test_train_one_batch_loss_decreases():
    """A handful of gradient steps on ONE fixed batch should reduce the loss."""
    torch.manual_seed(0)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x = torch.randn(16, 120, 3)
    ic_true = torch.rand(16, 50)
    soh_true = torch.rand(16) * 0.4 + 0.6

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        ic_pred, soh_pred = model(x)
        loss, _, _ = chargeic_loss(ic_pred, ic_true, soh_pred, soh_true, soh_weight=0.5)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
