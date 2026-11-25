import math

import numpy as np
import torch

from nr.model import (HistoryDenoiser, InteractionTransformer, constant_velocity, kalman_cv, wta_loss)
from nr.noise import add_noise, staleness
from nr.scene import N_AG, T_F, T_H, make_dataset


def _data(n=64):
    d = make_dataset(n, 3)
    st = torch.from_numpy(d["states"])
    return st[:, :, :T_H], st[:, :, T_H:, :2], torch.from_numpy(d["v2x"])


def test_scene_shapes_and_finite():
    d = make_dataset(50, 0)
    assert d["states"].shape == (50, N_AG, T_H + T_F, 4)
    assert np.isfinite(d["states"]).all()
    assert (d["states"][..., 3] >= 0).all()
    assert not d["v2x"][:, 0].any()


def test_level_zero_is_identity():
    h, _, v = _data()
    n, f = add_noise(h, v, 0.0, torch.Generator().manual_seed(0))
    assert torch.allclose(n[..., :2], h[..., :2], atol=1e-5) and f.min() == 1


def test_noise_grows_with_level_and_v2x_is_worse():
    h, _, v = _data(200)
    g = torch.Generator().manual_seed(0)
    e = []
    for s in (1.0, 3.0):
        n, _ = add_noise(h, v, s, g)
        err = (n[..., -1, :2] - h[..., -1, :2]).norm(dim=-1)
        e.append(err.mean())
        assert err[v].mean() > err[~v].mean()
    assert e[1] > 2 * e[0]


def test_packet_loss_holds_previous_value():
    h, _, v = _data(100)
    n, f = add_noise(h, v, 4.0, torch.Generator().manual_seed(1))
    lost = f[..., 1:] == 0
    assert lost.any()
    assert torch.equal(n[..., 1:, :][lost], n[..., :-1, :][lost])
    assert (f[~v] == 1).all()                                       # sensor-tracked agents never lose messages


def test_staleness():
    f = torch.tensor([[[1, 0, 0, 1, 0]]], dtype=torch.float32)
    assert staleness(f).tolist() == [[[0, 1, 2, 0, 1]]]


def test_predictor_shapes_and_backward():
    h, fut, v = _data(8)
    m = InteractionTransformer()
    o = m(h, torch.ones(h.shape[:3]), v)
    assert o["pred"].shape == (8, N_AG, 6, T_F, 2) and o["logit"].shape == (8, N_AG, 6)
    loss, info = wta_loss(o, fut)
    loss.backward()
    assert math.isfinite(loss.item()) and all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_denoiser_starts_as_identity():
    h, _, v = _data(8)
    dn = HistoryDenoiser()
    out, f = dn(h, torch.ones(h.shape[:3]), v)
    assert torch.allclose(out[..., :2], h[..., :2], atol=1e-5) and torch.allclose(out[..., 3], h[..., 3], atol=1e-5)


def test_constant_velocity_exact_on_straight_track():
    t = torch.arange(T_H, dtype=torch.float32) * 0.1
    h = torch.stack([20 * t, torch.zeros_like(t), torch.zeros_like(t), torch.full_like(t, 20.0)], -1)[None, None]
    p = constant_velocity(h)[0, 0, 0]
    assert torch.allclose(p[-1], torch.tensor([20 * t[-1] + 20 * 3.0, 0.0]), atol=1e-4)


def test_kalman_reduces_white_position_noise():
    torch.manual_seed(0)
    t = torch.arange(T_H, dtype=torch.float32) * 0.1
    clean = torch.stack([25 * t, torch.zeros_like(t), torch.zeros_like(t), torch.full_like(t, 25.0)], -1)
    clean = clean.expand(500, 1, T_H, 4).clone()
    noisy = clean.clone()
    noisy[..., :2] += torch.randn_like(noisy[..., :2]) * 0.6
    kf, _ = kalman_cv(noisy, torch.ones(500, 1, T_H))
    e_raw = (noisy[:, :, -1, :2] - clean[:, :, -1, :2]).norm(dim=-1).mean()
    e_kf = (kf[:, :, -1, :2] - clean[:, :, -1, :2]).norm(dim=-1).mean()
    assert e_kf < 0.7 * e_raw
