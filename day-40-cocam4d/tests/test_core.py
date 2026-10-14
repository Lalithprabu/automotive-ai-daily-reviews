import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from src.cocam4d import (MotionUncertaintyNet, ModelConfig, DOP, compensate_latency, associate,
                         information_fuse, fuse_agents, pack_dop, unpack_dop, DOP_BYTES, CHI2_3_95)
from src.cocam4d.model import adjacent_causal_mask, compute_loss


def mk(mu, v=(0, 0, 0), pv=1.0, vv=0.1, dyn=1.0):
    return DOP(np.array(mu, float), np.array(v, float), np.full(3, pv), np.full(3, vv), dyn)


def test_shapes():
    m = MotionUncertaintyNet(); o = m(torch.randn(5, 6, 7))
    assert o["vel"].shape == (5, 3) and o["vel_logvar"].shape == (5, 3)
    assert o["dyn_logit"].shape == (5,) and o["pos_sigma"].shape == (5, 3) and (o["pos_sigma"] > 0).all()


def test_mask_window():
    m = adjacent_causal_mask(5, 3)
    assert not m[4, 4] and not m[4, 3] and not m[4, 2] and m[4, 1] and m[0, 1]  # no future, window=3


def test_causality():
    torch.manual_seed(0); m = MotionUncertaintyNet().eval(); x = torch.randn(2, 6, 7)
    y = m.encode(x); x2 = x.clone(); x2[:, 4:] += 5.0
    assert torch.allclose(y[:, :4], m.encode(x2)[:, :4], atol=1e-5)       # past unaffected by future
    assert not torch.allclose(y[:, 5], m.encode(x2)[:, 5], atol=1e-5)


def test_grad_flow():
    m = MotionUncertaintyNet(); o = m(torch.randn(8, 6, 7))
    l, _ = compute_loss(o, {"vel": torch.randn(8, 3), "dyn": torch.ones(8), "pos_err": torch.randn(8, 3)})
    l.backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_unc_loss_stopgrad_target():
    o = {"vel": torch.zeros(2, 3), "vel_logvar": torch.zeros(2, 3), "dyn_logit": torch.zeros(2),
         "pos_sigma": torch.ones(2, 3, requires_grad=True)}
    pe = torch.randn(2, 3, requires_grad=True)
    l, _ = compute_loss(o, {"vel": torch.zeros(2, 3), "dyn": torch.zeros(2), "pos_err": pe}); l.backward()
    assert pe.grad is None or pe.grad.abs().sum() == 0


def test_latency_compensation_math():
    d = compensate_latency(mk([0, 0, 0], v=[10, 0, 0], pv=1.0, vv=0.5), 0.5)
    assert np.allclose(d.mu, [5, 0, 0]) and (d.pos_var > 1.0).all()
    s = compensate_latency(mk([1, 1, 1], v=[10, 0, 0], dyn=0.1), 0.5)
    assert np.allclose(s.mu, [1, 1, 1])                                   # static: not moved


def test_information_filter():
    f = information_fuse(mk([0, 0, 0], pv=4.0), mk([2, 0, 0], pv=4.0))
    assert np.allclose(f.mu, [1, 0, 0]) and np.allclose(f.pos_var, 2.0)  # equal cov: mean, var/2
    g = information_fuse(mk([0, 0, 0], pv=1.0), mk([10, 0, 0], pv=9.0))
    assert abs(g.mu[0] - 1.0) < 1e-9                                      # precision-weighted


def test_gating_and_hungarian():
    tracks = [mk([0, 0, 0]), mk([20, 0, 0])]; dets = [mk([20.5, 0, 0]), mk([0.3, 0, 0]), mk([80, 0, 0])]
    matches, new = associate(tracks, dets)
    assert sorted(matches) == [(0, 1), (1, 0)] and new == [2]
    assert CHI2_3_95 == 7.815


def test_dop_is_35_bytes_and_roundtrips():
    d = mk([12.5, -3.25, 0.8], v=[4, 1, 0], pv=0.25, vv=0.04, dyn=0.9); b = pack_dop(d)
    assert len(b) == DOP_BYTES == 35
    u = unpack_dop(b)
    assert np.allclose(u.mu, d.mu, atol=1e-5) and np.allclose(u.vel, d.vel, atol=1e-2)
    assert np.allclose(np.sqrt(u.pos_var), 0.5, rtol=0.05) and abs(u.p_dyn - 0.9) < 0.01


def test_latcomp_beats_stale_when_velocity_known():
    """Core claim: with delayed messages, velocity compensation shrinks error (exact velocity, 500 ms)."""
    truth_now = np.array([15.0, 0, 0]); v = np.array([10.0, 0, 0])
    stale = [mk(truth_now - v * 0.5, v=v, pv=0.3, vv=0.05)]
    ego = []
    on = fuse_agents([ego, stale], [0.0, 0.5], compensate=True)[0]
    off = fuse_agents([ego, stale], [0.0, 0.5], compensate=False)[0]
    assert np.linalg.norm(on.mu - truth_now) < 0.1 < np.linalg.norm(off.mu - truth_now)


def test_overfit_tiny_batch():
    torch.manual_seed(1); m = MotionUncertaintyNet(ModelConfig(d_model=32, n_layers=2))
    x = torch.randn(16, 6, 7); t = {"vel": torch.randn(16, 3), "dyn": (torch.rand(16) > .5).float(), "pos_err": torch.randn(16, 3) * .1}
    opt = torch.optim.Adam(m.parameters(), 3e-3); first = None
    for i in range(300):
        l, p = compute_loss(m(x), t); opt.zero_grad(); l.backward(); opt.step()
        first = first or p["vel"]
    assert p["vel"] < 0.15 * first
