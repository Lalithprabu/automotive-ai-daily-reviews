import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from caa_gmm.data import (N_NB, T_F, T_H, build_raster, make_dataset, path_xy,  # noqa: E402
                          perturb_neighbours, take)
from caa_gmm.losses import gmm_loss, metrics  # noqa: E402
from caa_gmm.model import CAAGMM, RasterBaseline  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return make_dataset(600, 7)


def _fwd(m, b, **kw):
    return m(build_raster(b), b["ego"], b["nb"], b["mask"], **kw)


# ---------------------------------------------------------------- geometry / data
def test_path_continuous_at_junction():
    for b in range(3):
        u = np.array([9.99, 10.0, 10.01])
        p = path_xy(u, 10.0, b)
        assert np.abs(np.diff(p, axis=0)).max() < 0.05


def test_turn_branches_are_mirror_images():
    u = np.linspace(0, 40, 50)
    l, r = path_xy(u, 12.0, 0), path_xy(u, 12.0, 2)
    assert np.allclose(l[:, 0], -r[:, 0]) and np.allclose(l[:, 1], r[:, 1])


def test_dataset_shapes_and_hint_reliability(data):
    assert data["fut"].shape[1:] == (T_F, 2) and data["nb"].shape[1:] == (N_NB, T_H, 4)
    big = make_dataset(3000, 11)
    assert abs((big["hint"] == big["branch"]).float().mean().item() - 0.75) < 0.04


def test_lead_vehicle_slows_ego():
    d = make_dataset(1500, 5)
    f = d["fut"][:, -1, 1]
    straight = d["branch"] == 1
    near = d["has_lead"] & straight
    assert f[near].mean() < f[(~d["has_lead"]) & straight].mean() - 1.0


def test_raster_channels(data):
    r = build_raster(take(data, torch.arange(16)))
    assert r.shape == (16, 4, 64, 64) and r.min() >= 0 and r.max() <= 1
    assert (r[:, 3].sum((1, 2)) > 0).all()                     # ego footprint always present
    b = perturb_neighbours(take(data, torch.arange(16)), drop_p=1.0)
    assert build_raster(b)[:, 2].sum() == 0                    # all neighbours dropped -> empty occupancy


# ---------------------------------------------------------------- model
@pytest.mark.parametrize("M", [CAAGMM, RasterBaseline])
def test_output_shapes(M, data):
    b = take(data, torch.arange(8))
    lo, mu, ls = _fwd(M().eval(), b)
    assert lo.shape == (8, 6) and mu.shape == (8, 6, T_F, 2) and ls.shape == mu.shape
    assert torch.isfinite(mu).all() and ls.min() >= -3.0 and ls.max() <= 1.5


def test_all_neighbours_dropped_is_finite(data):
    m = CAAGMM().eval()
    b = perturb_neighbours(take(data, torch.arange(16)), drop_p=1.0)
    out = _fwd(m, b)
    assert all(torch.isfinite(o).all() for o in out)


def test_attention_weights_normalised_and_masked(data):
    m = CAAGMM().eval()
    b = take(data, torch.arange(32))
    *_, a = _fwd(m, b, return_attn=True)
    total = a["w_int"].sum(1) + a["w_null"]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
    assert (a["w_int"][~b["mask"]] < 1e-6).all()               # masked slots get zero weight
    assert torch.allclose(a["gate"].sum(-1), torch.ones(32), atol=1e-5)
    assert torch.allclose(a["w_map"].sum((1, 2)), torch.ones(32), atol=1e-4)


def test_neighbour_slot_permutation_invariance(data):
    m = CAAGMM().eval()
    b = take(data, torch.arange(8))
    p = torch.randperm(N_NB)
    b2 = dict(b, nb=b["nb"][:, p], mask=b["mask"][:, p])
    for x, y in zip(_fwd(m, b), _fwd(m, b2)):
        assert torch.allclose(x, y, atol=1e-5)


def test_gradient_flows_to_all_branches(data):
    m = CAAGMM()
    b = take(data, torch.arange(32))
    loss, _ = gmm_loss(*_fwd(m, b), b["fut"])
    loss.backward()
    for name in ("raster_enc", "ego_gru", "nb_gru", "int_attn", "map_attn", "gate", "head"):
        g = [p.grad for p in getattr(m, name).parameters()]
        assert any(x is not None and x.abs().sum() > 0 for x in g), name


def test_neighbours_influence_prediction_after_perturbation(data):
    m = CAAGMM().eval()
    b = take(data, torch.arange(16))
    a = _fwd(m, b)[1]
    c = _fwd(m, perturb_neighbours(b, drop_p=1.0))[1]
    assert (a - c).abs().max() > 1e-4


# ---------------------------------------------------------------- loss / metrics
def test_loss_prefers_correct_mode_and_metrics_consistent():
    B, K = 4, 6
    gt = torch.cumsum(torch.full((B, T_F, 2), 0.5), 1)
    mu = gt[:, None].repeat(1, K, 1, 1) + 5.0
    mu[:, 2] = gt
    ls = torch.full((B, K, T_F, 2), -1.0)
    good = torch.full((B, K), -5.0)
    good[:, 2] = 5.0
    l_good, _ = gmm_loss(good, mu, ls, gt)
    l_bad, _ = gmm_loss(good.flip(1), mu, ls, gt)
    assert l_good < l_bad
    m = metrics(good, mu, ls, gt)
    assert m["minADE"] < 1e-5 and m["minFDE"] < 1e-5 and m["MR"] == 0 and m["top1ADE"] < 1e-5


def test_overfit_tiny_batch(data):
    """Core wiring sanity check: the model can drive minADE on a fixed 16-sample batch way down."""
    torch.manual_seed(0)
    m = CAAGMM()
    b = take(data, torch.arange(16))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    start = metrics(*_fwd(m, b), b["fut"])["minADE"]
    for _ in range(250):
        loss, _ = gmm_loss(*_fwd(m, b), b["fut"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    end = metrics(*_fwd(m.eval(), b), b["fut"])["minADE"]
    assert end < 0.25 * start and end < 0.8
