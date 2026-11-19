import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idm.baselines import forecast_metrics, kinematic_forecast            # noqa: E402
from idm.data import CONDS, make_dataset, make_driver, simulate_drive    # noqa: E402
from idm.model import GRUClassifier, IDMAE, patchify, unpatchify        # noqa: E402
from evaluate import paired_auc                                          # noqa: E402


def small():
    return IDMAE(d=32, layers=1, heads=2, ff=64)


def test_patchify_roundtrip():
    x = torch.randn(2, 100, 4)
    assert torch.equal(unpatchify(patchify(x), 4), x)


def test_shapes():
    m = small()
    g, v, r = torch.randn(3, 100, 2), torch.randn(3, 100, 4), torch.randn(3, 100, 1)
    p, e = m.classify(g, v, r)
    assert p.shape == (3, 4) and e.shape == (3, 32)
    assert torch.allclose(p.sum(-1), torch.ones(3), atol=1e-5)
    fg, fv = m.forecast(g, v, r, n_future=4)
    assert fg.shape == g.shape and fv.shape == v.shape


def test_forecast_keeps_history():
    m = small().eval()
    g, v, r = torch.randn(2, 100, 2), torch.randn(2, 100, 4), torch.randn(2, 100, 1)
    fg, fv = m.forecast(g, v, r, n_future=4, steps=2)
    assert torch.equal(fg[:, :80], g[:, :80]) and torch.equal(fv[:, :80], v[:, :80])


def test_future_is_ignored():
    """The forecast must not depend on the (masked) ground-truth future."""
    m = small().eval()
    g, v, r = torch.randn(1, 100, 2), torch.randn(1, 100, 4), torch.randn(1, 100, 1)
    g2, v2 = g.clone(), v.clone()
    g2[:, 80:] += 5; v2[:, 80:] -= 5
    a = m.forecast(g, v, r, 4)
    b = m.forecast(g2, v2, r, 4)
    assert torch.allclose(a[0][:, 80:], b[0][:, 80:]) and torch.allclose(a[1][:, 80:], b[1][:, 80:])


def test_masked_label_cannot_leak():
    """With the label masked, the class embedding input must not matter (it is replaced by [MASK])."""
    m = small().eval()
    g, v, r = torch.randn(2, 100, 2), torch.randn(2, 100, 4), torch.randn(2, 100, 1)
    z = torch.zeros(2, 20, dtype=torch.bool)
    l1 = m(g, v, r, torch.full((2,), 4), z, z)[0]
    l2 = m.classify(g, v, r)[0].log()
    assert torch.allclose(l1.log_softmax(-1), l2, atol=1e-5)


def test_loss_backward_and_unlabelled():
    m = small()
    g, v, r = torch.randn(4, 100, 2), torch.randn(4, 100, 4), torch.randn(4, 100, 1)
    L, parts = m.loss(g, v, r, torch.tensor([0, 1, 2, 3]), torch.zeros(4, dtype=torch.bool))
    L.backward()
    assert parts["cls"] == 0.0 and parts["rec"] > 0
    assert GRUClassifier()(g, v, r).shape == (4, 4)


def test_dataset_location_matched():
    d = make_dataset([3], n_loc=2, win_s=5.0, warm_s=1.0)
    assert d["gaze"].shape == (8, 50, 2) and d["veh"].shape == (8, 50, 4)
    for pr in np.unique(d["pair"]):
        assert sorted(d["y"][d["pair"] == pr]) == list(range(len(CONDS)))
    assert np.isfinite(d["veh"]).all() and np.abs(d["veh"][..., 0]).max() < 3.0


def test_visual_condition_looks_down():
    drv = make_driver(5); drv["suscept"] = 1.0
    base = simulate_drive(drv, np.zeros(600, int), 1, 1, 1)
    vis = simulate_drive(drv, np.full(600, 2), 1, 1, 1)
    assert vis["phone"].any() and not base["phone"].any()
    assert vis["gpitch"].mean() < base["gpitch"].mean() - 2


def test_paired_auc_and_kinematic():
    y = np.array([0, 1, 0, 2]); pair = np.array([0, 0, 1, 1])
    assert paired_auc(np.array([0.1, 0.9, 0.5, 0.2]), y, pair) == 0.5
    g = np.zeros((2, 100, 2), np.float32); v = np.zeros((2, 100, 4), np.float32)
    v[:, :, 0] = np.linspace(0, 1, 100)
    kg, kv = kinematic_forecast(g, v, 20)
    met = forecast_metrics(kg, kv, g, v, 20)
    assert met["lat_ade"].max() < 1e-3 and met["gaze_deg"].max() < 1e-6
