import numpy as np
import torch

from dfm.baselines import sqrt_fit
from dfm.data import (N_FEAT, N_MAX, condition_folds, conditions, cycle_features, lifetime, make_cell, make_dataset,
                      normalise, soh_curve)
from dfm.model import (VARIANTS, DecayFormer, FreeHead, MasterCurveHead, build, interp1d, trajectory_loss)


def test_dataset_shapes_and_lifetimes():
    d = make_dataset(cells_per_cond=1)
    assert d["x"].shape == (48, 100, N_FEAT)
    assert np.all((d["life"] > 100) & (d["life"] <= N_MAX))
    assert np.isfinite(d["x"]).all()


def test_folds_are_condition_exclusive():
    folds = condition_folds(6, 0)
    allc = np.concatenate(folds)
    assert len(allc) == len(conditions()) == len(set(allc.tolist()))


def test_master_curve_monotone():
    torch.manual_seed(0)
    head = MasterCurveHead(32)
    out = head(torch.randn(16, 32) * 3)
    s = head.soh(out, torch.linspace(0, N_MAX, 400))
    assert (s[:, 1:] - s[:, :-1] <= 1e-6).all()


def test_master_curve_hits_80pct_at_lifetime():
    head = MasterCurveHead(8)
    out = head(torch.randn(5, 8))
    life = out["log_life"].exp()
    s = head.soh(out, life[:, None])
    assert torch.allclose(s[:, 0], torch.full((5,), 0.8), atol=1e-5)


def test_recurrence_ignores_unobserved_cycles():
    torch.manual_seed(0)
    m = DecayFormer(N_FEAT).eval()
    x = torch.randn(2, 100, N_FEAT)
    valid = torch.arange(100)[None].expand(2, -1) < 40
    x2 = x.clone(); x2[:, 40:] = torch.randn(2, 60, N_FEAT) * 10
    with torch.no_grad():
        a, b = m(x, valid), m(x2, valid)
    assert torch.allclose(a["log_life"], b["log_life"], atol=1e-5)
    assert torch.allclose(a["G"], b["G"], atol=1e-5)


def test_interp_extrapolates_linearly():
    xp = torch.tensor([0.0, 1.0, 2.0])
    fp = torch.tensor([[0.0, 1.0, 3.0]])
    q = torch.tensor([[0.5, 1.5, 3.0, -1.0]])
    assert torch.allclose(interp1d(q, xp, fp), torch.tensor([[0.5, 2.0, 5.0, -1.0]]))


def test_free_head_lifetime_crossing():
    head = FreeHead(4)
    grid = head.grid
    out = dict(soh_grid=(1.0 - grid / 5000.0)[None])                       # crosses 0.8 at n = 1000
    assert abs(float(head.lifetime(out)[0]) - 1000.0) < 6.0


def test_all_variants_train_step():
    d = make_dataset(cells_per_cond=1)
    grid = torch.from_numpy(d["grid"])
    for name, v in VARIANTS.items():
        m = build(name)
        x = torch.from_numpy(d["x"][:4][:, :, v["feats"]])
        valid = torch.ones(4, 100, dtype=torch.bool)
        out = m(x, valid)
        loss, _ = trajectory_loss(m, out, torch.from_numpy(d["traj"][:4]), grid, torch.from_numpy(d["life"][:4]))
        loss.backward()
        assert torch.isfinite(loss)


def test_sqrt_fit_on_knee_free_cell():
    cell = make_cell((45.0, 0.5, 0.6), 3)
    cell["k_knee"] = 0.0
    f, _ = cycle_features(cell, 100, np.random.default_rng(0))
    grid = np.arange(0, N_MAX + 1, 10.0)
    _, L = sqrt_fit(normalise(f)[None], 100, grid)
    true = lifetime(cell)
    assert abs(L[0] - true) / true < 0.35
