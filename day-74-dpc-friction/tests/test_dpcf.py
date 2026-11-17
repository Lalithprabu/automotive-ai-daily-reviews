import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dpcf import vehicle as V  # noqa: E402
from dpcf.control import BoxQP, DPCController, ModelMPC  # noqa: E402
from dpcf.episode import collect, run  # noqa: E402
from dpcf.hankel import DPCPredictor, HankelData, block_hankel, multistep_rmse  # noqa: E402


def lti_data(T=400, seed=0):
    """Exact 2nd-order LTI system with 2 inputs / 2 outputs: Willems' lemma must hold exactly."""
    rng = np.random.default_rng(seed)
    A = np.array([[0.9, 0.1], [-0.1, 0.8]]); B = np.array([[0.5, 0.1], [0.0, 0.3]]); C = np.eye(2)
    x = np.zeros(2); U, Y = [], []
    for _ in range(T):
        u = rng.normal(size=2); x = A @ x + B @ u
        U.append(u); Y.append(C @ x)
    return torch.tensor(np.array(U)), torch.tensor(np.array(Y))


def test_block_hankel_layout():
    w = torch.arange(12, dtype=torch.float64).view(6, 2)
    H = block_hankel(w, 3)
    assert H.shape == (6, 4)
    assert torch.equal(H[:, 0], w[0:3].reshape(-1)) and torch.equal(H[:, 3], w[3:6].reshape(-1))


def test_fundamental_lemma_exact_on_lti():
    u, y = lti_data()
    P = DPCPredictor(HankelData([(u, y)], T_ini=4, N=10), lam=1e-8)
    ut, yt = lti_data(seed=1)
    e = multistep_rmse(P, ut, yt)
    assert e.max() < 1e-3


def test_pooling_concatenates_columns():
    u, y = lti_data(); u2, y2 = lti_data(seed=3)
    a = HankelData([(u, y)], 4, 10); b = HankelData([(u, y), (u2, y2)], 4, 10, stats=a.stats)
    assert b.n_cols == 2 * a.n_cols
    assert torch.allclose(b.Up[:, :a.n_cols], a.Up)


def test_affine_form_matches_forward():
    u, y = lti_data()
    P = DPCPredictor(HankelData([(u, y)], 4, 10), lam=1.0)
    ui, yi = u[:4], y[:4]
    kf = u[4:14, 1]
    d = torch.randn(10, dtype=torch.float64)
    f0, G = P.affine_in_delta(ui, yi, kf)
    direct = P(ui[None], yi[None], torch.stack([d, kf], -1)[None])[0].reshape(-1)
    assert torch.allclose(f0 + G @ d, direct, atol=1e-8)


def test_track_geometry_closed():
    tr = V.Track()
    assert abs((tr.psi[-1] - tr.psi[0]) - 2 * np.pi) < 0.05
    assert tr.kappa.min() < 0 < tr.kappa.max()            # contains an S-bend


def test_frenet_roundtrip():
    tr = V.Track()
    xy = tr.xy_from_frenet(np.array([5.0]), np.array([0.12]))[0]
    i, ey, _ = tr.frenet(xy[0], xy[1], tr.psi[tr.idx(5.0)])
    assert abs(ey - 0.12) < 2e-3


def test_cornering_stiffness_scales_with_mu():
    a, b = V.cornering_stiffness(1.0), V.cornering_stiffness(0.5)
    assert np.isclose(a[0], 2 * b[0]) and np.isclose(a[1], 2 * b[1])


def test_mismatch_hurts_and_pooling_helps_prediction():
    rec = {mu: collect(V.Track(mu_high=mu, mu_low=mu), 500, seed=7) for mu in (1.0, 0.5)}
    test = collect(V.Track(mu_high=0.5, mu_low=0.5), 400, seed=8)
    P = {k: DPCPredictor(HankelData(r, 5, 15)) for k, r in
         dict(m=[rec[0.5]], mm=[rec[1.0]], pool=[rec[1.0], rec[0.5]]).items()}
    e = {k: multistep_rmse(p, *test, stride=3)[:, 0].mean().item() for k, p in P.items()}
    assert e["m"] < e["pool"] < e["mm"]


def test_closed_loop_dpc_and_model_stay_on_track():
    rec = {mu: collect(V.Track(mu_high=mu, mu_low=mu), 500, seed=9) for mu in (1.0, 0.5)}
    P = {"pool": DPCPredictor(HankelData([rec[1.0], rec[0.5]], 5, 15))}
    tr = V.Track(wet=[(0.2, 0.5)])
    qp = BoxQP(15, q=(80, 4), r=0.2, r_d=3)
    r1 = run(tr, DPCController(P, "pool", qp), "dpc", seed=1, laps=0.6)
    r2 = run(tr, ModelMPC(15, qp), "model", seed=1, laps=0.6)
    assert not r1["off_track"] and not r2["off_track"] and r1["max_ey"] < 0.2


def test_qp_respects_bounds():
    qp = BoxQP(10)
    G = np.random.default_rng(0).normal(size=(20, 10)); f0 = np.full(20, 5.0)
    d = qp.solve(f0, G, 0.0)
    assert np.all(np.abs(d) <= V.P["delta_max"] + 1e-4)
    assert np.all(np.abs(np.diff(np.concatenate([[0.0], d]))) <= V.P["ddelta_max"] * V.DT + 1e-4)
