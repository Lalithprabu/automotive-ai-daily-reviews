import numpy as np
import torch
import yaml
from pathlib import Path

from petmpc import world as Wd
from petmpc.planner import MPC
from petmpc.predictor import (PETPredictor, ResidualMoments, StructuredKoopman, constant_velocity, cvtr_prior,
                              markov_margin, sparsity_mask)
from petmpc.episode import run_episode

CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config.yaml"))


def _line_track(n=20, v=(1.0, 0.2)):
    z = []
    for t in range(n):
        x, y = v[0] * 0.1 * t, v[1] * 0.1 * t
        z.append([x, y, v[0], v[1], np.arctan2(v[1], v[0]), 0.0, np.hypot(*v)])
    return np.array(z)


def test_koopman_recovers_constant_velocity():
    k = StructuredKoopman(0.1, 12, 2.0)
    Z = torch.tensor(_line_track()[-13:])[None]
    K = k.identify(Z, torch.ones(1, 12, dtype=torch.bool), k.K0[None])
    pos = k.propagate(K, Z[:, -1], 20)[0, :, :2]
    exp = Z[0, -1, :2] + torch.arange(1, 21, dtype=torch.float64)[:, None] * 0.1 * torch.tensor([1.0, 0.2])
    assert torch.allclose(pos, exp, atol=1e-3)


def test_sparsity_and_fixed_rows():
    k = StructuredKoopman()
    Z = torch.randn(4, 13, 7, dtype=torch.float64)
    K = k.identify(Z, torch.ones(4, 12, dtype=torch.bool), k.K0.expand(4, 7, 7))
    assert torch.all(K[:, sparsity_mask() == 0] == 0)
    assert torch.allclose(K[:, :2], cvtr_prior(0.1)[:2].expand(4, 2, 7))


def test_stabilisation_bounds_spectral_radius():
    k = StructuredKoopman()
    K = cvtr_prior(0.1).expand(3, 7, 7).clone()
    K[:, 2, 2] = 1.3
    K[:, 3, 3] = 1.2
    Ks = k.stabilise(K)
    idx = torch.tensor([2, 3, 6])
    rho = torch.linalg.eigvals(Ks[:, idx][:, :, idx]).abs().amax(-1)
    assert torch.all(rho <= 1.0 + 1e-9)


def test_rotation_is_captured():
    """A track on a circle: the identified velocity block should rotate velocity (beats constant velocity)."""
    w, s = 0.6, 1.0
    z = []
    for t in range(25):
        th = w * 0.1 * t
        z.append([s / w * np.sin(th), s / w * (1 - np.cos(th)), s * np.cos(th), s * np.sin(th), th, w, s])
    z = np.array(z)
    k = StructuredKoopman(0.1, 12, 0.01)
    Z = torch.tensor(z[:13])[None]
    K = k.identify(Z, torch.ones(1, 12, dtype=torch.bool), k.K0[None])
    pk = k.propagate(K, Z[:, -1], 10)[0, :, :2].numpy()
    pc = constant_velocity(Z[:, -1], 10, 0.1)[0].numpy()
    truth = z[13:23, :2]
    assert np.linalg.norm(pk - truth, axis=1).mean() < 0.5 * np.linalg.norm(pc - truth, axis=1).mean()


def test_markov_margin_formula_and_coverage():
    S = torch.tensor([0.04, 0.25])
    assert torch.allclose(markov_margin(S, 0.25, cap=10), torch.tensor([0.4, 1.0]))
    # Markov guarantee on a heavy-tailed error: P(|e| > m) <= delta
    e = torch.distributions.StudentT(3.0).sample((200000, 2)).double() * 0.2
    S = (e ** 2).sum(-1).mean()
    m = markov_margin(S[None], 0.25, cap=100)[0]
    assert ((e ** 2).sum(-1).sqrt() > m).double().mean() <= 0.25


def test_residual_moments_update():
    rm = ResidualMoments(N=3, beta=0.5, n_prior=0.0)
    rm.push_prediction(0, 0, torch.zeros(3, 2, dtype=torch.float64))
    e2 = rm.update(0, 1, torch.tensor([0.3, 0.4], dtype=torch.float64))
    assert abs(float(e2[0]) - 0.25) < 1e-12
    assert torch.isnan(e2[1])
    S = rm.second_moment(0)                                   # n_prior = 0 -> pure track estimate
    assert abs(float(S[0]) - (0.5 * 0.05 ** 2 + 0.5 * 0.25)) < 1e-9


def test_predictor_shapes_and_finiteness():
    w = Wd.World("crowd", 3, n_ped=6)
    p = PETPredictor(N=20)
    for t in range(30):
        pos, m = p.predict([a.hist for a in w.agents], t)
        w.step()
    assert pos.shape == (6, 20, 2) and m.shape == (6, 20)
    assert torch.isfinite(pos).all() and torch.isfinite(m).all() and (m >= 0).all()


def test_mpc_respects_hard_constraint_when_feasible():
    mpc = MPC(N=10)
    pred = np.tile(np.array([[2.0, 0.0]]), (1, 10, 1))                   # static obstacle 2 m ahead
    for _ in range(40):
        v, w, u, plan = mpc.solve(np.array([0.0, 0.0, 0.0]), np.array([20.0, 0.0]), pred, np.zeros((1, 10)),
                                  ["obs"], np.array([Wd.R_OBS]), np.zeros(2))
    d = np.linalg.norm(plan - pred[0, 0], axis=1)
    assert d.min() >= Wd.R_ROBOT + Wd.R_OBS - 1e-2


def test_paired_seeds_identical_agent_motion():
    a = run_episode("obstacle", 5, "koopman_nominal", CFG, log=True)
    b = run_episode("obstacle", 5, "pet_mpc", CFG, log=True)
    n = min(len(a["frames"]), len(b["frames"]))
    assert np.allclose(a["frames"][n - 1]["agents"], b["frames"][n - 1]["agents"])


def test_margins_widen_with_horizon():
    w = Wd.World("obstacle", 7)
    p = PETPredictor(N=20)
    for t in range(80):
        _, m = p.predict([a.hist for a in w.agents], t)
        w.step()
    assert float(m[0, -1]) > float(m[0, 0])
