import sys, pathlib, numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from kglc.data import make_split, discretise, N_ENT, N_REL, REL_VALUES, VMAX, LANE_W, H0, features_for
from kglc.model import TransE, macro_f1
import math


def _train_small(epochs=25, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tr = make_split(3000, rng); va = make_split(900, rng)
    kg = TransE(32, 6.0, 1.0); opt = torch.optim.Adam(kg.parameters(), lr=0.003)
    e, y = torch.from_numpy(tr.ents), torch.from_numpy(tr.label)
    for _ in range(epochs):
        for i in range(0, len(y), 256):
            opt.zero_grad(); kg.loss(e[i:i+256], y[i:i+256]).backward(); opt.step()
    kg.calibrate_tau(torch.from_numpy(va.ents), torch.from_numpy(va.label))
    return kg.eval()


def test_bell_profile_integrates_to_one_lane():
    assert abs(VMAX * H0 * math.sqrt(math.pi) - LANE_W) < 1e-6


def test_discretise_ids_in_relation_support():
    s = make_split(300, np.random.default_rng(1))
    for r in range(N_REL):
        assert set(s.ents[:, r]).issubset(set(REL_VALUES[r]))
    assert s.ents.max() < N_ENT


def test_posterior_normalised_and_shape():
    kg = TransE(16); kg.log_tau.fill_(0.0)
    e = torch.from_numpy(make_split(20, np.random.default_rng(0)).ents)
    lp = kg.log_posterior(e); assert lp.shape == (20, 3)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(20), atol=1e-5)


def test_gradient_flows_to_entity_and_relation_embeddings():
    kg = TransE(16); s = make_split(64, np.random.default_rng(0))
    kg.loss(torch.from_numpy(s.ents), torch.from_numpy(s.label)).backward()
    assert kg.ent.weight.grad.abs().sum() > 0 and kg.rel.weight.grad.abs().sum() > 0


def test_missing_evidence_equals_uniform_when_all_missing():
    kg = TransE(16); e = torch.from_numpy(make_split(8, np.random.default_rng(0)).ents)
    lp = kg.log_posterior(e, observed=torch.zeros(8, N_REL, dtype=torch.bool))
    assert torch.allclose(lp.exp(), torch.full((8, 3), 1 / 3), atol=1e-5)


def test_explain_sums_to_log_posterior_ratio():
    kg = _train_small(5)
    s = make_split(5, np.random.default_rng(3), fixed_h=1.0)
    e = torch.from_numpy(s.ents); lp = kg.log_posterior(e)
    for i in range(5):
        ex = kg.explain(e[i], ["a"] * N_REL)
        assert abs(sum(x[1] for x in ex) - float(lp[i, 0] - lp[i, 1])) < 1e-3
        assert abs(sum(x[2] for x in ex) - float(lp[i, 2] - lp[i, 1])) < 1e-3


def test_transE_learns_translation_vs_random_init():
    """Trained facts should be closer than untrained ones: d(LLC,hasLatVel,left) < d(LLC,hasLatVel,strongRight)."""
    kg = _train_small(); d = lambda t: float(kg.dist(torch.tensor(0), torch.tensor(0), torch.tensor(t)))
    assert d(3 + 3) < d(3 + 0)           # 'left' closer than 'strongRight' to (LLC + hasLatVel)


def test_core_claim_f1_high_near_crossing_and_decays_with_horizon():
    kg = _train_small(); rng = np.random.default_rng(5); f = {}
    for h in (1.0, 4.0):
        s = make_split(1200, rng, fixed_h=h)
        f[h] = macro_f1(kg.log_posterior(torch.from_numpy(s.ents)).argmax(-1), torch.from_numpy(s.label))
    assert f[1.0] > 0.75 and f[1.0] > f[4.0] + 0.1 and f[4.0] > 0.34   # above chance at 4 s


def test_closed_loop_predictive_beats_reactive_and_no_false_brake():
    from kglc.decision import run_cutin, lc_log_prior, false_alarm_rate
    kg = _train_small(); cfg = dict(dt=0.05, ego_speed=28.0, cutter_speed=18.0, crossing_gap_m=7.0, reaction_delay_s=0.4,
                                    max_decel=9.0, comfort_decel=3.5, trigger_prob=0.6, trigger_frames=6, prior_lane_change=0.05, n_monte_carlo=12)
    lp = lc_log_prior(cfg)
    for seed in range(4):
        re_, pr = run_cutin(kg, "reactive", cfg, seed, lp), run_cutin(kg, "predictive", cfg, seed, lp)
        assert re_.collided and not pr.collided
        assert pr.stats()["peak_decel_before_impact"] < re_.stats()["peak_decel_before_impact"]
    assert false_alarm_rate(kg, cfg, lp, n=12)["false_brake_rate"] <= 0.25
