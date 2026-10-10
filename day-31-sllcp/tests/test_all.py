import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from sllcp.conformal import SplitLabelConditionalCP, SplitLabelLocalizedCP
from sllcp.data import make_dataset, simulate_collisions
from sllcp.models import SafetyScorer


def _trained(n=4000, steps=300):
    torch.manual_seed(0)
    d = make_dataset(n, [1 / 3] * 3, 11)
    m = SafetyScorer(); opt = torch.optim.Adam(m.parameters(), 2e-3)
    for _ in range(steps):
        idx = torch.randint(0, n, (128,))
        l = torch.nn.functional.cross_entropy(m(d["x"][idx]), d["y"][idx]); opt.zero_grad(); l.backward(); opt.step()
    return m


def test_simulator_is_ego_reactive_and_has_both_classes():
    d = make_dataset(5000, [1 / 3] * 3, 0)
    r = d["y"].float().mean().item()
    assert 0.05 < r < 0.4
    p = dict(d["params"]); p2 = {k: v.clone() for k, v in p.items()}
    p2["c"] = torch.full_like(p2["c"], 0.0)  # other agent never yields -> strictly >= collisions
    assert simulate_collisions(p2).sum() >= simulate_collisions(p).sum()


def test_model_output_shape_and_grad():
    m = SafetyScorer(); x = torch.randn(7, 10, requires_grad=True)
    out = m(x); assert out.shape == (7, 2)
    out.sum().backward(); assert x.grad is not None and x.grad.abs().sum() > 0


def test_localized_matches_global_with_huge_bandwidth():
    m = _trained(); cal = make_dataset(3000, [1 / 3] * 3, 5); te = make_dataset(1000, [1 / 3] * 3, 6)
    pc, pt = m.probs(cal["x"]), m.probs(te["x"])
    g = SplitLabelConditionalCP(0.1).fit(pc, cal["y"]).predict_sets(pt)
    l = SplitLabelLocalizedCP(0.1, bandwidth=1e4).fit(pc, cal["emb"], cal["y"]).predict_sets(pt, te["emb"])
    assert (g == l).float().mean() > 0.97  # flat kernel == unweighted conformal


def test_label_conditional_coverage_guarantee_exchangeable():
    m = _trained(); cal = make_dataset(6000, [1 / 3] * 3, 7); te = make_dataset(8000, [1 / 3] * 3, 8)
    pc, pt = m.probs(cal["x"]), m.probs(te["x"])
    for cp in (SplitLabelConditionalCP(0.1).fit(pc, cal["y"]),):
        S = cp.predict_sets(pt)
        for k in (0, 1):
            cov = S[te["y"] == k, k].float().mean().item()
            assert cov >= 0.90 - 0.03, (k, cov)
    S = SplitLabelLocalizedCP(0.1, 0.5).fit(pc, cal["emb"], cal["y"]).predict_sets(pt, te["emb"])
    for k in (0, 1):
        assert S[te["y"] == k, k].float().mean().item() >= 0.90 - 0.04


def test_core_claim_localization_fixes_shifted_regime_undercoverage():
    """Calibrate on mostly-clear scenes, test on occluded: global CP undercovers there, SLLCP doesn't (as much)."""
    m = _trained(steps=400)
    cal = make_dataset(8000, [0.6, 0.3, 0.1], 21); te = make_dataset(8000, [0.0, 0.0, 1.0], 22)
    pc, pt = m.probs(cal["x"]), m.probs(te["x"])
    gs = SplitLabelConditionalCP(0.1).fit(pc, cal["y"]).predict_sets(pt)[:, 1]
    ls = SplitLabelLocalizedCP(0.1, 0.5).fit(pc, cal["emb"], cal["y"]).predict_sets(pt, te["emb"])[:, 1]
    u = te["y"] == 1
    g_cov, l_cov = gs[u].float().mean().item(), ls[u].float().mean().item()
    assert l_cov >= g_cov - 1e-6
    assert l_cov >= 0.85


def test_quantile_inf_when_too_few_calibration_points():
    cp = SplitLabelLocalizedCP(0.1, 0.5)
    probs = torch.tensor([[0.9, 0.1]] * 3); y = torch.zeros(3, dtype=torch.long); emb = torch.zeros(3, 6)
    cp.fit(probs, emb, y)
    q = cp.quantiles(torch.zeros(2, 6))
    assert torch.isinf(q[:, 1]).all()  # no unsafe calibration data -> always include 'unsafe' (fail-safe)
