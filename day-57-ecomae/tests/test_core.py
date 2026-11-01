import torch, numpy as np
from ecomae.model import eCoMAE, block_mask, pretrain_loss, nig_nll
from ecomae.synthetic import make_dataset, L, F
from ecomae.metrics import ece, aurc

D = make_dataset(16, ["clear", "fog"], 0)


def test_shapes_and_nig_constraints():
    m = eCoMAE(); vis = D["valid"]
    o = m.decode(m.encode(D["X"], vis), vis, D["valid"])
    assert o["rec"].shape == (16, L, F) and o["mu"].shape == (16, L, F)
    assert (o["nu"] > 0).all() and (o["alpha"] > 1).all() and (o["beta"] > 0).all()
    v, a, e = m.variances(o); assert torch.allclose(v, a + e, rtol=1e-4) and (v > 0).all()


def test_block_mask_valid_only_and_rate():
    mk = block_mask(D["valid"], 0.6, 3)
    assert not (mk & ~D["valid"]).any()
    frac = (mk.sum(1).float() / D["valid"].sum(1).float()); assert (frac - 0.6).abs().max() < 0.1


def test_masked_tokens_do_not_reach_encoder():
    m = eCoMAE().eval(); vis = D["valid"] & (torch.rand(16, L) > 0.5)
    x2 = D["X"].clone(); x2[~vis] = torch.randn_like(x2[~vis]) * 10        # corrupt hidden steps
    with torch.no_grad(): a, b = m.encode(D["X"], vis), m.encode(x2, vis)
    assert torch.allclose(a, b, atol=1e-5)


def test_pretrain_loss_backward_and_reference_no_grad():
    m = eCoMAE(); loss, t = pretrain_loss(m, D["X"], D["valid"]); loss.backward()
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in t.values())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_timae_mode_has_no_evidential_terms():
    m = eCoMAE(); _, t = pretrain_loss(m, D["X"], D["valid"], use_evi=False); assert t["evi"] == 0 and t["rep"] == 0


def test_nig_nll_minimised_near_truth():
    y = torch.zeros(1); lo = nig_nll(y, torch.zeros(1), torch.ones(1) * 2, torch.ones(1) * 2, torch.ones(1) * .1)
    hi = nig_nll(y, torch.ones(1) * 3, torch.ones(1) * 2, torch.ones(1) * 2, torch.ones(1) * .1); assert lo < hi


def test_padding_invariance_of_classifier():
    m = eCoMAE().eval(); x, v = D["X"][:4].clone(), D["valid"][:4].clone()
    v[:, :5] = False; x2 = x.clone(); x2[:, :5] = 7.0
    with torch.no_grad(): assert torch.allclose(m.classify(x, v), m.classify(x2, v), atol=1e-5)


def test_metrics_sane():
    p = np.eye(4)[[0, 1, 2, 3]]; y = np.array([0, 1, 2, 3]); assert ece(p, y) == 0 and aurc(p, y) == 0
