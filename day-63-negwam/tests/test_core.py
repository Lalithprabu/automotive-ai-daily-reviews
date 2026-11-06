import numpy as np, torch
from negwam.world import build_dataset, safety, expert_trajectory, sample_scenes, norm_scene, norm_traj, T
from negwam.model import NegWAM, N_RIDER
from negwam.losses import training_loss


def _batch(D, n=16):
    t = lambda a: torch.tensor(a[:n])
    return dict(scene=t(norm_scene(D["scene"])), expert=t(norm_traj(D["expert"])), neg_long=t(norm_traj(D["neg_long"])),
                neg_lat=t(norm_traj(D["neg_lat"])), lead=t(D["lead"] / 60.0))


def test_expert_is_safe_and_negatives_are_unsafe_but_near():
    D = build_dataset(400, 5)
    sc = D["scene"].astype(float)
    assert not safety(sc, D["expert"].astype(float))["collision"].any()
    s_lat = safety(sc, D["neg_lat"].astype(float)); assert s_lat["offlane"].mean() > 0.95
    s_lon = safety(sc, D["neg_long"].astype(float)); assert (s_lon["collision"] | ~s_lon["ttc_ok"] | s_lon["offlane"]).mean() > 0.99
    assert np.abs(D["neg_lat"] - D["expert"]).max() < 3.0          # hugs the expert


def test_shapes():
    m = NegWAM(d=32, layers=2); D = build_dataset(8, 1); b = _batch(D, 8)
    v, lead, logit = m(b["scene"], b["expert"], torch.rand(8), torch.stack([b["expert"], b["neg_long"], b["neg_lat"]], 1))
    assert v.shape == (8, T, 2) and lead.shape == (8, T) and logit.shape == (8, N_RIDER)
    assert m.sample(b["scene"]).shape == (8, T, 2)


def test_riders_do_not_change_inference_outputs():
    """Zero-overhead claim: nothing attends to rider tokens, so outputs are identical with/without them."""
    torch.manual_seed(0)
    m = NegWAM(d=32, layers=2).eval()
    for p in m.act_head.parameters(): torch.nn.init.normal_(p, std=0.1)        # head is zero-init; make it non-trivial
    D = build_dataset(8, 2); b = _batch(D, 8); t = torch.rand(8)
    r = torch.stack([b["expert"], b["neg_long"], b["neg_lat"]], 1)
    v1, l1, _ = m(b["scene"], b["expert"], t, None); v2, l2, _ = m(b["scene"], b["expert"], t, r)
    assert torch.allclose(v1, v2, atol=1e-5) and torch.allclose(l1, l2, atol=1e-5)


def test_world_tokens_cannot_see_actions():
    """World queries must not depend on the (noisy) action tokens."""
    torch.manual_seed(0)
    m = NegWAM(d=32, layers=2).eval(); D = build_dataset(4, 3); b = _batch(D, 4); t = torch.rand(4)
    _, l1, _ = m(b["scene"], torch.randn(4, T, 2), t); _, l2, _ = m(b["scene"], torch.randn(4, T, 2), t)
    assert torch.allclose(l1, l2, atol=1e-5)


def test_loss_backward_and_neg_terms():
    m = NegWAM(d=32, layers=2); b = _batch(build_dataset(32, 4), 32)
    loss0, p0 = training_loss(m, b, use_neg=False); loss1, p1 = training_loss(m, b, use_neg=True)
    loss1.backward()
    assert "rep" in p1 and "rider" in p1 and "rep" not in p0
    assert all(p.grad is not None for n, p in m.named_parameters() if n.startswith("rider"))


def test_repulsion_is_zero_when_prediction_is_perfect_and_positive_when_on_negative():
    """x1_hat == expert -> d_pos=0 < d_neg -> hinge inactive; x1_hat == neg -> hinge active."""
    import torch.nn.functional as F
    ex = torch.zeros(1, T, 2); neg = torch.full((1, T, 2), 0.05)
    d = lambda a, b: ((a - b) ** 2).mean((1, 2)).sqrt()
    assert F.relu(d(ex, ex) - d(ex, neg) + 0.02).item() == 0.0
    assert F.relu(d(neg, ex) - d(neg, neg) + 0.02).item() > 0.0
