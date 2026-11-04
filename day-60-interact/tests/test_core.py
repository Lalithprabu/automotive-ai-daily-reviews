import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, torch
from interact.synthetic import World, oracle_costs, NA, IMG
from interact.model import WorldModel, SlotPlanner, DensePlanner, candidate_futures


def test_world_deterministic_and_render():
    a, b = World(np.random.default_rng(3)), World(np.random.default_rng(3))
    for m in (2, 0, 3, 4): a.step(m); b.step(m)
    assert np.allclose(a.x, b.x) and a.xe == b.xe
    im = a.render(); assert im.shape == (3, IMG, IMG) and im.dtype == np.uint8 and im[2].max() == 255


def test_clone_does_not_alias_and_oracle_shape():
    w = World(np.random.default_rng(1)); c = oracle_costs(w); assert c.shape == (NA,)
    w2 = w.clone(); w2.step(3); assert w.xe == 0.0 and w2.xe > 0


def test_ego_action_changes_traffic():
    """A car in the adjacent lane cuts in only if the ego does not accelerate (the interaction the model must learn)."""
    found = 0
    for s in range(60):
        w = World(np.random.default_rng(s)); v = w.clone()
        for _ in range(4): w.step(2); v.step(3)
        if (w.merging != v.merging).any() or not np.allclose(w.y, v.y): found += 1
    assert found > 0


def test_shapes_and_slot_permutation_equivariance():
    torch.manual_seed(0); wm = WorldModel().eval(); x = torch.rand(2, 3, 3, IMG, IMG)
    S = wm.enc(x); assert S.shape == (2, 3, 7, 64)
    img, masks = wm.dec(S[:, 0]); assert img.shape == (2, 3, IMG, IMG) and torch.allclose(masks.sum(1), torch.ones(2, IMG, IMG), atol=1e-4)
    code, probs, y = wm.lat(S[:, 0], S[:, 1]); assert code.shape == (2, 64) and torch.allclose(y.sum(-1), torch.ones(2))
    perm = torch.randperm(7)
    p1 = wm.pred(S[:, 0], code)[:, perm]; p2 = wm.pred(S[:, 0][:, perm], code)
    assert torch.allclose(p1, p2, atol=1e-4)                 # predictor treats slots as a set


def test_planner_shapes_and_future_rollout():
    wm = WorldModel().eval(); S = torch.randn(3, 7, 64); aemb = torch.randn(NA, 64)
    fut = candidate_futures(wm, S, aemb, 3); assert fut.shape == (3, NA, 3, 7, 64)
    assert SlotPlanner(use_future=True)(S, fut[:, :, [0, 2]], aemb).shape == (3, NA)
    assert SlotPlanner(use_future=False)(S, None, aemb).shape == (3, NA)
    assert DensePlanner()(torch.rand(3, 3, IMG, IMG), torch.rand(3, 3, IMG, IMG)).shape == (3, NA)


def test_action_conditioning_matters():
    """Different action embeddings must give different predicted futures (otherwise 'action-conditioned' would be a lie)."""
    torch.manual_seed(0); wm = WorldModel().eval(); S = torch.randn(1, 7, 64)
    f = candidate_futures(wm, S, torch.randn(NA, 64), 2)
    assert (f[0, 0] - f[0, 1]).abs().max() > 1e-4


def test_recurrent_encoder_uses_history():
    torch.manual_seed(0); wm = WorldModel().eval(); a, b = torch.rand(1, 3, IMG, IMG), torch.rand(1, 3, IMG, IMG)
    s_ab = wm.enc.step(b, wm.enc.step(a)); s_b = wm.enc.step(b)
    assert (s_ab - s_b).abs().max() > 1e-4
