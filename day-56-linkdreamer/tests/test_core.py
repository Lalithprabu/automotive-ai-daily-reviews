import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from linkdreamer.synthetic import G, CELL, XMIN, YMIN, EGO_CELLS, shift_rows, make_episode, visible, link_sequence
from linkdreamer.models import LatentWorldModel, TSNeuroFuzzy, FusionDetector, build_modes, decision_state, mode_mask, COMM
from linkdreamer.metrics import ap_f1, match, decode
from linkdreamer import agent


def test_shift_rows_moves_content_back_and_zero_fills():
    x = torch.zeros(1, 1, G, G); x[0, 0, 10, 5] = 1
    y = shift_rows(x, torch.tensor([4]))
    assert y[0, 0, 6, 5] == 1 and y.sum() == 1
    assert shift_rows(x, torch.tensor([G + 3])).sum() == 0          # shifted out of the window


def test_truck_shadow_hides_object_behind_it():
    occ = np.array([[10.0, 0.0, 2.0]])
    assert not visible(np.array(22.0), np.array(0.0), 0.0, 0.0, occ)           # directly behind the truck
    assert visible(np.array(22.0), np.array(12.0), 0.0, 0.0, occ)             # off to the side
    assert not visible(np.array(40.0), np.array(0.0), 0.0, 0.0, occ)           # beyond sensor range


def test_episode_shapes_and_targets():
    e = make_episode(np.random.default_rng(0), 6)
    assert e["ego"].shape == (6, 2, G, G) and e["collab"].shape == (6, 2, G, G) and e["tgt"].shape == (6, G, G)
    assert e["tgt"].max() <= 1.0 + 1e-6 and e["ego"][:, 1].min() >= 0 and e["ego"][:, 1].max() <= 1


def test_world_model_shapes_and_variable_age():
    wm = LatentWorldModel(); B = 3
    mean, lv = wm(torch.rand(B, 2, 2, G, G), torch.tensor([[1], [2], [3]]), torch.tensor([1, 4, 8]))
    assert mean.shape == (B, 2, G, G) and lv.shape == (B, 2, G, G)
    assert 0 <= mean.min() and mean.max() <= 1 and lv.min() >= -6 and lv.max() <= 2
    mean.sum().backward()                                                        # gradients flow through the rollout
    assert all(p.grad is not None for p in wm.parameters())


def test_neuro_fuzzy_firing_strengths_normalised_and_differentiable():
    c = TSNeuroFuzzy(); x = torch.rand(7, 5, requires_grad=True)
    z = (x[:, None, :] - c.c[None]) / c.log_s.exp()[None]
    assert torch.allclose(torch.softmax(-0.5 * (z ** 2).sum(-1), 1).sum(1), torch.ones(7), atol=1e-5)
    c(x).sum().backward()
    assert c.c.grad is not None and c.A.grad is not None and len(c.describe()) == 9


def test_mode_mask_respects_link_quality_and_memory():
    m = mode_mask(torch.tensor([0., .5, 1.]), torch.tensor([True, True, False]))
    assert m[:, 0].all()
    assert m[0].tolist() == [True, True, True, False, False]
    assert m[1].tolist() == [True, True, True, True, False]
    assert m[2].tolist() == [True, False, False, True, True]


def test_decision_state_in_unit_interval_and_modes_shapes():
    wm = LatentWorldModel(); B = 4
    col, ego = torch.rand(B, 2, G, G), torch.rand(B, 2, G, G)
    hist, gaps, ages, hm = torch.rand(B, 2, 2, G, G), torch.ones(B, 1, dtype=torch.long), torch.tensor([1, 2, 3, 5]), torch.ones(B, dtype=torch.bool)
    coll, sig, ws, reuse = build_modes(wm, col, hist, gaps, ages, hm)
    assert coll.shape == (B, 5, 2, G, G) and sig.shape == (B, 5, 1, G, G)
    assert torch.equal(coll[:, 4], col) and coll[:, 0].abs().sum() == 0
    s = decision_state(ego, reuse, ws, ages, hm, torch.ones(B), torch.full((B,), .5))
    assert s.shape == (B, 5) and s.min() >= 0 and s.max() <= 1
    logits = FusionDetector()(ego, coll[:, 2], sig[:, 2], torch.full((B,), 2))
    assert logits.shape == (B, 1, G, G)


def test_ap_perfect_and_empty():
    gt = [(10.0, 0.0), (20.0, 5.0)]
    r, miss = match([(10.2, 0.1, .9), (20.0, 5.0, .8)], gt)
    assert miss == 0 and ap_f1(r, 2) == (1.0, 1.0)
    assert ap_f1([], 2)[0] == 0.0


def test_runner_never_uses_unavailable_modes_or_overspends_budget():
    torch.manual_seed(0)
    ds = {"ego": torch.rand(2, 24, 2, G, G), "collab": torch.rand(2, 24, 2, G, G), "tgt": torch.zeros(2, 24, G, G), "objs": [[np.zeros((0, 2), np.float32)] * 24] * 2}
    q = link_sequence(np.random.default_rng(0), 2)
    o = agent.run(ds, [0, 1], q, "ctrl", LatentWorldModel().eval(), FusionDetector().eval(), TSNeuroFuzzy(), budget_frac=0.25)
    m = o["modes"].numpy()
    assert not ((q < 1) & (m == 4))[:, 2:].any()                 # high-rate refresh needs a fully up link
    assert not ((q < .5) & (m == 3))[:, 2:].any()
    assert o["comm"].sum(1).max() <= 0.25 * 24 + 2 * COMM[4] + 1e-4   # 2 forced warm-up frames + budget
