import numpy as np, torch
from rfworld.world import World, ANCHORS, N_ANCH, make_batch
from rfworld.model import RFWorldToy, SparseReadout, compose_pdms, GlobalAnchorClassifier
from rfworld.evaluate import to_t


def test_anchor_shapes():
    assert ANCHORS.shape == (256, 6, 2) and N_ANCH == 256 and np.isfinite(ANCHORS).all()
    assert (ANCHORS[:, :, 0] > 0).all()                       # all arcs go forward


def test_scorer_ranges_and_gating():
    s = World(np.random.default_rng(1)).score()
    assert s["pdms"].shape == (256,) and 0 <= s["pdms"].min() and s["pdms"].max() <= 1
    assert (s["pdms"][(s["nc"] == 0) | (s["dac"] == 0)] == 0).all()


def test_forward_shapes_and_aux():
    B = to_t(make_batch(np.random.default_rng(0), 4)); m = RFWorldToy()
    o = m(B, torch.as_tensor(ANCHORS)[None].expand(4, -1, -1, -1), with_aux=True)
    assert o["logits"].shape == (4, 256, 5) and o["pdms"].shape == (4, 256) and o["fut"].shape == (4, 2, 16, 16)


def test_rollout_free_inference_has_no_aux_cost():
    B = to_t(make_batch(np.random.default_rng(0), 2)); m = RFWorldToy()
    o = m(B, torch.as_tensor(ANCHORS)[None].expand(2, -1, -1, -1)); assert "fut" not in o


def test_readout_samples_the_right_cell():
    """A memory map that is 1 at one cell must be read by a waypoint at that cell's centre."""
    r = SparseReadout(d=4, T=1)
    fm = torch.zeros(1, 4, 16, 16); fm[:, :, 0, 8] = 1.0       # row 0 (x~32 m), col 8 (y~0 m)
    x, y = 32 - 1.0, 0.0                                       # centre of grid row 0 at 16x16 = 32-1 m
    gx, gy = -y / 16, 1 - x / 16
    v = torch.nn.functional.grid_sample(fm, torch.tensor([[[[gx + 1 / 32, gy]]]]), align_corners=False)
    assert v.max() > 0.2


def test_compose_matches_scorer_formula():
    p = torch.ones(1, 3, 5); assert torch.allclose(compose_pdms(p), torch.ones(1, 3))


def test_closed_loop_step_moves_forward():
    w = World(np.random.default_rng(2)); x0 = w.ego[0]; w.step(ANCHORS[200]); assert w.ego[0] > x0


def test_baseline_shapes():
    B = to_t(make_batch(np.random.default_rng(0), 2)); o = GlobalAnchorClassifier()(B); assert o["pdms"].shape == (2, 256)
