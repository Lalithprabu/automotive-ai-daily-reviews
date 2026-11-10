import torch

from comet import CoMETConfig, CoMETTracker, mutual_nn_pairs


def _msg(offset, n=4, seed=0, noise=0.0):
    g = torch.Generator().manual_seed(seed)
    ego = torch.rand(1, n, 2, generator=g) * 20 - 10
    road = ego + torch.tensor(offset).view(1, 1, 2) + noise * torch.randn(1, n, 2, generator=g)
    m = torch.ones(1, n, dtype=torch.bool)
    return ego, road, m, m.clone()


def test_no_learned_parameters():
    assert sum(p.numel() for p in CoMETTracker().parameters()) == 0


def test_recovers_constant_offset():
    tr = CoMETTracker()
    for s in range(30):
        out = tr.step(*_msg([1.2, -0.7], seed=s, noise=0.05))
    assert torch.allclose(out["offset"][0], torch.tensor([1.2, -0.7]), atol=0.1)


def test_correction_applies_to_unmatched_roadside_boxes():
    tr = CoMETTracker()
    ego, road, em, rm = _msg([1.0, 1.0], n=4)
    extra = torch.tensor([[[40.0, 40.0]]])                  # roadside-only box, ego never sees it
    road2 = torch.cat([road, extra + 1.0], 1)
    rm2 = torch.ones(1, 5, dtype=torch.bool)
    for _ in range(20):
        out = tr.step(ego, road2, em, rm2)
    assert torch.allclose(out["corrected_road_xy"][0, 4], torch.tensor([40.0, 40.0]), atol=0.15)


def test_carry_forward_when_no_pair():
    tr = CoMETTracker()
    for s in range(10):
        tr.step(*_msg([0.8, 0.0], seed=s))
    x_before, p_before = tr.x.clone(), tr.P.clone()
    ego = torch.zeros(1, 1, 2)
    road = torch.full((1, 1, 2), 90.0)                       # far away -> outside any gate
    m = torch.ones(1, 1, dtype=torch.bool)
    out = tr.step(ego, road, m, m.clone())
    assert int(out["n_pairs"][0]) == 0
    assert torch.equal(tr.x, x_before)                       # estimate carried forward
    assert tr.P > p_before                                   # ...with growing uncertainty


def test_mutual_nn_rejects_one_sided_pairs():
    ego = torch.tensor([[[0.0, 0.0], [0.4, 0.0]]])
    road = torch.tensor([[[0.5, 0.0]]])                      # nearest to ego[1]; ego[0] is not mutual
    ok = mutual_nn_pairs(ego, road, torch.ones(1, 2, dtype=torch.bool),
                         torch.ones(1, 1, dtype=torch.bool), torch.tensor([2.0]))
    assert ok[0, 1, 0] and not ok[0, 0, 0]


def test_empty_inputs_are_safe():
    tr = CoMETTracker()
    out = tr.step(torch.zeros(1, 0, 2), torch.zeros(1, 3, 2),
                  torch.zeros(1, 0, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool))
    assert torch.isfinite(out["offset"]).all() and int(out["n_pairs"][0]) == 0


def test_batched_senders_are_independent():
    tr2 = CoMETTracker(CoMETConfig(num_senders=2))
    a, b = _msg([1.0, 0.0], seed=1), _msg([-2.0, 1.0], seed=2)
    for _ in range(25):
        out = tr2.step(torch.cat([a[0], b[0]]), torch.cat([a[1], b[1]]),
                       torch.cat([a[2], b[2]]), torch.cat([a[3], b[3]]))
    assert torch.allclose(out["offset"][0], torch.tensor([1.0, 0.0]), atol=0.1)
    assert torch.allclose(out["offset"][1], torch.tensor([-2.0, 1.0]), atol=0.1)


def test_filter_beats_current_message_only_on_noisy_stream():
    import numpy as np
    errs = {}
    for use in (True, False):
        tr = CoMETTracker(CoMETConfig(use_filter=use))
        e = []
        for s in range(60):
            out = tr.step(*_msg([1.0, 1.0], n=3, seed=s, noise=0.4))
            e.append(float((out["offset"][0] - torch.tensor([1.0, 1.0])).norm()))
        errs[use] = np.mean(e[10:])
    assert errs[True] < errs[False]


def test_end_to_end_comet_beats_uncorrected_in_toy_world():
    from evaluate import run_sequence
    from comet.metrics import prf
    cnt, _, err, _ = run_sequence(2, 200.0)
    assert prf(*cnt["comet"])[2] > prf(*cnt["raw"])[2] + 0.2
    assert sum(err["comet"]) < sum(err["raw"])


def test_known_limitation_no_overlap_means_no_correction():
    """Seed 7 of the toy world has ~95% unmatched messages: CoMET cannot observe the offset,
    stays at its prior (0) and degrades gracefully to the uncorrected baseline -- it does not
    make things worse.  This documents the failure mode instead of hiding it."""
    from evaluate import run_sequence
    from comet.metrics import prf
    cnt, _, err, no_pair = run_sequence(7, 200.0)
    assert no_pair > 0.8
    assert prf(*cnt["comet"])[2] <= prf(*cnt["raw"])[2] + 0.05
