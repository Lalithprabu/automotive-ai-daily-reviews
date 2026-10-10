import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, torch
from messenger import PairFlowEstimator, MemoryForecaster, rollout, knn_interp, make_sequence, SceneFlowDataset, collate


def _entries(B=2, N=40, L=3, seed=0, v=(1.0, 0.5)):
    g = torch.Generator().manual_seed(seed)
    P0 = torch.rand(B, N, 2, generator=g) * 20
    vel = torch.tensor(v)
    ents = [dict(P=P0 + s * vel, f=vel.expand(B, N, 2).clone(), h=torch.zeros(B, N, 48)) for s in range(L)]
    return P0 + L * vel, ents


def test_knn_interp_exact_on_constant_field():
    ref = torch.rand(1, 30, 2) * 10
    val = torch.ones(1, 30, 3) * 2.5
    out, d = knn_interp(torch.rand(1, 5, 2) * 10, ref, val)
    assert torch.allclose(out, torch.full_like(out, 2.5), atol=1e-5)


def test_dataset_shapes_and_labels():
    it = make_sequence(np.random.default_rng(0))
    assert it["pts"].shape == (6, 192, 2) and it["disp"].shape == (6, 192, 4, 2)
    bg = it["lab"] < 0
    assert np.abs(it["disp"][~bg]).max() > 0.5            # vehicles move
    assert np.abs(it["disp"][bg]).max() == 0.0            # background static
    assert (it["lab"][0] != it["lab"][1]).any()           # point order shuffled per frame


def test_pair_estimator_shapes_and_grad():
    m = PairFlowEstimator(16)
    a, b = torch.rand(2, 50, 2) * 20, torch.rand(2, 50, 2) * 20
    f, h = m(a, b)
    assert f.shape == (2, 50, 2) and h.shape == (2, 50, 16)
    f.sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_retrieval_chain_aligns_with_current_points():
    """With exact constant-velocity memory the age-a advanced entry must land on the current cloud."""
    P_cur, ents = _entries()
    m = MemoryForecaster(48, K=3)
    r = m.retrieve(P_cur, ents)
    for a in range(3):
        assert r[a]["mask"] == 1.0
        assert r[a]["dist"].max() < 1e-3, f"age {a+1} misaligned: {r[a]['dist'].max()}"
        assert torch.allclose(r[a]["f"], torch.tensor([1.0, 0.5]).expand_as(r[a]["f"]), atol=1e-3)


def test_forecaster_outputs_and_short_memory_mask():
    P_cur, ents = _entries(L=1)
    m = MemoryForecaster(48, K=3)
    flow, logsig, h, gates = m(P_cur, ents)
    assert flow.shape == (2, 40, 2) and logsig.shape == (2, 40) and h.shape == (2, 40, 48)
    assert gates.shape == (3, 2, 40) and gates[1:].abs().max() == 0   # ages 2,3 missing -> gate 0
    assert torch.isfinite(flow).all()


def test_permutation_equivariance_over_current_points():
    P_cur, ents = _entries()
    m = MemoryForecaster(48, K=3).eval()
    perm = torch.randperm(40)
    f1 = m(P_cur, ents)[0]
    f2 = m(P_cur[:, perm], ents)[0]
    assert torch.allclose(f1[:, perm], f2, atol=1e-4)


def test_older_memory_actually_influences_output():
    """Core claim of the design: the age-3 entry must reach the output (gradient + value change)."""
    P_cur, ents = _entries()
    m = MemoryForecaster(48, K=3).eval()
    base = m(P_cur, ents)[0]
    ents2 = [dict(e) for e in ents]
    ents2[0] = dict(ents[0], f=ents[0]["f"] + 3.0, h=ents[0]["h"] + 1.0)   # corrupt only the oldest
    assert (m(P_cur, ents2)[0] - base).abs().max() > 1e-4
    k1 = MemoryForecaster(48, K=1).eval()
    assert torch.allclose(k1(P_cur, ents)[0], k1(P_cur, ents2)[0], atol=1e-6)   # K=1 can't see it


def test_rollout_cumulative_and_entries_not_mutated():
    P_cur, ents = _entries()
    m = MemoryForecaster(48, K=3)
    n0 = len(ents)
    outs = rollout(m, P_cur, ents, 4)
    assert len(outs) == 4 and len(ents) == n0
    assert outs[3][0].shape == P_cur.shape


def test_overfit_tiny_batch_sanity():
    torch.manual_seed(0)
    ds = SceneFlowDataset(4, 0)
    b = collate([ds[i] for i in range(4)])
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from train import Pipeline
    p = Pipeline(24, 3, 6, True)
    opt = torch.optim.Adam(p.parameters(), 3e-3)
    def loss():
        fl, hs = p.observe(b["pts"])
        outs = rollout(p.fc, b["pts"][:, 5], p.entries_at(b["pts"], fl, hs, 5), 2)
        return sum((c - b["disp"][:, 5, :, h]).norm(dim=-1).mean() for h, (c, _, _) in enumerate(outs))
    l0 = loss().item()
    for _ in range(40):
        opt.zero_grad(); l = loss(); l.backward(); opt.step()
    assert l.item() < 0.7 * l0
