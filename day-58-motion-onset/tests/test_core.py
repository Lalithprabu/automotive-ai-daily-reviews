import numpy as np, torch
from motion_onset.synthetic import episode, batch, N, T, H, D
from motion_onset.model import MotionOnsetGNN, GATv2, masked_bce
from motion_onset.metrics import bss, pr_auc, ece


def _b(n=2, seed=0): return {k: torch.as_tensor(v) for k, v in batch(np.random.default_rng(seed), n).items()}


def test_shapes_and_range():
    b = _b(); m = MotionOnsetGNN().eval(); p = m.rollout(b["pos"], b["vel"], b["vis"], b["sem"])
    assert p.shape == (2, T, N) and p.min() >= 0 and p.max() <= 1


def test_labels_consistent_with_onset():
    e = episode(np.random.default_rng(3)); t = np.arange(T)[:, None]
    assert not (e["y"] & ~(e["onset"][None] > t)).any()                       # positives are static now
    assert (e["y"] == ((e["onset"][None] > t) & (e["onset"][None] <= t + H))).all()
    assert not (e["q"] & ~e["vis"]).any()


def test_causality_no_future_leak():
    b = _b(); m = MotionOnsetGNN().eval(); p1 = m.rollout(b["pos"], b["vel"], b["vis"], b["sem"])
    pos2 = b["pos"].clone(); pos2[:, 30:] += 5.0                              # corrupt the future
    p2 = m.rollout(pos2, b["vel"], b["vis"], b["sem"]); assert torch.allclose(p1[:, :30], p2[:, :30], atol=1e-6)


def test_invisible_objects_have_no_influence():
    b = _b(1); m = MotionOnsetGNN().eval(); vis = b["vis"][:, 20].clone(); vis[0, 3] = False
    p1, _ = m(b["pos"][:, 20], b["vel"][:, 20], vis, b["sem"]); pos = b["pos"][:, 20].clone(); pos[0, 3] += 100; vel = b["vel"][:, 20].clone(); vel[0, 3] += 9
    p2, _ = m(pos, vel, vis, b["sem"]); keep = torch.ones(N, dtype=torch.bool); keep[3] = False
    assert torch.allclose(p1[0, keep], p2[0, keep], atol=1e-6)


def test_node_permutation_equivariance():
    b = _b(1); m = MotionOnsetGNN().eval(); perm = torch.randperm(N)
    p1 = m.rollout(b["pos"], b["vel"], b["vis"], b["sem"])
    p2 = m.rollout(b["pos"][:, :, perm], b["vel"][:, :, perm], b["vis"][:, :, perm], b["sem"][:, perm])
    assert torch.allclose(p1[:, :, perm], p2, atol=1e-4)


def test_memory_matters_only_when_recurrent():
    b = _b(1); mr = MotionOnsetGNN(recurrent=True).eval(); ms = MotionOnsetGNN(recurrent=False).eval()
    args = (b["pos"][:, 25], b["vel"][:, 25], b["vis"][:, 25], b["sem"])
    h = torch.randn(1, N + 1, 64)
    assert not torch.allclose(mr(*args, h)[0], mr(*args, None)[0], atol=1e-5)        # memory changes the output
    assert torch.allclose(ms(*args, h)[0], ms(*args, None)[0], atol=1e-6)            # stateless ignores it


def test_gat_rows_without_edges_are_safe():
    g = GATv2(64, 4); h = torch.randn(1, 3, 64); adj = torch.zeros(1, 3, 3, dtype=torch.bool); adj[0, 0, 0] = True
    out = g(h, torch.zeros(1, 3, 3, 5), adj); assert torch.isfinite(out).all()


def test_loss_ignores_non_query_and_metrics():
    p = torch.rand(2, 5, 4); y = (torch.rand(2, 5, 4) > .5).float(); q = torch.zeros(2, 5, 4); q[0, 0, 0] = 1
    l1 = masked_bce(p, y, q); p2 = p.clone(); p2[1] = 0.5; assert torch.isclose(l1, masked_bce(p2, y, q))
    yy = np.array([0, 0, 1, 1.]); pp = np.array([.1, .2, .8, .9])
    assert pr_auc(pp, yy) == 1.0 and abs(bss(np.full(4, .5), yy, .5)) < 1e-9 and ece(np.full(40, .5), np.tile([1., 0.], 20)) < 1e-9
