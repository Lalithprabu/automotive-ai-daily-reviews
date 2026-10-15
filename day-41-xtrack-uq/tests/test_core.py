import sys, math, torch; sys.path.insert(0, "src")
from xtrack_uq.data import make_dataset, to_tensors
from xtrack_uq.model import XTrackUQ, KinematicBicycle, GraphAttention, LockedDropout, loss_fn
from xtrack_uq.uncertainty import propagate_samples, ConformalCalibrator, mahalanobis
CFG = dict(dt=0.2, t_obs=15, t_pred=25, n_neighbors=4)
def batch(n=16): return to_tensors(make_dataset(n, CFG, 1))

def test_shapes():
    b = batch(); m = XTrackUQ(32); o = m(b, 8)
    assert o["mu"].shape == (16, 25, 2) and o["samples"].shape == (16, 8, 25, 2) and o["mean_traj"].shape == (16, 25, 2)

def test_kinematic_straight_line():
    k = KinematicBicycle(0.2); u = torch.zeros(1, 10, 2); p = k(u, torch.tensor([10.0]), torch.tensor([0.0]))
    assert torch.allclose(p[0, -1], torch.tensor([20.0, 0.0]), atol=1e-4)

def test_kinematic_matches_numpy_dataset():
    d = make_dataset(4, CFG, 3); b = to_tensors(d); k = KinematicBicycle(0.2)
    p = k(b["fut"], b["v0"], b["psi0"]); assert (p - b["pos"]).abs().max() < 1e-3

def test_gat_masks_non_neighbours():
    g = GraphAttention(16, 2); x = torch.randn(1, 3, 16); adj = torch.eye(3).unsqueeze(0)
    y1 = g(x, adj); x2 = x.clone(); x2[0, 2] += 5
    assert torch.allclose(g(x2, adj)[0, 0], y1[0, 0])          # node 0 unaffected by non-neighbour 2

def test_locked_dropout_same_mask_over_time():
    d = LockedDropout(0.5); d.train(); y = d(torch.ones(2, 6, 8))
    assert all(torch.equal(y[:, 0], y[:, t]) for t in range(6))

def test_mc_dropout_stochastic_only_when_on():
    m = XTrackUQ(32, p_drop=0.3); torch.nn.init.normal_(m.head.weight, std=0.3); m.eval(); b = batch(4)
    a1 = m.motion_dist(b["hist"], b["adj"])[0]; a2 = m.motion_dist(b["hist"], b["adj"])[0]
    assert torch.equal(a1, a2)
    m.set_mc(True); m2 = m.motion_dist(b["hist"], b["adj"])[0]
    assert not torch.equal(a1, m2)

def test_gradient_flows_through_kinematic_layer():
    m = XTrackUQ(32); b = batch(8); o = m(b, 4); l, _ = loss_fn(o, b); l.backward()
    assert m.head.bias.grad is not None and m.head.bias.grad.abs().sum() > 0 and torch.isfinite(l)

def test_overfit_tiny_batch():
    torch.manual_seed(0); m = XTrackUQ(48); b = batch(8); opt = torch.optim.Adam(m.parameters(), 3e-3)
    l0 = loss_fn(m(b, 8), b)[0].item()
    for _ in range(120):
        l, _ = loss_fn(m(b, 8), b); opt.zero_grad(); l.backward(); opt.step()
    assert l.item() < l0 - 1.0

def test_total_covariance_is_ale_plus_epi_and_psd():
    ms = [XTrackUQ(32) for _ in range(3)]; o = propagate_samples(ms, batch(6), 8)
    assert torch.allclose(o["cov_total"], o["cov_ale"] + o["cov_epi"])
    assert (torch.linalg.eigvalsh(o["cov_total"]) > -1e-5).all()

def test_epistemic_zero_for_single_deterministic_model():
    o = propagate_samples([XTrackUQ(32)], batch(4), 8, 1); assert o["cov_epi"].abs().max() == 0

def test_conformal_hits_target_coverage_on_exchangeable_data():
    torch.manual_seed(0); n = 4000; Tp = 5
    cov = torch.eye(2).expand(n, Tp, 2, 2) * 0.3                       # deliberately WRONG (overconfident) variance
    err = torch.randn(n, Tp, 2)                                          # true var = 1
    mean = torch.zeros(n, Tp, 2); pos = err
    cp = ConformalCalibrator(0.05).fit(pos[:2000], mean[:2000], cov[:2000])
    raw = (mahalanobis(pos[2000:], cov[2000:]) <= math.sqrt(-2 * math.log(0.05))).float().mean()
    post = cp.coverage(pos[2000:], mean[2000:], cov[2000:]).mean()
    assert raw < 0.8 and abs(post - 0.95) < 0.02
