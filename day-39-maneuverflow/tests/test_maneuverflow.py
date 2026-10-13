import os, sys, pytest, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from maneuverflow.data import make_dataset
from maneuverflow.model import ManeuverFlow, NORM, M, TF
from maneuverflow.scoring import factors, pdms

@pytest.fixture(scope="module")
def ds():
    return make_dataset(64, 3)

def test_scoring_collision_and_offroad():
    traj = torch.stack([torch.arange(1, TF + 1).float() * 5, torch.zeros(TF)], -1)[None]
    ag = traj[:, None].clone() + torch.tensor([3.0, 0.5])           # box right on top of the path
    f = factors(traj, ag, torch.ones(1, 1))
    assert f["nc"].item() == 0 and pdms(f).item() == 0
    off = traj.clone(); off[..., 1] = 4.6
    assert factors(off, ag + 50, torch.ones(1, 1))["dac"].item() == 0

def test_shapes_and_candidate_count(ds):
    m = ManeuverFlow().eval()
    o = m.predict(ds["hist"][:5], ds["agents"][:5])
    assert o["cands"].shape == (5, 2 * 3 * 2, TF, 2)                  # retained + refined
    assert o["best"].shape == (5, TF, 2) and torch.isfinite(o["cands"]).all()
    o2 = m.predict(ds["hist"][:5], ds["agents"][:5], refine=False)
    assert o2["cands"].shape[1] == 6

def test_router_selects_distinct_maneuvers(ds):
    m = ManeuverFlow().eval()
    o = m.predict(ds["hist"][:4], ds["agents"][:4], refine=False)
    for b in range(4):
        assert len(set(o["maneuvers"][b].tolist())) == 3

def test_refiner_identity_at_init_and_bounded(ds):
    m = ManeuverFlow().eval()
    t = torch.randn(2, 3, TF, 2)
    assert torch.allclose(m.refiner(t, torch.randn(2, 3, 64)), t)
    for p in m.refiner.net[-1].parameters(): torch.nn.init.normal_(p, std=50.0)
    r = m.refiner(t, torch.randn(2, 3, 64))
    assert ((r - t).abs() <= torch.tensor([4.0, 1.0]) + 1e-5).all()

def test_future_evidence_changes_score(ds):
    m = ManeuverFlow().eval()
    for p in m.scorer.parameters(): torch.nn.init.normal_(p, std=0.2)
    cond, tok, valid, fpos, ftok = m.encode(ds["hist"][:3], ds["agents"][:3])
    c = ds["gt"][:3, None].repeat(1, 2, 1, 1)
    a, _ = m.scorer(c, cond, tok, valid, ftok, True)
    b, _ = m.scorer(c, cond, tok, valid, ftok, False)
    assert not torch.allclose(a, b)

def test_meanflow_loss_finite_and_grads_flow(ds):
    m = ManeuverFlow()
    cond, *_ = m.encode(ds["hist"][:16], ds["agents"][:16])
    loss = m.meanflow_loss((ds["gt"][:16] / NORM).flatten(1), cond, ds["maneuver"][:16])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in m.flow.experts[int(ds["maneuver"][0])].parameters())
    assert m.enc.ego[0].weight.grad is not None                       # gradient reaches the scene encoder

def test_one_step_sample_is_exact_for_straight_path_field(ds):
    """If the learned average velocity equals eps - x (straight conditional path), one step recovers x exactly."""
    m = ManeuverFlow().eval()
    x = (ds["gt"][:4] / NORM).flatten(1)
    mm = ds["maneuver"][:4]
    g = torch.Generator().manual_seed(0)
    eps = m.prior_sample(mm, torch.Generator().manual_seed(0))
    m.flow.forward = lambda z, r, t, c, mn: z - x                     # u = z1 - x  (with z1=eps -> x_hat = x)
    out = m._gen_one_step(mm, torch.zeros(4, 64), torch.Generator().manual_seed(0)) / NORM
    assert torch.allclose(out.flatten(1), x, atol=1e-5)

def test_overfit_tiny_batch_meanflow(ds):
    torch.manual_seed(0)
    m = ManeuverFlow()
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    h, a, g, mn = ds["hist"][:8], ds["agents"][:8], ds["gt"][:8], ds["maneuver"][:8]
    first = None
    for i in range(120):
        cond, *_ = m.encode(h, a)
        l = m.meanflow_loss((g / NORM).flatten(1), cond, mn)
        opt.zero_grad(); l.backward(); opt.step()
        first = first or l.item()
    assert l.item() < 0.5 * first

@pytest.mark.skipif(not os.path.exists("maneuverflow.pt"), reason="run train.py first")
def test_core_claim_full_beats_multiquery_on_safety():
    from train import evaluate
    from maneuverflow.baseline import MultiQueryPlanner
    ck, bk = torch.load("maneuverflow.pt"), torch.load("baseline.pt")
    cfg = ck["cfg"]
    mf = ManeuverFlow(cfg["d_model"], cfg["topk"], cfg["per_man"], ck["prior"], cfg["prior_sigma"]); mf.load_state_dict(ck["state"]); mf.eval()
    bl = MultiQueryPlanner(cfg["d_model"], cfg["topk"] * cfg["per_man"]); bl.load_state_dict(bk["state"]); bl.eval()
    d = make_dataset(300, 999)
    rm = evaluate(lambda h, a: mf.predict(h, a, gen=torch.Generator().manual_seed(1)), d)
    rb = evaluate(bl.predict, d)
    assert rm["nc"] > rb["nc"] + 0.1 and rm["pdms"] > rb["pdms"]
