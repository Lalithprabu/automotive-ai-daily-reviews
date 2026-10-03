import torch, pytest
from momworld import MomWorld, MomWorldConfig, make_dataset
from momworld.data import H, T_HIST, F_IN
from momworld.metrics import collision, stale_momentum_extrapolation, ade_fde

@pytest.fixture(scope="module")
def batch():
    return make_dataset(32, 7)

def test_shapes_and_finite(batch):
    m = MomWorld(); o = m(batch.hist)
    assert o["plan"].shape == (32, H, 2) and o["base"].shape == (32, H, 2) and o["scene"].shape == (32, H, 2)
    assert o["z"].shape == o["m"].shape == o["keep"].shape == (32, H, 64)
    assert all(torch.isfinite(v).all() for v in o.values())

def test_expert_never_collides():
    d = make_dataset(2000, 11)
    assert not collision(d.future, d.lead_future).any()

def test_dataset_has_events_and_free_episodes():
    d = make_dataset(1000, 3)
    assert 0.2 < d.event.float().mean() < 0.6

def test_stale_momentum_fails_on_events_not_free():
    d = make_dataset(800, 5); p = stale_momentum_extrapolation(d.hist)
    ade, _ = ade_fde(p, d.future)
    assert ade[d.event].mean() > 3 * ade[~d.event].mean()      # trend persistence is wrong exactly when the scene changes
    assert collision(p, d.lead_future)[d.event].float().mean() > 0.4

def test_gradients_reach_history_encoder_and_momentum_extractor(batch):
    m = MomWorld(); loss, _ = m.compute_loss(batch.hist, batch.future, batch.lead_future); loss.backward()
    assert m.enc.gru.weight_ih_l0.grad.abs().sum() > 0
    assert m.world.extract.proj.weight.grad.abs().sum() > 0
    assert m.world.gate.weight.grad.abs().sum() > 0
    assert m.moflow.phi.grad.abs().sum() > 0                    # fusion weights are actually trained (Day-28 bug regression)

def test_reset_gate_zero_suppresses_momentum(batch):
    m = MomWorld(); m.eval()
    with torch.no_grad():
        m.world.gate.weight.zero_(); m.world.gate.bias.fill_(-30.0)   # keep ~ 0 -> m_k = g_k only
        o = m(batch.hist)
        assert o["keep"].max() < 1e-6
        # with keep=0 the initial history momentum m0 cannot influence the rollout
        h = m.enc(batch.hist); z1, _, _ = m.world(h)
        m.world.extract.proj.weight.mul_(5.0)
        z2, _, _ = m.world(h)
        assert torch.allclose(z1, z2, atol=1e-5)

def test_momentum_zero_gate_one_persists_initial_momentum(batch):
    m = MomWorld(); m.eval()
    with torch.no_grad():
        m.world.gate.weight.zero_(); m.world.gate.bias.fill_(30.0)
        h = m.enc(batch.hist); z1, _, _ = m.world(h)
        m.world.extract.proj.weight.mul_(5.0); z2, _, _ = m.world(h)
        assert not torch.allclose(z1, z2, atol=1e-4)

def test_old_way_has_no_momentum_path(batch):
    m = MomWorld(MomWorldConfig(use_momentum=False)); o = m(batch.hist)
    assert o["m"].abs().sum() == 0

def test_moflow_changes_plan_and_fusion_in_unit_interval(batch):
    m = MomWorld(); m.eval()
    with torch.no_grad():
        o = m(batch.hist)
    assert not torch.allclose(o["plan"], o["base"])
    w = m.moflow.fusion_weights(); assert ((w > 0) & (w < 1)).all()

def test_overfit_tiny_batch():
    torch.manual_seed(0); d = make_dataset(16, 1); m = MomWorld()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for i in range(600):
        loss, _ = m.compute_loss(d.hist, d.future, d.lead_future); opt.zero_grad(); loss.backward(); opt.step()
        first = first or loss.item()
    m.eval()
    with torch.no_grad(): ade, _ = ade_fde(m(d.hist)["plan"], d.future)
    assert loss.item() < 0.3 * first and ade.mean() < 1.5
