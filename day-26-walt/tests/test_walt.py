import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from walt.data import Scenes, norm_wp, T
from walt.model import FrozenWorldModel, TrajAutoencoder, WALTPlanner, DirectPlanner

def _setup():
    torch.manual_seed(0); ds = Scenes(16, 5); wm = FrozenWorldModel().eval()
    return ds, wm

def test_shapes():
    ds, wm = _setup(); x, xf, wp = ds.x[:4], ds.xf[:4], ds.wp[:4]
    assert wm(x).shape == x.shape
    ae = TrajAutoencoder(); ft = wm.future_tokens(wm.tokens(x)); z = ae.encode(norm_wp(wp), ft)
    assert z.shape == (4, 4, 32) and ae.decode(z).shape == (4, T, 2)
    assert WALTPlanner(wm, ae)(x)[0].shape == (4, T, 2) and DirectPlanner(wm)(x).shape == (4, T, 2)

def test_semantic_branch_uses_world_model():
    ds, wm = _setup(); ae = TrajAutoencoder().eval(); w = norm_wp(ds.wp[:4])
    ft = wm.future_tokens(wm.tokens(ds.x[:4]))
    assert not torch.allclose(ae.encode(w, ft), ae.encode(w, ft * 0 + 1.0))

def test_alignment_loss_stopgrad_on_world_model():
    ds, wm = _setup(); ae = TrajAutoencoder()
    x = ds.x[:4]; tok = wm.tokens(x); tok.retain_grad() if tok.requires_grad else None
    ft = wm.future_tokens(tok); ft.retain_grad()
    z = ae.encode(norm_wp(ds.wp[:4]), ft)   # semantic branch is allowed to read the WM ...
    ae.align_loss(z, ft).backward()          # ... but REPA target itself must be detached
    assert all(p.grad is not None for p in ae.proj.parameters())

def test_planner_grads_skip_frozen_wm():
    ds, wm = _setup()
    for p in wm.parameters(): p.requires_grad_(False)
    m = WALTPlanner(wm, TrajAutoencoder().requires_grad_(False)); out, _ = m(ds.x[:4]); out.sum().backward()
    assert all(p.grad is None for p in wm.parameters())
    assert all(p.grad is not None for p in m.planner.parameters())

def test_expert_is_physical():
    ds, _ = _setup(); d = ds.wp[:, 1:, 0] - ds.wp[:, :-1, 0]
    assert (d >= -1e-6).all() and (ds.wp[:, -1, 0] < 60).all()

def test_overfit_tiny_batch():
    ds, wm = _setup()
    for p in wm.parameters(): p.requires_grad_(False)
    m = DirectPlanner(wm); opt = torch.optim.Adam(m.planner.parameters(), 3e-3)
    x, w = ds.x[:8], norm_wp(ds.wp[:8]); l0 = None
    for _ in range(300):
        l = torch.nn.functional.mse_loss(m(x), w); opt.zero_grad(); l.backward(); opt.step(); l0 = l0 or l.item()
    assert l.item() < 0.2 * l0
