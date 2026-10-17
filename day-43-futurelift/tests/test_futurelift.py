import torch, numpy as np, pytest
from futurelift.data import JunctionDataset, make_scene, T_F, T_H, N_MAP
from futurelift.backbone import MapBlindBackbone
from futurelift.model import FutureLift
from futurelift.losses import futurelift_loss
from futurelift.metrics import forecast_metrics

torch.manual_seed(0)
DS = JunctionDataset(64, 7)
B = DS.batch(torch.arange(16))


def props(K=6):
    g = torch.Generator().manual_seed(1)
    base = B["gt"][:, None] + torch.randn(16, K, T_F, 2, generator=g) * 1.5
    return base, torch.randn(16, K, generator=g)


def test_shapes_and_scale_positive():
    m = FutureLift(32, 4, 1, 3); P, L = props()
    r, s, lg = m(B, P, L)
    assert r.shape == (16, 6, T_F, 2) and s.shape == r.shape and lg.shape == (16, 6)
    assert (s >= 0.1).all()                                   # ELU+1+0.1 floor


def test_identity_at_init():
    m = FutureLift(32, 4, 1, 3); P, L = props()
    r, _, lg = m(B, P, L)
    assert torch.allclose(r, P, atol=1e-5) and torch.allclose(lg, L, atol=1e-5)


def test_scene_geometry():
    sc = make_scene(np.random.default_rng(0))
    assert sc["map"].shape == (N_MAP, 4) and sc["gt"].shape == (T_F, 2) and sc["hist"].shape == (4, T_H, 4)
    assert np.isfinite(sc["gt"]).all()


def _randomize(m):
    for p in m.head[-1].parameters(): torch.nn.init.normal_(p, std=0.05)


@pytest.mark.parametrize("mode,early_changes", [("bidirectional", True), ("causal", False), ("none", False)])
def test_temporal_information_flow(mode, early_changes):
    """Perturb ONLY the last interval of the proposals. Does the first interval's refinement change?"""
    m = FutureLift(32, 4, 2, 3, mode).eval(); _randomize(m); P, L = props()
    with torch.no_grad():
        r0 = m(B, P, L)[0] - P
        P2 = P.clone(); P2[:, :, -4:] += 3.0                  # last interval (R=3 -> 4 steps each)
        r1 = m(B, P2, L)[0] - P2
    early = (r0[:, :, :4] - r1[:, :, :4]).abs().max().item()
    assert (early > 1e-5) == early_changes, (mode, early)


def test_gradients_flow_and_stop_at_proposals():
    m = FutureLift(32, 4, 1, 3); _randomize(m); P, L = props()
    P.requires_grad_(True); L.requires_grad_(True)
    r, s, lg = m(B, P, L); loss, _, _ = futurelift_loss(r, s, lg, B["gt"]); loss.backward()
    assert P.grad is None and L.grad is None                  # stop-gradient at proposals
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.head.parameters())
    assert m.enc.map[0].weight.grad.abs().sum() > 0           # scene encoder trained jointly


def test_masking_absent_agents_is_finite():
    b = {k: v.clone() for k, v in B.items()}; b["mask"][:, 1:] = False
    m = FutureLift(32, 4, 1, 3); _randomize(m); P, L = props()
    assert torch.isfinite(m(b, P, L)[0]).all()


def test_overfit_tiny_batch_beats_proposals():
    m = FutureLift(32, 4, 2, 3); P, L = props(); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    base = forecast_metrics(P, L, B["gt"])["minADE"]
    for _ in range(500):
        r, s, lg = m(B, P, L); loss, _, _ = futurelift_loss(r, s, lg, B["gt"])
        opt.zero_grad(); loss.backward(); opt.step()
    assert forecast_metrics(r.detach(), lg.detach(), B["gt"])["minADE"] < 0.5 * base


def test_backbone_is_map_blind():
    bb = MapBlindBackbone(); b2 = {k: v.clone() for k, v in B.items()}; b2["map"] = b2["map"] * 0 + 1
    assert torch.equal(bb(B)[0], bb(b2)[0])


def test_metrics_perfect():
    m = forecast_metrics(B["gt"][:, None].repeat(1, 6, 1, 1), torch.zeros(16, 6), B["gt"])
    assert m["minADE"] < 1e-6 and m["MR"] == 0
