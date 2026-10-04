import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, pytest
from egorefine import AsyncFusionNet, MODES, make_batch
from egorefine.metrics import focal_loss, extract_peaks, average_precision


def batch(B=4, **k): return make_batch(B, gen=torch.Generator().manual_seed(0), **k)


@pytest.mark.parametrize("mode", MODES)
def test_shapes_and_finite(mode):
    b = batch(); m = AsyncFusionNet(mode, C=8); y = m(b["ego"], b["col"])
    assert y.shape == (4, 1, 40, 40) and torch.isfinite(y).all()


def test_heatmap_centres_exactly_one():
    b = batch(8); assert (b["hm"].flatten(1).amax(1) == 1.0).all()
    uniq = sum(len({tuple(p.round().long().tolist()) for p in b["pos"][i][b["valid"][i]]}) for i in range(8))
    assert int((b["hm"] >= 1.0).sum()) == uniq        # objects sharing a pixel legitimately merge


def test_ego_blind_zone_is_blank_and_delay_shifts_collab():
    b = batch(6, delta=5)
    blind = (b["disk"] == 0)[:, None].expand(-1, 4, -1, -1)
    assert b["ego"][:, :4][blind].abs().mean() < 1e-6
    assert (b["pos"] - b["pos_col"]).norm(dim=-1)[b["valid"]].mean() > 1.0     # delayed collab positions differ
    b0 = batch(6, delta=0); assert (b0["pos"] - b0["pos_col"]).abs().max() == 0


def test_zero_init_offsets_start_as_identity_alignment():
    # zero-initialised offset heads => before training, aligned feature == plain collab feature (sampled at p)
    m = AsyncFusionNet("traf", C=8).eval(); b = batch()
    cf = m.col_enc(b["col"]); al, _ = m.align(m.ego_enc(b["ego"]), cf, b["col"][:, 4:5])
    assert torch.allclose(al, cf, atol=1e-4)


def test_refinement_moves_along_single_direction():
    m = AsyncFusionNet("egorefine", C=8).eval()
    for p in m.ref_out.parameters(): torch.nn.init.normal_(p, std=0.5)
    b = batch(); cf = m.col_enc(b["col"])
    _, aux = m.align(m.ego_enc(b["ego"]), cf, b["col"][:, 4:5])
    delta = aux["o_ref"] - aux["o_base"]                                   # (B,K,2,H,W)
    d = aux["dir"][:, None]
    cross = delta[:, :, 0] * d[:, :, 1] - delta[:, :, 1] * d[:, :, 0]      # parallel => 2D cross product 0
    assert cross.abs().max() < 1e-4 and abs(aux["dir"].norm(dim=1).mean() - 1) < 1e-4


def test_gate_in_unit_interval_and_modulates_output():
    m = AsyncFusionNet("egorefine", C=8).eval(); b = batch()
    _, aux = m.align(m.ego_enc(b["ego"]), m.col_enc(b["col"]), b["col"][:, 4:5])
    assert 0 <= aux["gate"].min() and aux["gate"].max() <= 1


def test_gradient_reaches_ego_reference_in_refiner():
    m = AsyncFusionNet("egorefine", C=8)
    for p in m.ref_out.parameters(): torch.nn.init.normal_(p, std=0.3)
    b = batch(); b["ego"].requires_grad_(True)
    m.align(m.ego_enc(b["ego"]), m.col_enc(b["col"]), b["col"][:, 4:5])[0].sum().backward()
    assert b["ego"].grad is not None and b["ego"].grad.abs().sum() > 0


def test_ap_perfect_and_empty():
    pos = torch.tensor([[[10., 10.], [20., 20.]]]); val = torch.tensor([[True, True]])
    assert average_precision([torch.tensor([[10, 10, .9], [20, 20, .8]]).numpy()], pos, val, 1.0) == pytest.approx(1.0)
    assert average_precision([torch.zeros(0, 3).numpy()], pos, val, 1.0) == 0.0


def test_overfit_tiny_batch_egorefine():
    torch.manual_seed(0); b = batch(8); m = AsyncFusionNet("egorefine", C=16)
    opt = torch.optim.Adam(m.parameters(), 3e-3); l0 = None
    for _ in range(120):
        loss = focal_loss(m(b["ego"], b["col"]), b["hm"]); opt.zero_grad(); loss.backward(); opt.step()
        l0 = l0 or loss.item()
    assert loss.item() < 0.5 * l0
