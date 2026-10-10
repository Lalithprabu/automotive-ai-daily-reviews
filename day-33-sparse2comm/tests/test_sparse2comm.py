import torch, pytest
from sparse2comm import Sparse2comm, ModelCfg, DataCfg, sample_scene, make_batch
from sparse2comm.model import warp, SelfCalibratingFusion
from sparse2comm.losses import focal_loss, total_loss
from sparse2comm.metrics import extract_peaks, average_precision

CFG = DataCfg()


def batch(B=8, seed=0, **kw):
    g = torch.Generator().manual_seed(seed)
    return make_batch(sample_scene(B, CFG, g), CFG, g, **kw)


def test_batch_shapes_and_centre_forced_to_one():
    b = batch()
    assert b["sparse"].shape == (8, 2, 8, 32, 32) and b["heat"].shape == (8, 1, 32, 32)
    assert int((b["heat"] >= 0.999).sum()) >= int(b["valid"].sum()) * 0.8   # centres collide only rarely


def test_message_is_sparse_and_respects_keep_ratio():
    lo = batch(B=64, keep_ratio=0.15, loss_p=0.0)["bandwidth"].mean()
    hi = batch(B=64, keep_ratio=0.6, loss_p=0.0)["bandwidth"].mean()
    assert hi > 2.5 * lo and hi < 0.2    # a message is a few % of the cells, never dense


def test_ego_cannot_see_occluded_vehicles():
    b = batch(B=32)
    assert (b["ego_vis"] < b["valid"]).any()


def test_warp_zero_flow_is_identity_and_integer_shift_moves_peak():
    f = torch.zeros(1, 1, 16, 16); f[0, 0, 5, 7] = 1
    assert torch.allclose(warp(f, torch.zeros(1, 2, 16, 16)), f, atol=1e-6)
    out = warp(f, torch.tensor([2.0, 3.0]).view(1, 2, 1, 1).expand(1, 2, 16, 16))   # flow (dx=2, dy=3)
    idx = torch.nonzero(out[0, 0] > 0.99)[0]
    assert tuple(idx.tolist()) == (8, 9)    # (y=5+3, x=7+2)


def test_decoder_keeps_received_cells_verbatim():
    m = Sparse2comm(ModelCfg()); b = batch()
    out = m(b); r = out["recon"]; mk = b["mask"][:, 1]
    assert torch.allclose(r * mk, b["sparse"][:, 1] * mk, atol=1e-6)
    assert not torch.allclose(r * (1 - mk), torch.zeros_like(r))   # missing cells actually filled in


def test_calibration_recovers_known_offset_without_labels():
    torch.manual_seed(0)
    fus = SelfCalibratingFusion(8, 24, ModelCfg())
    ego = torch.zeros(1, 8, 32, 32)
    sig = torch.randn(8); sig = 1.5 * sig / sig.norm()
    ys, xs = torch.meshgrid(torch.arange(32.), torch.arange(32.), indexing="ij")
    def blob(cx, cy): return sig[:, None, None] * torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 1.2 ** 2))
    pts = [(8, 8), (20, 10), (14, 22), (24, 24)]
    ego = sum(blob(x, y) for x, y in pts)[None]
    o = torch.tensor([1.0, -1.5])
    collab = sum(blob(x + o[0], y + o[1]) for x, y in pts)[None]
    est = fus.estimate_offset(ego, collab)[0]
    assert (est - o).norm() < 0.35, est


def test_calibration_prior_gives_zero_with_no_evidence():
    fus = SelfCalibratingFusion(8, 24, ModelCfg())
    est = fus.estimate_offset(torch.zeros(2, 8, 32, 32), torch.randn(2, 8, 32, 32))
    assert est.abs().max() < 1e-4


def test_forward_shapes_all_variants():
    b = batch()
    for kw in (dict(), dict(use_recon=False, use_flow=False, use_calib=False), dict(cooperate=False)):
        out = Sparse2comm(ModelCfg(**kw))(b)
        assert out["logits"].shape == (8, 1, 32, 32)


def test_toggling_modules_changes_output():
    torch.manual_seed(0); b = batch()
    m = Sparse2comm(ModelCfg()).eval()
    a = m(b)["logits"]
    m.cfg.use_flow = False; f = m(b)["logits"]
    m.cfg.use_flow = True; m.cfg.use_recon = False; r = m(b)["logits"]
    assert not torch.allclose(a, f) and not torch.allclose(a, r)


def test_gradients_reach_every_module():
    b = batch(); m = Sparse2comm(ModelCfg())
    out = m(b); loss, _ = total_loss(out, b, dict(lambda_recon=2, lambda_flow=1, lambda_calib=.5)); loss.backward()
    for name in ("decoder", "align", "fusion", "ego_enc", "head"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(m, name).parameters()), name


def test_focal_loss_positive_mask_not_empty():
    b = batch()
    assert (b["heat"] >= 0.999).sum() > 0
    assert focal_loss(torch.zeros_like(b["heat"]), b["heat"]) > 0


def test_ap_perfect_and_empty():
    pos = torch.tensor([[[5.0, 5.0], [10.0, 12.0]]]); val = torch.ones(1, 2)
    perfect = [torch.tensor([[5.0, 5.0, 0.9], [10.0, 12.0, 0.8]])]
    assert average_precision(perfect, pos, val, 1.0) == pytest.approx(1.0)
    assert average_precision([torch.zeros(0, 3)], pos, val, 1.0) == 0.0


def test_overfit_tiny_batch():
    torch.manual_seed(0); b = batch(B=4, seed=3, delay=1, loss_p=0.1)
    m = Sparse2comm(ModelCfg()); opt = torch.optim.Adam(m.parameters(), 3e-3)
    t = dict(lambda_recon=2, lambda_flow=1, lambda_calib=.5)
    l0 = total_loss(m(b), b, t)[0].item()
    for _ in range(120):
        out = m(b); l, _ = total_loss(out, b, t); opt.zero_grad(); l.backward(); opt.step()
    assert l.item() < 0.6 * l0
