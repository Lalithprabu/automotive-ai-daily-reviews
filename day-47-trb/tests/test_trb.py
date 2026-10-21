import os, sys
import numpy as np, pytest, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from trb.model import AsyncCollabDetector, TRBConfig, TemporalResidualBottleneck
from trb.xlstm import XLSTMStack, MLSTMCell, SLSTMCell
from trb.data import make_batch, render, warp_history, HALF, GRID, C_FEAT, MAX_DELAY
from trb.warp import rigid_warp
from trb.losses import app_loss, det_loss
from trb.metrics import detection_stats


def test_rigid_warp_identity_and_static_object_consistency():
    pos = torch.tensor([[[5.0, -3.0]]]); cls = torch.zeros(1, 1, dtype=torch.long); m = torch.ones(1, 1)
    f = render(pos, cls, m)
    assert torch.allclose(rigid_warp(f, torch.zeros(1, 3), HALF), f, atol=1e-4)
    # collaborator moved +4m in x without rotating: a static object appears 4 m closer in x in the NEW frame
    new = render(pos - torch.tensor([4.0, 0.0]), cls, m)
    warped = rigid_warp(f, torch.tensor([[4.0, 0.0, 0.0]]), HALF)  # q_stale = q_now + (4,0)
    inside = (new.abs().sum(1) > 0.2)
    assert (warped - new).abs()[:, :, :].max() < 0.25


def test_batch_shapes_and_determinism():
    a, b = make_batch(3, 5), make_batch(3, 5)
    assert a["hist"].shape == (3, 4, C_FEAT, GRID, GRID) and a["heat"].shape == (3, 1, GRID, GRID)
    assert torch.equal(a["hist"], b["hist"]) and torch.equal(a["heat"], b["heat"])
    assert (a["heat"].flatten(1).max(1).values == 1.0).all()


def test_zero_delay_stale_equals_target_up_to_noise():
    b = make_batch(4, 3, delays=[0] * 4, collab_noise=0.0)
    assert (b["hist"][:, 0] - b["target"]).abs().max() < 1e-5


def test_xlstm_shapes_gradients_and_finite_under_long_sequence():
    st = XLSTMStack(32, "msm", 4); x = torch.randn(10, 12, 32, requires_grad=True) * 3
    x.retain_grad(); y = st(x); assert y.shape == (10, 32) and torch.isfinite(y).all()
    y.sum().backward(); assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_trb_zero_init_means_fhat_equals_warped_anchor():
    cfg = TRBConfig(); m = TemporalResidualBottleneck(cfg).eval(); b = make_batch(2, 1)
    with torch.no_grad():
        o = m(b["hist"], b["rel"], b["delay"])
    assert torch.allclose(o["F_hat"], o["B"], atol=1e-6)


def test_gate_floor_range_and_no_grad():
    m = TemporalResidualBottleneck(TRBConfig()); b = make_batch(2, 2)
    o = m(b["hist"], b["rel"], b["delay"]); G = o["G"]
    assert G.min() >= 0.05 - 1e-6 and G.max() <= 1.0 + 1e-6 and not G.requires_grad


def test_gradient_reaches_xlstm_and_delay_conditioning():
    det = AsyncCollabDetector("trb"); b = make_batch(2, 4)
    with torch.no_grad():  # un-zero the output so gradients can flow to upstream modules
        det.trb.bn_out.weight.normal_(0, 0.05)
    out = det(b); (out["logits"].mean() + out["F_hat"].pow(2).mean()).backward()
    assert det.trb.xlstm.cells[0].qkv.weight.grad.abs().sum() > 0
    assert det.trb.film.weight.grad.abs().sum() > 0


def test_delay_changes_output():
    det = AsyncCollabDetector("trb").eval(); b = make_batch(2, 4)
    with torch.no_grad():
        det.trb.bn_out.weight.normal_(0, 0.05)
        o1 = det.trb(b["hist"], b["rel"], torch.zeros(2)); o2 = det.trb(b["hist"], b["rel"], torch.full((2,), 5.0))
    assert (o1["F_hat"] - o2["F_hat"]).abs().max() > 1e-4


def test_app_loss_zero_for_perfect_residual_and_positive_for_wrong():
    B = torch.randn(2, 8, 8, 8); T = B + 0.5 * torch.randn_like(B)
    assert app_loss(B, T - B, T) < 1e-4
    assert app_loss(B, -(T - B), T) > 0.5


def test_variants_forward_all():
    b = make_batch(2, 7)
    for v in ["stale", "warp", "trb"]:
        assert AsyncCollabDetector(v)(b)["logits"].shape == (2, 1, GRID, GRID)


def test_overfit_tiny_batch_and_beats_nothing_baseline():
    torch.manual_seed(0); b = make_batch(4, 11); det = AsyncCollabDetector("trb")
    opt = torch.optim.Adam(det.parameters(), 2e-3); first = None
    for i in range(60):
        out = det(b); loss = det_loss(out["logits"], b["heat"]) + 0.02 * app_loss(out["B"], out["dF"], b["target"])
        opt.zero_grad(); loss.backward(); opt.step(); first = first if first is not None else float(loss)
    assert float(loss) < 0.7 * first and torch.isfinite(loss)


def test_metrics_perfect_prediction():
    b = make_batch(3, 2); logits = (b["heat"] - 0.5) * 40
    s = detection_stats(logits, b["pos"], b["mask"], 0.5)
    assert s["fn"] == 0 and s["fp"] == 0
