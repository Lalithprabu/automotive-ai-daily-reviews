import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from mocar.data import SyntheticDrivingDataset
from mocar.model import MoCAR, OneShotBaseline, segment, compose, rot, wrap

ds = SyntheticDrivingDataset(32, seed=0)
B = {k: v[:8] for k, v in ds.d.items()}


def test_segment_compose_roundtrip():
    loc, dth, st = segment(B["fut_pts"], B["fut_hd"])
    pts, starts, _ = compose(loc, dth, st[:, 0])
    assert (pts - B["fut_pts"][:, 1:]).abs().max() < 1e-4
    assert loc.shape == (8, 6, 5, 2) and dth.shape == (8, 6)


def test_rotation_orthonormal():
    R = rot(torch.tensor([0.3, -1.2])); assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(2).expand(2, 2, 2), atol=1e-6)


def test_output_shapes_and_finite():
    m = MoCAR(); o = m(B); assert o["pts"].shape == (8, 3, 30, 2) and o["scores"].shape == (8, 3)
    assert torch.isfinite(o["pts"]).all()
    assert OneShotBaseline()(B)["pts"].shape == (8, 3, 30, 2)


def test_causality_future_codes_do_not_leak():
    """Teacher-forced prediction of code j must not depend on ground-truth code >= j (decoder-only causal mask)."""
    m = MoCAR().eval(); torch.manual_seed(0)
    o1 = m(B, teacher_forcing=True)["pred_codes"]
    B2 = dict(B); fp = B["fut_pts"].clone(); fp[:, 20:] += 5.0  # corrupt the last ground-truth future segments
    B2["fut_pts"] = fp
    o2 = m(B2, teacher_forcing=True)["pred_codes"]
    # codes 0..2 are predicted from history + GT codes < 3 (segments 0..2 span steps 0..15) -> unchanged
    assert torch.allclose(o1[:, :, :3], o2[:, :, :3], atol=1e-5)


def test_free_running_does_not_use_ground_truth_future():
    m = MoCAR().eval()
    B2 = dict(B); B2["fut_pts"] = B["fut_pts"] * 0 + 99.0; B2["fut_hd"] = B["fut_hd"] * 0
    assert torch.allclose(m(B)["pts"], m(B2)["pts"], atol=1e-6)


def test_coordinate_awareness_changes_output():
    torch.manual_seed(0); a = MoCAR(coord_aware=True).eval(); b = MoCAR(coord_aware=False).eval()
    b.load_state_dict(a.state_dict()); assert not torch.allclose(a(B)["pts"], b(B)["pts"])


def test_gradients_reach_tokenizer_and_predictor():
    m = MoCAR(); l, _ = m.loss(B); l.backward()
    assert m.tok.dec[0].weight.grad.abs().sum() > 0 and m.next_code[-1].weight.grad.abs().sum() > 0
    assert m.scene.pt[0].weight.grad.abs().sum() > 0


def test_overfit_tiny_batch():
    torch.manual_seed(0); m = MoCAR(); opt = torch.optim.AdamW(m.parameters(), 3e-3)
    b = {k: v[:8] for k, v in ds.d.items()}
    first = None
    for i in range(150):
        l, st = m.loss(b); opt.zero_grad(); l.backward(); opt.step(); first = first or st["fr_ade"]
    assert st["fr_ade"] < 0.5 * first
