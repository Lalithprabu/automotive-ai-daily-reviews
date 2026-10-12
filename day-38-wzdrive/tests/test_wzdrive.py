import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from src.wzdrive import WZDrive, GazeConditionedSceneAttention, generate_dataset, wta_loss, min_ade_fde, topk_min_ade
from src.wzdrive.data import CLASS_NAMES


def batch(n=16, seed=0):
    return generate_dataset(n, seed)


def test_shapes():
    d = batch(); m = WZDrive().eval()
    tr, lg = m(d["kin"], d["ctrl"], d["gaze"], d["scene"])
    assert tr.shape == (16, 6, 12, 2) and lg.shape == (16, 6) and torch.isfinite(tr).all()


def test_padded_scene_tokens_ignored():
    d = batch(); m = WZDrive(dropout=0.0).eval()
    a, _ = m(d["kin"], d["ctrl"], d["gaze"], d["scene"])
    s2 = d["scene"].clone(); pad = s2[..., 6] < 0.5; s2[pad] = torch.tensor([1., 0, 50, 3, 2, 2, 0, 0])
    b, _ = m(d["kin"], d["ctrl"], d["gaze"], s2)
    assert torch.allclose(a, b, atol=1e-4)


def test_gaze_attention_prefers_looked_at_object():
    layer = GazeConditionedSceneAttention(32, 4)
    q = torch.zeros(1, 32); tok = torch.zeros(1, 3, 32)           # identical content -> only gaze bias differs
    bearing = torch.tensor([[-0.3, 0.0, 0.3]]); valid = torch.ones(1, 3, dtype=torch.bool)
    _, att = layer(q, tok, bearing, torch.tensor([0.3]), valid)
    assert att[0].argmax() == 2
    _, att0 = layer(q, tok, bearing, torch.tensor([0.3]), valid, use_gaze=False)
    assert torch.allclose(att0[0], torch.full((3,), 1 / 3), atol=1e-5)


def test_gradient_reaches_gaze_and_control():
    d = batch(); m = WZDrive(dropout=0.0)
    g, c = d["gaze"].clone().requires_grad_(), d["ctrl"].clone().requires_grad_()
    tr, lg = m(d["kin"], c, g, d["scene"]); wta_loss(tr / 5, lg, d["fut"] / 5)[0].backward()
    assert g.grad.abs().sum() > 0 and c.grad.abs().sum() > 0


def test_ablation_flags_zero_out_inputs():
    d = batch(); m = WZDrive(dropout=0.0, use_gaze=False, use_control=False).eval()
    a, _ = m(d["kin"], d["ctrl"], d["gaze"], d["scene"])
    b, _ = m(d["kin"], d["ctrl"] * 5, d["gaze"] * -3, d["scene"])
    assert torch.allclose(a, b, atol=1e-5)


def test_wta_loss_uses_best_mode_and_min_metrics():
    gt = torch.zeros(2, 12, 2); pred = torch.randn(2, 6, 12, 2); pred[:, 3] = gt
    ade, fde, best = min_ade_fde(pred, gt)
    assert (best == 3).all() and ade.max() < 1e-6
    loss, reg, _ = wta_loss(pred, torch.randn(2, 6), gt, relax_eps=0.0)
    assert reg < 1e-6


def test_dataset_gaze_leads_early_merge():
    d = generate_dataset(800, 3); cls = d["cls"]
    side = d["gaze"][..., 0][:, -3:].abs().mean(1)                 # |gaze yaw| in last history steps
    early, late = side[cls == 1].mean(), side[cls == 2].mean()
    assert early > 2.0 * late, (early, late)
    assert set(cls.tolist()) == {0, 1, 2, 3}


def test_overfit_tiny_batch():
    torch.manual_seed(0); d = batch(8, 5); m = WZDrive(dropout=0.0); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(600):
        tr, lg = m(d["kin"], d["ctrl"], d["gaze"], d["scene"]); l = wta_loss(tr / 5, lg, d["fut"] / 5)[0]
        opt.zero_grad(); l.backward(); opt.step()
    ade, _, _ = min_ade_fde(m(d["kin"], d["ctrl"], d["gaze"], d["scene"])[0].detach(), d["fut"])
    assert ade.mean() < 0.5, ade.mean()


def test_core_claim_gaze_control_beat_baseline():
    """Falsifiable: with identical budgets, adding gaze+control lowers the *top-1 (ranked)* ADE on held-out synthetic data.
    (minADE@6 is deliberately not asserted: K=6 modes already cover this world's 4 intents, so it saturates.)"""
    torch.manual_seed(1); tr, va = generate_dataset(1500, 11), generate_dataset(400, 12); res = {}
    for tag, kw in [("b", dict(use_gaze=False, use_control=False)), ("w", {})]:
        torch.manual_seed(1); m = WZDrive(dropout=0.0, **kw); opt = torch.optim.AdamW(m.parameters(), 1e-3)
        for ep in range(25):
            perm = torch.randperm(1500)
            for i in range(0, 1500, 128):
                ix = perm[i:i + 128]; t, l = m(tr["kin"][ix], tr["ctrl"][ix], tr["gaze"][ix], tr["scene"][ix])
                loss = wta_loss(t / 5, l, tr["fut"][ix] / 5)[0]; opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); tr_, lg_ = m(va["kin"], va["ctrl"], va["gaze"], va["scene"]); res[tag] = topk_min_ade(tr_.detach(), lg_.detach(), va["fut"], 1).mean().item()
    assert res["w"] < res["b"], res
