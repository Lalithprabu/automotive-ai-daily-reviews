import torch
from models.motoanticipator import MotoAnticipator, EgoMotionCompensation
from src.synthetic import make_batch, future_target, T_OBS, H, K


def test_shapes():
    b = make_batch(8, torch.Generator().manual_seed(0)); m = MotoAnticipator()
    lg, pred, cv = m(b["ego"], b["agents_ego"], torch.full((8,), 12))
    assert lg.shape == (8, T_OBS, 4) and pred.shape == (8, K, H, 2) and cv.shape == pred.shape


def test_causality():  # changing the future frames must not change earlier logits
    b = make_batch(4, torch.Generator().manual_seed(1)); m = MotoAnticipator().eval()
    a, _, _ = m(b["ego"], b["agents_ego"]); e2 = b["ego"].clone(); p2 = b["agents_ego"].clone(); e2[:, 20:] += 5; p2[:, 20:] += 5
    c, _, _ = m(e2, p2); assert torch.allclose(a[:, :20], c[:, :20], atol=1e-5)


def test_starts_at_cv():  # zero-init head => prediction equals the compensated const-velocity forecast
    b = make_batch(4, torch.Generator().manual_seed(2)); m = MotoAnticipator()
    _, pred, cv = m(b["ego"], b["agents_ego"], torch.full((4,), 20)); assert torch.allclose(pred, cv)


def test_compensation_recovers_agent_velocity():  # noise-free: compensated CV beats apparent CV
    g = torch.Generator().manual_seed(3); b = make_batch(256, g)
    L = torch.full((256,), 25); tgt = future_target(b, L); tau = torch.arange(1, H + 1).float() * 0.2
    errs = []
    for comp in (True, False):
        v = EgoMotionCompensation(comp)(b["agents_ego"], b["ego"])[:, 24]
        errs.append(((b["agents_ego"][:, 24, :, None] + v[:, :, None] * tau[None, None, :, None]) - tgt).norm(dim=-1).mean().item())
    assert errs[0] < errs[1], errs


def test_target_frame():  # at L the ego frame target of a stationary-agent set is consistent
    b = make_batch(2, torch.Generator().manual_seed(4)); L = torch.tensor([10, 30]); t = future_target(b, L)
    assert t.shape == (2, K, H, 2) and torch.isfinite(t).all()
