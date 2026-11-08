import torch
from vflash import VFlashDrafter, ARBaseline

def test_shapes():
    m = VFlashDrafter(); ctx = torch.randn(3, 8, 128)
    assert m(ctx, "point").shape == (3, 1, 2)
    assert m(ctx, "box").shape == (3, 1, 4)
    assert m(ctx, "traj2d", 12).shape == (3, 12, 2)
    assert m(ctx, "traj3d").shape == (3, 16, 3)

def test_box_valid():
    m = VFlashDrafter(); b = m(torch.randn(8, 8, 128) * 5, "box")
    assert (b[..., 2:] > 0).all()

def test_single_pass():
    m = VFlashDrafter(); calls = []
    m.decoder.register_forward_hook(lambda *a: calls.append(1))
    m(torch.randn(2, 8, 128), "traj2d", 16)
    assert len(calls) == 1

def test_ar_passes_equal_length():
    m = ARBaseline(); calls = []
    m.decoder.register_forward_hook(lambda *a: calls.append(1))
    m(torch.randn(2, 8, 128), length=10)
    assert len(calls) == 10

def test_grad():
    m = VFlashDrafter(); m(torch.randn(2, 8, 128), "traj2d").sum().backward()
    assert m.slot_q.grad is not None
