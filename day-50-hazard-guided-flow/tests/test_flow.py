import torch, numpy as np
from models.flow_model import FlowNet, flow_matching_loss, hazard_energy, sample
from src.scene import box_sdf, inflated_box, encode, decode, make_context

def test_sdf_signs():
    box = inflated_box(torch.tensor([20.0]))
    assert box_sdf(torch.tensor([22.0]), torch.tensor([0.0]), box) < 0
    assert box_sdf(torch.tensor([5.0]), torch.tensor([0.0]), box) > 0
    assert box_sdf(torch.tensor([22.0]), torch.tensor([10.0]), box) > 0

def test_encode_decode_roundtrip():
    t = torch.randn(3, 20, 2) * 5
    assert torch.allclose(decode(encode(t)), t, atol=1e-5)

def test_energy_zero_when_clear_or_absent():
    far = encode(torch.stack([torch.linspace(0, 10, 20), torch.zeros(20)], -1)[None])
    assert hazard_energy(far, torch.tensor([60.0]), torch.tensor([True]), 0.8) == 0
    thru = encode(torch.stack([torch.linspace(0, 40, 20), torch.zeros(20)], -1)[None])
    assert hazard_energy(thru, torch.tensor([20.0]), torch.tensor([True])) > 0
    assert hazard_energy(thru, torch.tensor([20.0]), torch.tensor([False])) == 0

def test_energy_gradient_points_back_along_track():
    x = encode(torch.stack([torch.linspace(0, 40, 20), torch.zeros(20)], -1)[None]).requires_grad_(True)
    hazard_energy(x, torch.tensor([20.0]), torch.tensor([True])).sum().backward()
    gx = x.grad.view(1, 20, 2)[0, :, 0]
    xs = torch.linspace(0, 40, 20); front = xs < 20.0          # waypoints before the obstacle's midpoint
    assert (gx[front] >= 0).all() and gx.max() > 0           # descending E (x -= g) pulls them backwards (brake)

def test_weights_frozen_by_guidance():
    net = FlowNet(20, 7, 32, 2).eval(); before = [p.clone() for p in net.parameters()]
    c = make_context(torch.full((4, 4), 10.0), torch.full((4,), 25.0), torch.ones(4, dtype=torch.bool))
    sample(net, c, torch.full((4,), 25.0), torch.ones(4, dtype=torch.bool), 4, 3.0, 5.0)
    assert all(torch.equal(a, b) for a, b in zip(before, net.parameters()))
    assert all(p.grad is None for p in net.parameters())

def test_zero_lambda_equals_baseline_and_shapes():
    net = FlowNet(20, 7, 32, 2).eval()
    c = make_context(torch.full((2, 4), 10.0), torch.full((2,), 25.0), torch.ones(2, dtype=torch.bool))
    d, p = torch.full((2,), 25.0), torch.ones(2, dtype=torch.bool)
    n0 = torch.randn(2, 40)
    a = sample(net, c, d, p, 4, 0.0, noise=n0.clone()); b = sample(net, c, None, None, 4, 5.0, noise=n0.clone())
    assert a.shape == (2, 40) and torch.allclose(a, b)

def test_guidance_reduces_energy_on_random_net():
    torch.manual_seed(0); net = FlowNet(20, 7, 64, 2).eval()
    B = 64; c = make_context(torch.full((B, 4), 12.0), torch.full((B,), 25.0), torch.ones(B, dtype=torch.bool))
    d, p = torch.full((B,), 25.0), torch.ones(B, dtype=torch.bool); n0 = torch.randn(B, 40)
    e0 = hazard_energy(sample(net, c, d, p, 8, 0.0, noise=n0.clone()), d, p).mean()
    e1 = hazard_energy(sample(net, c, d, p, 8, 3.0, noise=n0.clone()), d, p).mean()
    assert e1 < e0

def test_overfit_tiny_batch_and_gradflow():
    torch.manual_seed(0); net = FlowNet(20, 7, 64, 2); opt = torch.optim.Adam(net.parameters(), 3e-3)
    x1 = torch.randn(8, 40) * 0.3; c = torch.randn(8, 7)
    l0 = flow_matching_loss(net, x1, c).item()
    for _ in range(300):
        l = flow_matching_loss(net, x1, c); opt.zero_grad(); l.backward(); opt.step()
    assert l.item() < 0.6 * l0
