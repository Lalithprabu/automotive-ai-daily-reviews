import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from eco.eco_layer import EndpointConstrainedOptimizer
from eco.policy import WaypointPolicy
from eco.env import Route, to_ego, from_ego


def straight(b=2, T=8, H=4, v=6.0, dt=0.5):
    hist = torch.stack([torch.tensor([-dt * (H - j) * v, 0.0]) for j in range(H)])[None].repeat(b, 1, 1)
    wp = torch.stack([torch.tensor([dt * (k + 1) * v, 0.0]) for k in range(T)])[None].repeat(b, 1, 1)
    return wp, hist


def jerk(wp, hist):
    s = torch.cat([hist, torch.zeros(hist.shape[0], 1, 2), wp], 1)
    return (s[:, 3:] - 3 * s[:, 2:-1] + 3 * s[:, 1:-2] - s[:, :-3]).norm(dim=-1).pow(2).mean().item()


def test_shapes():
    eco = EndpointConstrainedOptimizer(); wp, h = straight()
    assert eco(wp, h).shape == wp.shape


def test_endpoint_preserved_exactly():
    torch.manual_seed(0); eco = EndpointConstrainedOptimizer(); wp, h = straight()
    wp = wp + 0.5 * torch.randn_like(wp)
    assert torch.equal(eco(wp, h)[:, -1], wp[:, -1])


def test_clean_straight_line_is_fixed_point():
    eco = EndpointConstrainedOptimizer(); wp, h = straight()
    assert torch.allclose(eco(wp, h), wp, atol=1e-4)


def test_reduces_jerk_of_noisy_waypoints():
    torch.manual_seed(1); eco = EndpointConstrainedOptimizer(); wp, h = straight(b=64)
    noisy = wp + 0.4 * torch.randn_like(wp)
    assert jerk(eco(noisy, h), h) < 0.3 * jerk(noisy, h)


def test_high_fidelity_recovers_identity():
    torch.manual_seed(2); wp, h = straight(); wp = wp + 0.3 * torch.randn_like(wp)
    eco = EndpointConstrainedOptimizer(lambda_fidelity=1e6)
    assert torch.allclose(eco(wp, h), wp, atol=1e-3)


def test_anchors_to_executed_history():
    """Same raw waypoints, different executed history -> different first waypoints (history anchoring is real)."""
    eco = EndpointConstrainedOptimizer(lambda_fidelity=0.05); wp, h = straight(b=1)
    h2 = h.clone(); h2[..., 1] += torch.tensor([0.0, 0.0, 0.4, 0.8])      # vehicle was drifting left
    assert (eco(wp, h)[:, 0] - eco(wp, h2)[:, 0]).abs().max() > 1e-3


def test_gradient_flows_to_raw_waypoints_and_history():
    eco = EndpointConstrainedOptimizer(); wp, h = straight(); wp.requires_grad_(); h.requires_grad_()
    eco(wp, h).pow(2).sum().backward()
    assert wp.grad.abs().sum() > 0 and h.grad.abs().sum() > 0


def test_matches_reference_solver():
    """Closed form == scipy-free normal-equation check via autograd: gradient of objective ~ 0 at the solution."""
    torch.manual_seed(3); eco = EndpointConstrainedOptimizer(); wp, h = straight(b=1); wp = wp + 0.3 * torch.randn_like(wp)
    x = eco(wp, h)[:, :-1].clone().double().requires_grad_()
    s = torch.cat([h.double(), torch.zeros(1, 1, 2, dtype=torch.double), x, wp[:, -1:].double()], 1)
    d2 = s[:, 2:] - 2 * s[:, 1:-1] + s[:, :-2]; d3 = s[:, 3:] - 3 * s[:, 2:-1] + 3 * s[:, 1:-2] - s[:, :-3]
    f = 1.0 * d2.pow(2).sum() + 4.0 * d3.pow(2).sum() + 0.6 * (x - wp[:, :-1].double()).pow(2).sum()
    f.backward(); assert x.grad.abs().max() < 1e-2   # float32 solve, coordinates ~20 m


def test_policy_shapes():
    p = WaypointPolicy(); o = torch.randn(5, p.in_dim); assert p(o).shape == (5, 8, 2) and p.history_from_obs(o).shape == (5, 4, 2)


def test_frame_roundtrip():
    pts = np.random.randn(6, 2); back = from_ego(to_ego(pts[:, 0], pts[:, 1], 3.0, -2.0, 0.7), 3.0, -2.0, 0.7)
    assert np.allclose(back, pts, atol=1e-9)


def test_route_project_lateral_sign():
    r = Route(np.random.default_rng(0)); x, y, th, _ = r.at(50.0)
    _, lat = r.project(x - 1.0 * np.sin(th), y + 1.0 * np.cos(th)); assert abs(lat - 1.0) < 0.1
