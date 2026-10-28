import math, torch, pytest
from spatialbev.models import SpatialBEVDynamics, MODES, P, physics_rollout
from spatialbev.terrain import make_terrain, truth_step, sample_episodes, apply_delta, physics_delta, S, E

BANK = make_terrain(4, 0)


@pytest.mark.parametrize("mode", MODES)
def test_shapes_and_finite(mode):
    m = SpatialBEVDynamics(mode)
    s0 = torch.tensor([[0., 0., 0.3, 6.]] * 3)
    a = torch.zeros(3, 7, 2); a[..., 1] = 6
    out = m.rollout(s0, a, BANK[:3])
    assert out.shape == (3, 7, 4) and torch.isfinite(out).all()


@pytest.mark.parametrize("mode", MODES)
def test_zero_init_equals_physics(mode):
    m = SpatialBEVDynamics(mode)
    s0 = torch.tensor([[1., -2., 0.5, 7.], [0., 0., -1., 4.]])
    a = torch.randn(2, 10, 2) * 0.1 + torch.tensor([0., 6.])
    assert torch.allclose(m.rollout(s0, a, BANK[:2]), physics_rollout(s0, a), atol=1e-5)


def test_window_is_ego_aligned():
    """Height ramp along world +x. Facing +x: values must increase with the forward axis (cols);
    facing +y: constant along cols, and the lateral axis (rows) carries the ramp."""
    ramp = torch.linspace(-E, E, S).view(1, 1, 1, S).expand(1, 1, S, S)
    terr = torch.cat([ramp, torch.ones_like(ramp) * .7, torch.ones_like(ramp) * .5], 1)
    m = SpatialBEVDynamics("spatial")
    w0 = m.window(torch.tensor([[0., 0., 0., 5.]]), terr)[0, 0]
    assert (w0[:, 1:] - w0[:, :-1]).min() > 0
    w1 = m.window(torch.tensor([[0., 0., math.pi / 2, 5.]]), terr)[0, 0]
    assert (w1[:, 1:] - w1[:, :-1]).abs().max() < 1e-4        # forward = +y: no change
    assert (w1[:-1] - w1[1:]).abs().min() > 0                 # lateral axis now carries the ramp


def test_gradient_reaches_predicted_position_through_window():
    """The BEV query is differentiable w.r.t. the *predicted state* (needed for BPTT training)."""
    m = SpatialBEVDynamics("spatial")
    for p in m.mlp[-1].parameters(): torch.nn.init.normal_(p, std=0.1)
    s0 = torch.tensor([[0., 0., 0.2, 6.]], requires_grad=True)
    a = torch.zeros(1, 3, 2); a[..., 1] = 6
    m.rollout(s0, a, BANK[:1])[:, -1, :2].sum().backward()
    assert s0.grad is not None and s0.grad[0, :2].abs().sum() > 0


def test_truth_flat_high_friction_goes_straight():
    flat = torch.zeros(1, 3, S, S); flat[:, 1] = 1.0
    s = torch.tensor([[0., 0., 0., 6.]])
    for _ in range(20):
        s = truth_step(s, torch.tensor([[0., 6.]]), flat)
    assert abs(s[0, 1]) < 1e-4 and abs(s[0, 2]) < 1e-4 and abs(s[0, 0] - 12) < 0.2


def test_left_right_friction_difference_yaws_the_truth():
    """Mud on one side of the car, grip on the other -> opposite yaw pulls (asymmetric traction)."""
    t = torch.zeros(1, 3, S, S); t[:, 1] = 1.0; t[:, 1, :S // 2, :] = 0.3   # y < 0 is mud
    act = torch.tensor([[0., 6.]])
    mud_right = truth_step(torch.tensor([[0., -0.8, 0., 6.]]), act, t)[0, 2]   # heading +x, right wheels in mud
    t2 = t.flip(2)                                                              # mud now at y > 0
    mud_left = truth_step(torch.tensor([[0., 0.8, 0., 6.]]), act, t2)[0, 2]
    assert mud_right > 0 > mud_left
