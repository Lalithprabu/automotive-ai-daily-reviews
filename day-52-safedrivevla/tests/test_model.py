import numpy as np, torch
from src.synthetic import sample_scenes, build_batch, strict_follow_rollout, label_mode, executed_collision, FOLLOW, REFUSE
from src.safedrivevla import SafeDriveVLAMini, compute_loss


def _b(n=16, seed=0):
    return {k: torch.as_tensor(v) for k, v in build_batch(sample_scenes(n, np.random.default_rng(seed))).items()}


def test_shapes():
    b = _b(); out = SafeDriveVLAMini()(b["agents"], b["ego"], b["nav"])
    assert out["risk"].shape == (16, 8) and out["gap"].shape == (16, 8)
    assert out["mode_logits"].shape == (16, 3) and out["profile"].shape == (16, 8)


def test_offroad_is_refuse():
    sc = dict(ego_lane=np.array([0]), nav=np.array([1]), lane=np.array([[2]]), s=np.array([[30.]]), v=np.array([[0.]]),
              valid=np.ones((1, 1), np.float32))
    c, g, off = strict_follow_rollout(sc)
    assert off[0] and label_mode(c, off)[0] == REFUSE


def test_clear_road_is_follow():
    sc = dict(ego_lane=np.array([1]), nav=np.array([1]), lane=np.array([[2]]), s=np.array([[30.]]), v=np.array([[0.]]),
              valid=np.ones((1, 1), np.float32))
    c, g, off = strict_follow_rollout(sc)
    assert label_mode(c, off)[0] == FOLLOW


def test_blocked_lane_collides_when_obeying():
    sc = dict(ego_lane=np.array([1]), nav=np.array([1]), lane=np.array([[0]]), s=np.array([[2.]]), v=np.array([[0.]]),
              valid=np.ones((1, 1), np.float32))
    naive = np.array([[0, .25, .6, 1, 1, 1, 1, 1]], np.float32)
    assert executed_collision(sc, naive)[0]
    assert not executed_collision(sc, np.zeros((1, 8), np.float32))[0]


def test_loss_backward_and_ablation_flag():
    b = _b()
    for dream in (True, False):
        m = SafeDriveVLAMini(use_dream=dream); out = m(b["agents"], b["ego"], b["nav"], forced_mode=b["mode"])
        loss, parts = compute_loss(out, b, dict(risk=1, gap=1, mode=1, action=1)); loss.backward()
        assert ("risk" in parts) == dream and torch.isfinite(loss)


def test_padding_agents_do_not_change_output():
    b = _b(4); m = SafeDriveVLAMini().eval()
    a2 = b["agents"].clone(); inv = a2[..., 5] == 0; a2[inv, :5] = 9.0          # garbage in padded agents
    with torch.no_grad():
        o1, o2 = m(b["agents"], b["ego"], b["nav"]), m(a2, b["ego"], b["nav"])
    assert torch.allclose(o1["mode_logits"], o2["mode_logits"], atol=1e-5)
