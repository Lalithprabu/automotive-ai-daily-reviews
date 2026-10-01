import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from emplan.model import EMPlan, RegressionPlanner, kmeans_anchors
from emplan.reward import rule_terms, rule_reward
from emplan.data import make_dataset, parametric_trajs


def _scene(lead=15.0, left=False, right=False):
    s = torch.zeros(1, 9); s[0, 0] = 10.; s[0, 1] = lead; s[0, 2] = 0.
    s[0, 4] = 100.; s[0, 7] = 100.
    if left: s[0, 3] = 1.; s[0, 4] = 0.; s[0, 5] = 10.
    return s


def test_shapes():
    sc, ex = make_dataset(64, 0)
    m = EMPlan(kmeans_anchors(ex, 6), 32)
    lg, tr = m(sc)
    assert lg.shape == (64, 6) and tr.shape == (64, 6, 8, 2)
    assert m.plan(sc)[0].shape == (64, 8, 2) and RegressionPlanner(8, 32)(sc).shape == (64, 8, 2)


def test_reward_collision_and_clear():
    s = _scene(lead=15.0)
    t = torch.arange(1, 9).float() * 0.5
    straight = torch.stack([10 * t, torch.zeros(8)], -1)[None, None]
    swerve = straight.clone(); swerve[..., 1] = 3.5
    assert rule_terms(s, straight)["collision"].item() == 1.0
    assert rule_terms(s, swerve)["collision"].item() == 0.0
    assert rule_reward(s, swerve).item() > rule_reward(s, straight).item()


def test_left_vehicle_blocks_left_swerve():
    s = _scene(lead=15.0, left=True)
    t = torch.arange(1, 9).float() * 0.5
    swerve = torch.stack([10 * t, torch.full((8,), 3.5)], -1)[None, None]
    assert rule_terms(s, swerve)["collision"].item() == 1.0


def test_offroad():
    s = _scene(lead=100.)
    t = torch.arange(1, 9).float() * 0.5
    far = torch.stack([10 * t, torch.full((8,), 6.0)], -1)[None, None]
    assert rule_terms(s, far)["offroad"].item() == 1.0


def test_expert_is_multimodal_and_safe():
    sc, ex = make_dataset(400, 3)
    lat = ex[:, -1, 1]
    assert (lat > 2).any() and (lat < -2).any() and (lat.abs() < 1).any()   # left / right / stay
    assert rule_terms(sc, ex[:, None])["collision"].mean() < 0.02


def test_regression_mode_averages():
    """A single-mode regressor trained on multimodal demos ends up between modes -> collides far more than the expert."""
    sc, ex = make_dataset(1500, 1)
    m = RegressionPlanner(8, 64); opt = torch.optim.AdamW(m.parameters(), 3e-3)
    for _ in range(60):
        p = torch.randperm(len(sc))
        for i in range(0, len(sc), 128):
            b = p[i:i + 128]; l = (m(sc[b]) - ex[b]).abs().mean(); opt.zero_grad(); l.backward(); opt.step()
    vs, ve = make_dataset(500, 2)
    assert rule_terms(vs, m.plan(vs)[:, None])["collision"].mean() > 0.2


def test_gradient_flows_to_scores_and_offsets():
    sc, ex = make_dataset(16, 0); m = EMPlan(kmeans_anchors(ex, 4), 32)
    lg, tr = m(sc); (lg.sum() + tr.sum()).backward()
    assert m.score.weight.grad.abs().sum() > 0 and m.offset[-1].weight.grad.abs().sum() > 0


def test_stage2_uses_unpaired_labels_not_pairs():
    # each candidate receives an independent +/-1 label; collisions are always -1
    sc, ex = make_dataset(32, 0); m = EMPlan(kmeans_anchors(ex, 8), 32)
    _, tr = m(sc)
    c = rule_terms(sc, tr.detach())["collision"]
    r = rule_reward(sc, tr.detach())
    y = torch.where(r > r.max(1, keepdim=True).values - 3.0, 1.0, -1.0)
    y = torch.where(c > 0, -1.0, y)
    assert set(y.unique().tolist()) <= {-1.0, 1.0} and (y[c > 0] == -1).all()
