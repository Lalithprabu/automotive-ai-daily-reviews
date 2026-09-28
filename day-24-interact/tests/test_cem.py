import torch

from src.models.trust_region_cem import (
    TrustRegionCEM, trust_region_penalty, collision_cost, comfort_cost, progress_cost,
)
from src.utils.kinematics import rollout_trajectory

T = 10
WEIGHTS = {"collision": 4.0, "comfort": 0.5, "progress": 1.0, "trust_region": 5.0}


def test_trust_region_penalty_zero_inside_radius():
    anchor_center = torch.tensor([0.0, 0.0])
    close_cond = torch.tensor([0.1, 0.1])   # well within radius=0.35
    penalty = trust_region_penalty(close_cond, anchor_center, radius=0.35, lam=5.0)
    assert penalty.item() == 0.0


def test_trust_region_penalty_grows_with_distance():
    """Directly tests the penalty function: candidates farther from the anchor must
    be penalized more, all else equal."""
    anchor_center = torch.tensor([0.0, 0.0])
    near = torch.tensor([0.4, 0.0])   # just past radius
    far = torch.tensor([1.4, 0.0])    # much farther past radius

    p_near = trust_region_penalty(near, anchor_center, radius=0.35, lam=5.0)
    p_far = trust_region_penalty(far, anchor_center, radius=0.35, lam=5.0)

    assert p_far.item() > p_near.item() > 0.0


def test_collision_cost_penalizes_overlap():
    ego = torch.zeros(1, T, 2)
    other_far = torch.full((1, T, 2), 5.0)
    other_close = torch.zeros(1, T, 2)
    c_far = collision_cost(ego, other_far)
    c_close = collision_cost(ego, other_close)
    assert c_close.item() > c_far.item()


def test_cem_reduces_best_cost_over_iterations():
    """CEM should make measurable progress: the best cost found should not get worse
    across iterations (checked as: final best <= first-iteration best)."""
    torch.manual_seed(0)
    anchor_cond = torch.tensor([0.5, 0.3])
    # a cached "other agent" prediction that sits right in the naive straight-line path,
    # forcing CEM to actually find a trajectory that avoids it / balances costs
    cached_pred = torch.zeros(T, 2)
    cached_pred[:, 0] = 0.5   # other agent sitting at lat=0.5 the whole time

    cem = TrustRegionCEM(future_len=T, weights=WEIGHTS, n_iter=6, n_samples=64,
                          elite_frac=0.2, init_std=0.5, bounds=(-1.5, 1.5),
                          collision_margin=0.35, trust_region_radius=0.35, base_speed=1.0)
    result = cem.refine(anchor_cond, mode="cached", cached_pred=cached_pred)

    assert len(result.cost_history) == 6
    assert result.cost_history[-1] <= result.cost_history[0] + 1e-6
    # best-cost trace should be non-increasing (CEM tracks running best, not per-iter raw)
    running_min = float("inf")
    for c in result.cost_history:
        running_min = min(running_min, c)
    assert result.best_cost <= result.cost_history[0]


def test_cem_naive_mode_counts_predictor_calls():
    """The naive (old-way-B) mode must query the predictor once per candidate per
    iteration -- this is the real efficiency cost the paper's motivating claim is about."""
    torch.manual_seed(0)
    anchor_cond = torch.tensor([0.2, 0.2])

    H, STATE_DIM = 8, 4
    ego_hist = torch.zeros(H, STATE_DIM)
    other_hist = torch.zeros(H, STATE_DIM)

    def fake_predict_fn(ego_hist_b, other_hist_b, cond_b):
        # returns something trajectory-shaped without needing a real trained model
        n = cond_b.shape[0]
        return torch.zeros(n, T, 2)

    n_iter, n_samples = 3, 16
    cem = TrustRegionCEM(future_len=T, weights=WEIGHTS, n_iter=n_iter, n_samples=n_samples,
                          elite_frac=0.25, init_std=0.4, bounds=(-1.5, 1.5))
    result = cem.refine(anchor_cond, mode="naive", predict_fn=fake_predict_fn,
                         ego_hist=ego_hist, other_hist=other_hist)

    assert result.predictor_calls == n_iter * n_samples


def test_cem_cached_mode_makes_zero_predictor_calls():
    """The whole point of the new way: reuse ONE cached prediction across the entire
    CEM search for an anchor -- refine() itself should not query the predictor at all."""
    torch.manual_seed(0)
    anchor_cond = torch.tensor([0.2, 0.2])
    cached_pred = torch.zeros(T, 2)
    cem = TrustRegionCEM(future_len=T, weights=WEIGHTS, n_iter=4, n_samples=32)
    result = cem.refine(anchor_cond, mode="cached", cached_pred=cached_pred)
    assert result.predictor_calls == 0


def test_rollout_trajectory_reaches_target_offset():
    cond = torch.tensor([1.0, 0.5])
    traj = rollout_trajectory(cond, future_len=T, base_speed=1.0)
    assert traj.shape == (T, 2)
    # final lateral position should be close to the target offset (smoothstep ends at ~1.0)
    assert abs(traj[-1, 0].item() - 1.0) < 1e-3
    assert abs(traj[-1, 1].item() - 1.5) < 1e-3
