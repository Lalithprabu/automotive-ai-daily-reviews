"""
Shape + backward-pass smoke tests for the Planning Expert.
Run with: pytest tests/test_planning_expert.py -v
"""
import torch

from src.losses.flow_matching import FlowMatchingObjective
from src.models.planning_expert import PlanningExpert
from src.models.vlm_kv_cache import VLMKVCache
from src.sampling.euler_sampler import euler_sample

B = 2
MODEL_DIM = 64          # small dims for a fast test
N_HEADS = 8
N_LAYERS = 8             # scaled down from 32
N_CACHES = 4              # scaled down from 8
N_KV_HEADS = 4            # grouped-query heads in the (mock) VLM backbone
HEAD_DIM = MODEL_DIM // N_HEADS
L_CTX = 32                # mock cached context length
N_WAYPOINTS = 50          # 5s @ 10Hz, matches the paper


def _build_model_and_cache():
    torch.manual_seed(0)
    model = PlanningExpert(
        model_dim=MODEL_DIM, n_layers=N_LAYERS, n_heads=N_HEADS,
        n_caches=N_CACHES, cond_dim=MODEL_DIM, n_waypoints=N_WAYPOINTS,
    )
    kv_cache = VLMKVCache(
        keys=[torch.randn(B, N_KV_HEADS, L_CTX, HEAD_DIM) for _ in range(N_CACHES)],
        values=[torch.randn(B, N_KV_HEADS, L_CTX, HEAD_DIM) for _ in range(N_CACHES)],
    )
    return model, kv_cache


def test_flow_matching_loss_and_backward():
    model, kv_cache = _build_model_and_cache()
    instruction_embed = torch.randn(B, MODEL_DIM)
    ego_state_embed = torch.randn(B, MODEL_DIM)
    x1_gt = torch.randn(B, N_WAYPOINTS, 3)

    objective = FlowMatchingObjective()
    loss = objective(model, x1_gt, kv_cache, instruction_embed, ego_state_embed)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())


def test_euler_sampler_output_shape():
    model, kv_cache = _build_model_and_cache()
    instruction_embed = torch.randn(B, MODEL_DIM)
    ego_state_embed = torch.randn(B, MODEL_DIM)

    trajectory = euler_sample(
        model, kv_cache, instruction_embed, ego_state_embed,
        n_waypoints=N_WAYPOINTS, n_steps=10,
    )
    assert trajectory.shape == (B, N_WAYPOINTS, 3)


def test_mismatched_kv_cache_count_raises():
    model, kv_cache = _build_model_and_cache()
    kv_cache.keys = kv_cache.keys[:-1]   # drop one cache group
    kv_cache.values = kv_cache.values[:-1]
    instruction_embed = torch.randn(B, MODEL_DIM)
    ego_state_embed = torch.randn(B, MODEL_DIM)
    x_t = torch.randn(B, N_WAYPOINTS, 3)
    t = torch.rand(B)

    try:
        model(x_t, t, kv_cache, instruction_embed, ego_state_embed)
        assert False, "expected an assertion error for mismatched cache count"
    except AssertionError:
        pass
