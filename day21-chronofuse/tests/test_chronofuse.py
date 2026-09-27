"""
Regression tests for the ChronoFuse reconstruction.

The single most important correctness property of this repo is the one
the test suite checks most directly: that `enable_chronofuse=False`
degrades gracefully to a well-defined "no compensation" baseline using the
exact same trained weights (not a separate model), and that the
ChronoFuse-enabled path actually depends on the cached temporal features
and the latency embedding (i.e. it isn't silently ignoring them).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.synthetic_events import SyntheticDrivingEventDataset, collate_samples
from src.models.chronofuse import ChronoFuseBlock, MultiScaleChronoFuse
from src.models.model import ChronoFuseDetector
from src.utils.latency_embed import LatencyEmbedding
from src.utils.losses import DetectionLoss, build_targets, STRIDE, gaussian_radius, draw_gaussian
from src.utils.metrics import decode_centers, center_distance_error, stale_baseline_error


def _tiny_model():
    return ChronoFuseDetector(
        in_channels=2, backbone_channels=(8, 16, 32), fpn_channels=16,
        num_classes=2, cache_len=3, latency_dim=16, max_latency=8.0,
    )


def test_latency_embedding_shape_and_determinism():
    embed = LatencyEmbedding(embed_dim=16, max_latency=8.0)
    latency = torch.tensor([1.0, 2.0, 4.0])
    out1 = embed(latency)
    out2 = embed(latency)
    assert out1.shape == (3, 16)
    assert torch.allclose(out1, out2), "embedding must be deterministic in eval-equivalent forward"
    # Different latencies must produce different embeddings.
    assert not torch.allclose(out1[0], out1[1])


def test_chronofuse_block_output_shape():
    block = ChronoFuseBlock(channels=16, latency_dim=16, cache_len=3)
    current = torch.randn(2, 16, 8, 8)
    cache = torch.randn(2, 3, 16, 8, 8)
    latency_embed = torch.randn(2, 16)
    out = block(current, cache, latency_embed)
    assert out.shape == current.shape


def test_chronofuse_block_zero_gate_is_near_identity():
    """With the latency-gate forced to 0 (simulating latency_steps -> the
    region the sigmoid saturates toward for a very small/zero requested
    latency), the fused output should reduce to (a normalized version of)
    the current features -- i.e. no correction is hallucinated when none
    is requested. We test this directly by zeroing the gate's weights."""
    block = ChronoFuseBlock(channels=8, latency_dim=8, cache_len=2)
    with torch.no_grad():
        block.latency_gate[0].weight.zero_()
        block.latency_gate[0].bias.fill_(-1e6)  # sigmoid(-1e6) == 0

    current = torch.randn(1, 8, 4, 4)
    cache = torch.randn(1, 2, 8, 4, 4)
    latency_embed = torch.randn(1, 8)
    out = block(current, cache, latency_embed)
    expected = block.norm(current)
    assert torch.allclose(out, expected, atol=1e-5)


def test_multiscale_chronofuse_runs_all_scales():
    fusion = MultiScaleChronoFuse(channels_per_scale=(8, 16, 32), latency_dim=16, cache_len=2)
    current = (torch.randn(1, 8, 16, 16), torch.randn(1, 16, 8, 8), torch.randn(1, 32, 4, 4))
    cache = (torch.randn(1, 2, 8, 16, 16), torch.randn(1, 2, 16, 8, 8), torch.randn(1, 2, 32, 4, 4))
    latency_embed = torch.randn(1, 16)
    out = fusion(current, cache, latency_embed)
    assert len(out) == 3
    for o, c in zip(out, current):
        assert o.shape == c.shape


def test_model_forward_shapes():
    model = _tiny_model()
    event_seq = torch.rand(2, 5, 2, 32, 32)
    latency_steps = torch.tensor([1.0, 3.0])
    hm, size, off = model(event_seq, latency_steps)
    assert hm.shape == (2, 2, 8, 8)
    assert size.shape == (2, 2, 8, 8)
    assert off.shape == (2, 2, 8, 8)


def test_disabling_chronofuse_changes_output_but_keeps_shape():
    """The single most important behavioral property: enable_chronofuse
    toggles between two genuinely different forward paths using the SAME
    weights (not a shape mismatch, not a no-op)."""
    model = _tiny_model()
    model.eval()
    event_seq = torch.rand(1, 5, 2, 32, 32)
    latency_steps = torch.tensor([3.0])

    hm_on, size_on, off_on = model(event_seq, latency_steps, enable_chronofuse=True)
    hm_off, size_off, off_off = model(event_seq, latency_steps, enable_chronofuse=False)

    assert hm_on.shape == hm_off.shape
    assert not torch.allclose(hm_on, hm_off), \
        "enabling/disabling ChronoFuse must change predictions (it must not be a no-op)"


def test_chronofuse_depends_on_cache_not_just_current_frame():
    """If we scramble the cached (past) frames while holding the current
    frame fixed, ChronoFuse's output must change -- otherwise it would be
    silently ignoring the temporal cache it's supposed to fuse."""
    block = ChronoFuseBlock(channels=8, latency_dim=8, cache_len=2)
    block.eval()
    current = torch.randn(1, 8, 6, 6)
    cache_a = torch.randn(1, 2, 8, 6, 6)
    cache_b = torch.randn(1, 2, 8, 6, 6)
    latency_embed = torch.randn(1, 8)

    out_a = block(current, cache_a, latency_embed)
    out_b = block(current, cache_b, latency_embed)
    assert not torch.allclose(out_a, out_b)


def test_latency_embedding_changes_gate_and_output():
    model = _tiny_model()
    model.eval()
    event_seq = torch.rand(1, 5, 2, 32, 32)

    hm_short, _, _ = model(event_seq, torch.tensor([1.0]))
    hm_long, _, _ = model(event_seq, torch.tensor([4.0]))
    assert not torch.allclose(hm_short, hm_long), \
        "different requested latencies must produce different predictions"


def test_build_targets_places_positive_at_expected_cell():
    centers = [torch.tensor([[40.0, 60.0]])]
    sizes = [torch.tensor([[12.0, 12.0]])]
    classes = [torch.tensor([0])]
    heatmap, size_t, offset_t, pos_mask = build_targets(centers, sizes, classes, out_h=32, out_w=32, num_classes=2)

    ix, iy = int(40.0 / STRIDE), int(60.0 / STRIDE)
    assert pos_mask[0, iy, ix].item() is True
    assert heatmap[0, 0, iy, ix].item() == 1.0
    assert heatmap[0, 1].max().item() < 1.0  # wrong class untouched


def test_gaussian_radius_positive_and_finite():
    r = gaussian_radius(torch.tensor([16.0, 16.0]))
    assert r > 0 and r < 1e6


def test_detection_loss_is_finite_and_backprops():
    model = _tiny_model()
    event_seq = torch.rand(2, 5, 2, 32, 32)
    latency_steps = torch.tensor([2.0, 3.0])
    hm, size, off = model(event_seq, latency_steps)

    centers = [torch.tensor([[10.0, 10.0]]), torch.tensor([[20.0, 5.0], [3.0, 3.0]])]
    sizes = [torch.tensor([[10.0, 10.0]]), torch.tensor([[8.0, 8.0], [5.0, 5.0]])]
    classes = [torch.tensor([0]), torch.tensor([1, 0])]
    heatmap_t, size_t, offset_t, pos_mask = build_targets(centers, sizes, classes, 8, 8, 2)

    loss_fn = DetectionLoss()
    loss, parts = loss_fn(hm, size, off, heatmap_t, size_t, offset_t, pos_mask)
    assert torch.isfinite(loss)
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "loss must produce nonzero gradients through the model"


def test_synthetic_dataset_shapes_and_collate():
    ds = SyntheticDrivingEventDataset(num_samples=4, height=32, width=32, t_obs=3, max_agents=2,
                                       max_latency_steps=3, seed=1)
    sample = ds[0]
    assert sample.event_seq.shape == (3, 2, 32, 32)
    assert sample.future_centers.shape[0] == sample.classes.shape[0]

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_samples)
    batch = next(iter(loader))
    assert batch["event_seq"].shape == (2, 3, 2, 32, 32)
    assert len(batch["future_centers"]) == 2


def test_future_centers_differ_from_stale_centers_for_moving_agents():
    """Sanity-checks the dataset's own core premise: a nonzero latency
    should generally move an agent's true future position away from its
    last-observed ('stale') position."""
    ds = SyntheticDrivingEventDataset(num_samples=32, height=128, width=128, t_obs=5, max_agents=2,
                                       max_latency_steps=4, seed=7)
    total_shift = 0.0
    count = 0
    for i in range(len(ds)):
        s = ds[i]
        shift = torch.norm(s.future_centers - s.stale_centers, dim=-1)
        total_shift += shift.sum().item()
        count += shift.numel()
    assert (total_shift / count) > 1.0, "average future-vs-stale displacement should be clearly nonzero"


def test_decode_centers_recovers_planted_peak():
    heatmap_logits = torch.full((2, 16, 16), -10.0)
    heatmap_logits[0, 5, 7] = 10.0  # class 0, row=5 (y), col=7 (x)
    offset = torch.zeros(2, 16, 16)
    dets = decode_centers(heatmap_logits, offset, num_per_class=[1, 0])
    assert len(dets) == 1
    cls, x, y = dets[0]
    assert cls == 0
    assert abs(x - 7 * STRIDE) < 1e-4
    assert abs(y - 5 * STRIDE) < 1e-4


def test_center_distance_error_zero_for_perfect_prediction():
    heatmap_logits = torch.full((1, 8, 8), -10.0)
    heatmap_logits[0, 2, 3] = 10.0
    offset = torch.zeros(2, 8, 8)
    gt_centers = torch.tensor([[3.0 * STRIDE, 2.0 * STRIDE]])
    gt_classes = torch.tensor([0])
    err = center_distance_error(heatmap_logits, offset, gt_centers, gt_classes)
    assert err < 1e-3


def test_stale_baseline_error_matches_manual_computation():
    stale = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    future = torch.tensor([[3.0, 4.0], [10.0, 0.0]])
    err = stale_baseline_error(stale, future)
    assert abs(err - 2.5) < 1e-4  # (5.0 + 0.0) / 2
