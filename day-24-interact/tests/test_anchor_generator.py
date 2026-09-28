import torch

from src.models.anchor_generator import AnchorGenerator


def test_generates_k_anchors():
    gen = AnchorGenerator()
    anchors = gen.generate()
    assert len(anchors) == 5
    assert gen.k == 5


def test_anchor_names_match_config_defaults():
    gen = AnchorGenerator()
    anchors = gen.generate()
    names = [a.name for a in anchors]
    assert names == ["aggressive_merge", "accelerate_past", "hold_lane_delay",
                      "cautious_yield", "decelerate_follow"]


def test_anchors_are_distinct_intent_vectors():
    gen = AnchorGenerator()
    anchors = gen.generate()
    conds = torch.stack([a.cond for a in anchors])
    # every pair of anchors must be a genuinely different intent (not degenerate/duplicated)
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            assert torch.norm(conds[i] - conds[j]) > 0.1


def test_assertiveness_is_monotonic_across_the_canonical_ordering():
    """The 5 canonical anchors were designed to span assertive -> cautious;
    this checks that ordering actually holds after geometry scaling."""
    gen = AnchorGenerator()
    anchors = gen.generate()
    scores = [a.assertiveness for a in anchors]
    assert scores == sorted(scores, reverse=True), f"expected monotonically decreasing assertiveness, got {scores}"


def test_geometry_scaling_changes_anchor_magnitudes():
    """A tighter merge gap (more urgency) should scale speed-related anchor components."""
    gen = AnchorGenerator()
    wide_gap = gen.generate(gap_to_merge=6.0)
    tight_gap = gen.generate(gap_to_merge=1.0)
    # aggressive_merge anchor's speed component should differ between the two geometries
    assert abs(wide_gap[0].cond[1].item() - tight_gap[0].cond[1].item()) > 1e-4


def test_anchor_center_bank_shape():
    gen = AnchorGenerator()
    bank = gen.anchor_center_bank()
    assert bank.shape == (5, 2)
