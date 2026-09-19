"""
17 tests for the LDE reconstruction, including explicit regression tests
for the three critical bugs (+1 narrower bug) described in the README:

  bug #1 (untrained pseudo-label head)   -> test_bug1_*
  bug #2 (class-imbalance / mean BCE)    -> test_bug2_*
  bug #3 (feature-gate channel scramble) -> test_bug3_*
  bug #4 (div-by-zero on empty support)  -> test_bug4_*
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn.functional as F

from models.detector import BEVDetector
from models.feature_sharing import AdaptiveFeatureGate
from models.fov_align import FoVAligner
from models.lde import LDEModel
from src.utils.curriculum import CurriculumScheduler
from src.utils.losses import class_balanced_bce_loss, sigmoid_focal_loss
from src.utils.data import generate_scene, generate_batch


torch.manual_seed(0)


# ---------------------------------------------------------------------------
# BEVDetector
# ---------------------------------------------------------------------------

def test_bev_detector_output_shapes():
    net = BEVDetector(in_channels=8, feat_channels=64)
    x = torch.randn(3, 8, 32, 32)
    out = net(x)
    assert out["features"].shape == (3, 64, 32, 32)
    assert out["cls_logits"].shape == (3, 1, 32, 32)
    assert out["reg"].shape == (3, 4, 32, 32)


def test_bev_detector_extract_features_consistent():
    net = BEVDetector(in_channels=8, feat_channels=64)
    net.eval()
    x = torch.randn(2, 8, 32, 32)
    feat_direct = net.extract_features(x)
    feat_via_forward = net(x)["features"]
    assert torch.allclose(feat_direct, feat_via_forward, atol=1e-6)


# ---------------------------------------------------------------------------
# AdaptiveFeatureGate
# ---------------------------------------------------------------------------

def test_feature_gate_budget_ratio_respected():
    gate = AdaptiveFeatureGate(budget_ratio=0.25)
    feat = torch.randn(2, 16, 32, 32)
    _, mask = gate(feat)
    expected_k = round(32 * 32 * 0.25)
    assert mask.sum(dim=(1, 2, 3)).tolist() == [expected_k, expected_k]


def test_feature_gate_has_no_learnable_params():
    """Bug #3 (structural): the gate must be purely spatial -- zero
    learnable parameters, so it cannot possibly contain a channel-mixing
    conv like an untrained `channel_reduce`."""
    gate = AdaptiveFeatureGate(budget_ratio=0.4)
    assert sum(p.numel() for p in gate.parameters()) == 0


def test_feature_gate_zeroes_non_selected_cells():
    gate = AdaptiveFeatureGate(budget_ratio=0.3)
    feat = torch.randn(1, 8, 16, 16)
    gated, mask = gate(feat)
    zero_locations = (mask == 0).expand_as(gated)
    assert torch.all(gated[zero_locations] == 0.0)
    kept_locations = (mask == 1).expand_as(gated)
    assert torch.allclose(gated[kept_locations], feat[kept_locations])


# ---------------------------------------------------------------------------
# FoVAligner
# ---------------------------------------------------------------------------

def test_fov_aligner_zero_offset_is_identity_within_fov():
    aligner = FoVAligner(fov_radius_cells=100.0)  # big enough to cover whole grid
    feat = torch.randn(1, 4, 16, 16)
    zero = torch.zeros(1)
    warped, fov_mask = aligner(feat, zero, zero)
    assert torch.allclose(warped, feat, atol=1e-5)
    assert torch.all(fov_mask == 1.0)


def test_fov_aligner_range_mask_matches_circle_area():
    H = W = 32
    aligner = FoVAligner(fov_radius_cells=10.0)
    mask = aligner.range_mask((1, 1, H, W), device="cpu", dtype=torch.float32)
    count = mask.sum().item()
    expected = math.pi * 10.0 ** 2
    # discrete grid approximation of a continuous disk area -> loose tolerance
    assert abs(count - expected) / expected < 0.15


def test_fov_aligner_recovers_known_object_after_warp():
    """The collaborator's local-frame signal, warped with the correct
    (negated) offset, must land back on the true world-frame object cell."""
    g = torch.Generator().manual_seed(42)
    scene = generate_scene(generator=g, collab_range=100.0)  # ensure visible
    collab_bev = scene["collab_bev"].unsqueeze(0)
    dx, dy = scene["dx"], scene["dy"]

    aligner = FoVAligner(fov_radius_cells=100.0)
    warped, _ = aligner(collab_bev, torch.tensor([-dx]), torch.tensor([-dy]))

    saliency = warped[0].norm(dim=0)
    peak = (saliency == saliency.max()).nonzero()[0]
    world_obj_cells = scene["world_labels"].nonzero()

    dists = ((world_obj_cells.float() - peak.float()) ** 2).sum(dim=1)
    assert dists.min().item() <= 2.0  # within 1 cell (allowing bilinear blur)


# ---------------------------------------------------------------------------
# CurriculumScheduler
# ---------------------------------------------------------------------------

def test_curriculum_threshold_bounds_and_monotonic():
    sched = CurriculumScheduler(start=0.90, end=0.50, total_steps=100)
    assert math.isclose(sched.get_threshold(0), 0.90, abs_tol=1e-6)
    assert math.isclose(sched.get_threshold(100), 0.50, abs_tol=1e-6)
    assert math.isclose(sched.get_threshold(1000), 0.50, abs_tol=1e-6)  # clamped
    thresholds = [sched.get_threshold(s) for s in range(0, 101, 5)]
    assert all(thresholds[i] >= thresholds[i + 1] - 1e-9 for i in range(len(thresholds) - 1))


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def test_class_balanced_bce_basic_value():
    logits = torch.tensor([5.0, -5.0, 0.0])
    targets = torch.tensor([1.0, 0.0, 1.0])
    loss = class_balanced_bce_loss(logits, targets)
    # manual: pos cells = {0, 2}, neg cells = {1}
    pos = F.binary_cross_entropy_with_logits(torch.tensor([5.0, 0.0]), torch.tensor([1.0, 1.0]))
    neg = F.binary_cross_entropy_with_logits(torch.tensor([-5.0]), torch.tensor([0.0]))
    assert torch.isclose(loss, pos + neg, atol=1e-5)


def test_bug2_class_balanced_beats_plain_mean_bce_on_imbalanced_scene():
    """Bug #2 regression: on the SAME synthetic imbalanced scene (~6
    positive cells / 1024, ~99.4% background), training a shared-weight
    detector with plain mean-reduced BCE collapses positive-cell
    confidence toward 0 (the bias term gets dragged negative by the huge
    mass of easy negatives); class_balanced_bce_loss does not."""
    g = torch.Generator().manual_seed(7)
    scene = generate_scene(generator=g)
    x = scene["collab_bev"].unsqueeze(0)
    y = scene["collab_labels"].unsqueeze(0)
    assert y.sum().item() > 0, "scene must contain at least one positive cell"

    torch.manual_seed(123)
    init_state = BEVDetector(in_channels=8, feat_channels=64).state_dict()

    def train(loss_fn, steps=60, lr=0.05):
        net = BEVDetector(in_channels=8, feat_channels=64)
        net.load_state_dict(init_state)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            logits = net(x)["cls_logits"].squeeze(1)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
        with torch.no_grad():
            final_logits = net(x)["cls_logits"].squeeze(1)
            conf = torch.sigmoid(final_logits)
            pos_conf = conf[y > 0.5].mean().item()
        return pos_conf

    cb_pos_conf = train(class_balanced_bce_loss)
    plain_pos_conf = train(lambda lg, tg: F.binary_cross_entropy_with_logits(lg, tg, reduction="mean"))

    assert cb_pos_conf > 0.5, f"class-balanced BCE should confidently fire on positives, got {cb_pos_conf}"
    assert cb_pos_conf > plain_pos_conf + 0.2, (
        f"class-balanced ({cb_pos_conf:.3f}) should clearly beat plain mean BCE ({plain_pos_conf:.3f})"
    )


def test_sigmoid_focal_loss_reference_runs():
    logits = torch.randn(10, requires_grad=True)
    targets = (torch.rand(10) > 0.9).float()
    loss = sigmoid_focal_loss(logits, targets)
    assert loss.dim() == 0
    loss.backward()
    assert logits.grad is not None


# ---------------------------------------------------------------------------
# LDEModel / bug regressions
# ---------------------------------------------------------------------------

def _tiny_model():
    return LDEModel(in_channels=8, feat_channels=16, budget_ratio=1.0,
                     fov_radius_cells=100.0, curriculum_steps=100)


def test_bug1_pseudo_confidence_tracks_teacher_head_not_stuck_at_half():
    """Bug #1 regression: pseudo-label confidence must be decoded through
    the TEACHER's own cls_head. If a fresh, untrained, separate head were
    used instead, its logits would sit at ~0 (sigmoid ~0.5) regardless of
    how the teacher's real head is configured. Here we directly manipulate
    teacher.cls_head and confirm pseudo_conf follows it."""
    model = _tiny_model()
    collab_bev = torch.randn(2, 8, 32, 32)
    dx = torch.zeros(2)
    dy = torch.zeros(2)

    with torch.no_grad():
        model.teacher.cls_head.weight.zero_()
        model.teacher.cls_head.bias.fill_(10.0)
    out_hi = model.build_pseudo_labels(collab_bev, dx, dy, step=0)
    assert out_hi["pseudo_conf"].mean().item() > 0.999

    with torch.no_grad():
        model.teacher.cls_head.weight.zero_()
        model.teacher.cls_head.bias.fill_(-10.0)
    out_lo = model.build_pseudo_labels(collab_bev, dx, dy, step=0)
    assert out_lo["pseudo_conf"].mean().item() < 0.001

    # A broken "fresh head" implementation would give ~0.5 in both cases.
    assert abs(out_hi["pseudo_conf"].mean().item() - 0.5) > 0.4
    assert abs(out_lo["pseudo_conf"].mean().item() - 0.5) > 0.4


def test_bug3_feature_gate_preserves_teacher_confidence_when_budget_full():
    """Bug #3 regression: with budget_ratio=1.0 (nothing dropped) and zero
    ego/collaborator offset (identity warp), pseudo_logits must exactly
    match teacher.cls_head applied directly to the teacher's raw features
    -- proving no channel-mixing conv was inserted between the gate and
    the head. A stray untrained 1x1 conv would break this equality."""
    model = _tiny_model()
    collab_bev = torch.randn(1, 8, 32, 32)
    dx = torch.zeros(1)
    dy = torch.zeros(1)

    with torch.no_grad():
        raw_feat = model.teacher.extract_features(collab_bev)
        raw_logits = model.teacher.cls_head(raw_feat)

    out = model.build_pseudo_labels(collab_bev, dx, dy, step=0)
    pseudo_logits = torch.logit(out["pseudo_conf"].clamp(1e-6, 1 - 1e-6))

    assert torch.allclose(raw_logits, pseudo_logits, atol=1e-4)


def test_bug4_self_training_loss_handles_zero_supervised_cells():
    """Bug #4 regression: an empty confident_mask (e.g. collaborator fully
    out of range) must not divide by zero / produce NaN, and must still
    support backward()."""
    model = _tiny_model()
    student_logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    pseudo_labels = torch.zeros(2, 1, 8, 8)
    confident_mask = torch.zeros(2, 1, 8, 8)  # nothing eligible at all

    loss = model.self_training_loss(student_logits, pseudo_labels, confident_mask)
    assert torch.isfinite(loss).all()
    assert loss.item() == 0.0
    loss.backward()  # must not raise


def test_ema_update_moves_teacher_toward_student():
    model = _tiny_model()
    momentum = model.ema_momentum
    with torch.no_grad():
        s_param = next(model.student.parameters())
        t_param = next(model.teacher.parameters())
        t_before = t_param.clone()
        s_param.add_(1.0)  # move student far from teacher
    model.update_teacher()
    t_param = next(model.teacher.parameters())
    expected = momentum * t_before + (1 - momentum) * (t_before + 1.0)
    assert torch.allclose(t_param, expected, atol=1e-5)


def test_lde_full_training_step_end_to_end_reduces_loss():
    """A short real training loop (student optimized against teacher
    pseudo-labels) should reduce the self-training loss over a handful of
    steps on a fixed synthetic scene."""
    torch.manual_seed(3)
    model = _tiny_model()
    # give the teacher a real, confidently-trained head so pseudo-labels
    # are meaningful from step 0 (mirrors train.py's pretrain phase).
    g = torch.Generator().manual_seed(9)
    batch = generate_batch(4, generator=g, collab_range=100.0)
    opt_pre = torch.optim.Adam(model.teacher.parameters(), lr=0.05)
    for p in model.teacher.parameters():
        p.requires_grad_(True)
    for _ in range(40):
        opt_pre.zero_grad()
        logits = model.teacher(batch["collab_bev"])["cls_logits"].squeeze(1)
        loss = class_balanced_bce_loss(logits, batch["collab_labels"])
        loss.backward()
        opt_pre.step()
    for p in model.teacher.parameters():
        p.requires_grad_(False)
    model.student.load_state_dict(model.teacher.state_dict())  # fresh copy point

    # perturb student so there's something to learn
    with torch.no_grad():
        for p in model.student.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    opt = torch.optim.Adam(model.student.parameters(), lr=0.01)
    losses = []
    for step in range(15):
        opt.zero_grad()
        ego_logits = model.student(batch["ego_bev"])["cls_logits"]
        pseudo = model.build_pseudo_labels(batch["collab_bev"], batch["dx"], batch["dy"], step=step)
        loss = model.self_training_loss(ego_logits, pseudo["pseudo_labels"], pseudo["confident_mask"])
        loss.backward()
        opt.step()
        model.update_teacher()
        losses.append(loss.item())

    assert losses[-1] <= losses[0] + 1e-6
