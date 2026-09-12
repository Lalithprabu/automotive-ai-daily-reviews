"""Shape + backward-pass + PPO + distillation smoke tests for DriveZero."""
import torch

from src.models.drivezero import (
    DriveVFMConfig, DriveVFM, PrivilegedTeacherPolicy, ppo_clipped_update,
    CameraOnlyStudentPolicy, distillation_loss, value_guided_action_search,
)

SOURCE_DIMS = {"dinov3": 32, "siglip2": 40, "sam": 24, "depth_anything_v2": 16}


def _setup(b=4, n_tok=4, model_dim=32, priv_dim=16, goal_dim=8, action_dim=2):
    cfg = DriveVFMConfig(source_dims=SOURCE_DIMS, model_dim=model_dim, n_heads=4, n_fusion_layers=2)
    drive_vfm = DriveVFM(cfg)
    teacher = PrivilegedTeacherPolicy(model_dim, priv_dim, action_dim, hidden=32)
    student = CameraOnlyStudentPolicy(model_dim, goal_dim, action_dim, hidden=16)

    features = {name: torch.randn(b, n_tok, dim) for name, dim in SOURCE_DIMS.items()}
    return drive_vfm, teacher, student, features


def test_drivevfm_adapter_shape():
    drive_vfm, _, _, features = _setup()
    tokens = drive_vfm(features)
    n_expected = 4 * len(SOURCE_DIMS)  # n_tok * n_sources
    assert tokens.shape == (4, n_expected, 32)


def test_drivevfm_forward_backward():
    drive_vfm, _, _, features = _setup()
    tokens = drive_vfm(features)
    tokens.sum().backward()
    grads = [p.grad for p in drive_vfm.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_teacher_forward_and_act():
    drive_vfm, teacher, _, features = _setup()
    tokens = drive_vfm(features)
    privileged_state = torch.randn(4, 16)
    action, log_prob, value = teacher.act(tokens, privileged_state)
    assert action.shape == (4, 2)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)


def test_ppo_clipped_update():
    drive_vfm, teacher, _, features = _setup()
    tokens = drive_vfm(features).detach()
    privileged_state = torch.randn(4, 16)
    with torch.no_grad():
        actions, old_log_probs, _ = teacher.act(tokens, privileged_state)
    advantages = torch.randn(4)
    returns = torch.randn(4)

    out = ppo_clipped_update(teacher, tokens, privileged_state, actions,
                              old_log_probs, advantages, returns)
    out["ppo_loss"].backward()
    assert torch.isfinite(out["ppo_loss"])
    assert any(p.grad is not None for p in teacher.parameters())


def test_student_forward_and_distillation():
    drive_vfm, teacher, student, features = _setup()
    tokens = drive_vfm(features).detach()
    privileged_state = torch.randn(4, 16)
    goal_embed = torch.randn(4, 8)

    with torch.no_grad():
        teacher_dist, _ = teacher(tokens, privileged_state)
    loss = distillation_loss(student, teacher_dist, tokens, goal_embed)
    loss.backward()
    assert torch.isfinite(loss)


def test_sample_k_shape():
    drive_vfm, _, student, features = _setup()
    tokens = drive_vfm(features).detach()
    goal_embed = torch.randn(4, 8)
    samples = student.sample_k(tokens, goal_embed, k=5)
    assert samples.shape == (4, 5, 2)


def test_value_guided_action_search_shape_and_selection():
    drive_vfm, teacher, student, features = _setup()
    tokens = drive_vfm(features).detach()
    goal_embed = torch.randn(4, 8)
    privileged_estimate = torch.randn(4, 16)

    selected = value_guided_action_search(student, teacher, tokens, goal_embed,
                                           privileged_estimate, k=6)
    assert selected.shape == (4, 2)
    assert torch.isfinite(selected).all()


def test_config_mismatch_source_keys_raises():
    cfg = DriveVFMConfig(source_dims=SOURCE_DIMS, model_dim=32, n_heads=4, n_fusion_layers=1)
    drive_vfm = DriveVFM(cfg)
    bad_features = {"dinov3": torch.randn(2, 4, 32)}  # missing sources
    try:
        drive_vfm(bad_features)
        assert False, "expected an assertion error for missing modality sources"
    except AssertionError:
        pass


def test_param_count_sanity():
    drive_vfm, teacher, student, _ = _setup()
    assert sum(p.numel() for p in drive_vfm.parameters()) > 0
    assert sum(p.numel() for p in teacher.parameters()) > 0
    assert sum(p.numel() for p in student.parameters()) > 0
