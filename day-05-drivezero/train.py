"""
Smoke-test training loop for DriveZero: one PPO teacher update + one
distillation update per step, on synthetic data, then a value-guided
action-search shape check.

Usage:
    python train.py --steps 20
"""
import argparse

import torch

from src.models.drivezero import (
    DriveVFMConfig, DriveVFM, PrivilegedTeacherPolicy, ppo_clipped_update,
    CameraOnlyStudentPolicy, distillation_loss, value_guided_action_search,
)

SOURCE_DIMS = {"dinov3": 96, "siglip2": 112, "sam": 64, "depth_anything_v2": 48}
N_TOKENS_PER_SOURCE = 8
MODEL_DIM = 64
PRIVILEGED_DIM = 32
GOAL_DIM = 16
ACTION_DIM = 2  # e.g. (steering, acceleration)


def synthetic_features(batch_size):
    return {
        name: torch.randn(batch_size, N_TOKENS_PER_SOURCE, dim)
        for name, dim in SOURCE_DIMS.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    B = 8

    vfm_cfg = DriveVFMConfig(source_dims=SOURCE_DIMS, model_dim=MODEL_DIM, n_heads=4, n_fusion_layers=2)
    drive_vfm = DriveVFM(vfm_cfg)
    teacher = PrivilegedTeacherPolicy(MODEL_DIM, PRIVILEGED_DIM, ACTION_DIM, hidden=64)
    student = CameraOnlyStudentPolicy(MODEL_DIM, GOAL_DIM, ACTION_DIM, hidden=32)

    params = list(drive_vfm.parameters()) + list(teacher.parameters()) + list(student.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    for step in range(args.steps):
        features = synthetic_features(B)
        unified_tokens = drive_vfm(features)                        # [B, T, D]
        privileged_state = torch.randn(B, PRIVILEGED_DIM)
        goal_embed = torch.randn(B, GOAL_DIM)

        # ---- one PPO teacher update (synthetic rollout data) ----
        with torch.no_grad():
            actions, old_log_probs, _ = teacher.act(unified_tokens, privileged_state)
        advantages = torch.randn(B)
        returns = torch.randn(B)
        ppo_out = ppo_clipped_update(teacher, unified_tokens, privileged_state,
                                      actions, old_log_probs, advantages, returns)

        # ---- one distillation update (student <- teacher, on the same batch) ----
        with torch.no_grad():
            teacher_dist, _ = teacher(unified_tokens, privileged_state)
        distill_out = distillation_loss(student, teacher_dist, unified_tokens, goal_embed)

        total_loss = ppo_out["ppo_loss"] + distill_out
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"step {step:03d}  ppo_loss={ppo_out['ppo_loss'].item():.4f}  distill_loss={distill_out.item():.4f}")

    # ---- value-guided test-time action search shape check ----
    with torch.no_grad():
        features = synthetic_features(B)
        unified_tokens = drive_vfm(features)
        goal_embed = torch.randn(B, GOAL_DIM)
        privileged_estimate = torch.randn(B, PRIVILEGED_DIM)
        selected = value_guided_action_search(student, teacher, unified_tokens, goal_embed,
                                               privileged_estimate, k=8)
    print("value_guided_action_search selected action shape:", tuple(selected.shape))

    n_vfm = sum(p.numel() for p in drive_vfm.parameters())
    n_teacher = sum(p.numel() for p in teacher.parameters())
    n_student = sum(p.numel() for p in student.parameters())
    print(f"param counts -- DriveVFM: {n_vfm:,} | teacher: {n_teacher:,} | student: {n_student:,}")
    print(f"reached --steps limit ({args.steps}), stopping (smoke test mode)")


if __name__ == "__main__":
    main()
