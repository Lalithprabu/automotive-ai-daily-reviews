"""Coarse-to-fine trajectory supervision: every refinement stage is supervised
against the same ground truth, with linearly increasing weight from the
initializer's coarse guess to the final fine-stage prediction. This is this
project's own default (a standard coarse-to-fine training recipe), not
paper-sourced -- the paper's exact loss/weighting was not recoverable."""
import torch
import torch.nn.functional as F


def coarse_to_fine_loss(stage_trajectories, future_gt):
    n_stages = len(stage_trajectories)
    weights = torch.linspace(0.3, 1.0, n_stages)
    total = 0.0
    per_stage = []
    for w, traj in zip(weights, stage_trajectories):
        l = F.smooth_l1_loss(traj, future_gt)
        per_stage.append(l.detach())
        total = total + w * l
    return total / weights.sum(), per_stage


def ade_fde(pred, gt):
    """Average / final displacement error, in the same units as pred/gt (meters)."""
    disp = torch.norm(pred - gt, dim=-1)  # (B, T)
    ade = disp.mean().item()
    fde = disp[:, -1].mean().item()
    return ade, fde
