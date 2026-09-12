"""
FlowMatchingObjective -- the x-prediction flow-matching training loss plus
first/second-order temporal-difference regularization that discourages
waypoint jitter and abrupt acceleration changes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.planning_expert import PlanningExpert
from src.models.vlm_kv_cache import VLMKVCache


class FlowMatchingObjective(nn.Module):
    """
    Rectified-flow / flow-matching loss with x-prediction:

        x_t = (1 - t) * x0 + t * x1        (x0 ~ N(0, I) noise, x1 = ground truth)
        model(x_t, t)  ~=  x1              (direct clean-trajectory prediction)

    Plus first- and second-order temporal-difference regularization on the
    *predicted* trajectory, which the paper adds to discourage waypoint
    jitter and abrupt acceleration changes between consecutive waypoints.
    """

    def __init__(self, lambda_vel: float = 0.1, lambda_accel: float = 0.1):
        super().__init__()
        self.lambda_vel = lambda_vel
        self.lambda_accel = lambda_accel

    def forward(
        self,
        model: PlanningExpert,
        x1_gt: torch.Tensor,           # [B, T, 3] ground-truth clean waypoints
        kv_cache: VLMKVCache,
        instruction_embed: torch.Tensor,
        ego_state_embed: torch.Tensor,
    ) -> torch.Tensor:
        b = x1_gt.shape[0]
        device = x1_gt.device

        x0_noise = torch.randn_like(x1_gt)                       # [B, T, 3]
        t = torch.rand(b, device=device)                          # [B], uniform in [0, 1]
        t_ = t.view(b, 1, 1)                                       # broadcast over (T, 3)
        x_t = (1.0 - t_) * x0_noise + t_ * x1_gt                   # [B, T, 3] interpolated sample

        x1_pred = model(x_t, t, kv_cache, instruction_embed, ego_state_embed)  # [B, T, 3]

        flow_loss = F.mse_loss(x1_pred, x1_gt)

        # First-order difference (velocity) and second-order difference
        # (acceleration) smoothness penalties, computed along the time axis.
        pred_vel = x1_pred[:, 1:, :] - x1_pred[:, :-1, :]           # [B, T-1, 3]
        gt_vel = x1_gt[:, 1:, :] - x1_gt[:, :-1, :]                 # [B, T-1, 3]
        vel_loss = F.mse_loss(pred_vel, gt_vel)

        pred_accel = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]       # [B, T-2, 3]
        gt_accel = gt_vel[:, 1:, :] - gt_vel[:, :-1, :]             # [B, T-2, 3]
        accel_loss = F.mse_loss(pred_accel, gt_accel)

        return flow_loss + self.lambda_vel * vel_loss + self.lambda_accel * accel_loss
