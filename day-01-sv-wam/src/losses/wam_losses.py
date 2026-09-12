"""
Composite SV-WAM training loss:
  L = L_action (imitation loss on waypoints)
    + lambda_video    * L_video    (dense future-video supervision -- the
                                     "old vs new" innovation vs. plain
                                     imitation-only planners)
    + lambda_drivable * L_drivable (safety regularizer)
"""
import torch
import torch.nn.functional as F

from src.losses.drivable_area_regularizer import DrivableAreaRegularizer


def sv_wam_loss(
    pred_trajectory: torch.Tensor,
    gt_trajectory: torch.Tensor,
    pred_future_latents: torch.Tensor,
    gt_future_latents: torch.Tensor,
    sdf_map: torch.Tensor,
    map_resolution: float,
    map_origin: torch.Tensor,
    drivable_regularizer: DrivableAreaRegularizer,
    lambda_video: float = 1.0,
    lambda_drivable: float = 0.5,
) -> torch.Tensor:
    action_loss = F.smooth_l1_loss(pred_trajectory, gt_trajectory)
    video_loss = F.mse_loss(pred_future_latents, gt_future_latents)
    drivable_loss = drivable_regularizer(
        pred_trajectory, sdf_map, map_resolution, map_origin
    )
    return action_loss + lambda_video * video_loss + lambda_drivable * drivable_loss
