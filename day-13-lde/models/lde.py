"""
LDEModel: Mean-Teacher-style EMA self-training loop wiring together
BEVDetector (student/teacher), AdaptiveFeatureGate (bandwidth budget),
FoVAligner (warp + FoV filtering) and CurriculumScheduler (confidence
gating).

Student = the ego vehicle's detector, trained by gradient descent.
Teacher = an EMA copy of the student's weights (never touched by the
optimizer directly), playing the role of the collaborator's detector
whose pseudo-labels supervise the student -- Mean Teacher (Tarvainen &
Valpola, 2017) applied to collaborative-perception self-training.
"""

import copy

import torch
import torch.nn as nn

from models.detector import BEVDetector
from models.feature_sharing import AdaptiveFeatureGate
from models.fov_align import FoVAligner
from src.utils.curriculum import CurriculumScheduler
from src.utils.losses import class_balanced_bce_loss


class LDEModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        feat_channels: int = 64,
        budget_ratio: float = 0.4,
        fov_radius_cells: float = 16.0,
        ema_momentum: float = 0.99,
        curriculum_start: float = 0.90,
        curriculum_end: float = 0.50,
        curriculum_steps: int = 150,
    ):
        super().__init__()
        self.student = BEVDetector(in_channels, feat_channels)
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.feature_gate = AdaptiveFeatureGate(budget_ratio)
        self.fov_aligner = FoVAligner(fov_radius_cells)
        self.curriculum = CurriculumScheduler(curriculum_start, curriculum_end, curriculum_steps)
        self.ema_momentum = ema_momentum

    @torch.no_grad()
    def update_teacher(self) -> None:
        """EMA update: teacher <- momentum*teacher + (1-momentum)*student.
        BatchNorm running stats are copied directly (standard Mean-Teacher
        practice) since EMA-averaging running stats themselves tends to
        lag badly behind the student's actual current statistics."""
        m = self.ema_momentum
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.mul_(m).add_(s_param.detach(), alpha=1 - m)
        for t_buf, s_buf in zip(self.teacher.buffers(), self.student.buffers()):
            t_buf.copy_(s_buf)

    @torch.no_grad()
    def build_pseudo_labels(self, collab_bev: torch.Tensor, offset_dx: torch.Tensor,
                             offset_dy: torch.Tensor, step: int, dtheta: torch.Tensor = None) -> dict:
        """
        Decode pseudo-labels the student will be supervised with.

        `offset_dx`, `offset_dy` (B,): the collaborator's position minus
        the ego's position, in grid cells, i.e. collaborator_local_frame
        = world_frame - offset. FoVAligner warps INTO the ego/world frame,
        so we pass it the negated offset (see models/fov_align.py).

        BUG #1 FIX: pseudo-label confidence is decoded through the
        TEACHER's own (EMA-tracked, already-trained) cls_head -- never a
        separate, freshly-initialized head. A fresh head's logits sit at
        ~0 forever (sigmoid(0)=0.5 for everything, untrained), which can
        never clear the curriculum's 0.90 starting threshold in a
        meaningful, object-selective way -- `supervised_cells` would stay
        pinned near 0 (or admit noise indiscriminately) no matter how good
        the teacher's underlying features actually are.

        BUG #3 FIX: AdaptiveFeatureGate only zeroes non-selected cells; it
        never reprojects survivors through any conv. teacher.cls_head is
        applied directly to the gated+warped features, so it sees the
        same feature basis it was EMA-trained on (mixed only with zeros
        in place of unselected/out-of-frame cells), not a scrambled one.
        """
        teacher_feat = self.teacher.extract_features(collab_bev)             # (B, C, H, W)
        gated_feat, gate_mask = self.feature_gate(teacher_feat)              # spatial-only, no conv

        dx_cells, dy_cells = -offset_dx, -offset_dy
        warped_feat, fov_mask = self.fov_aligner(gated_feat, dx_cells, dy_cells, dtheta)
        warped_gate_mask = self.fov_aligner.warp_mask(gate_mask, dx_cells, dy_cells, dtheta)

        pseudo_logits = self.teacher.cls_head(warped_feat)                   # teacher's own head
        pseudo_conf = torch.sigmoid(pseudo_logits)                           # (B, 1, H, W)

        threshold = self.curriculum.get_threshold(step)
        # eligible only if the cell (a) survived the bandwidth budget AND
        # (b) lands inside the ego's own trusted FoV after warping.
        eligible_mask = (warped_gate_mask > 0.5).to(pseudo_conf.dtype) * fov_mask
        confident_mask = (pseudo_conf > threshold).to(pseudo_conf.dtype) * eligible_mask
        pseudo_labels = (pseudo_conf > 0.5).to(pseudo_conf.dtype)

        return {
            "pseudo_labels": pseudo_labels,
            "confident_mask": confident_mask,
            "pseudo_conf": pseudo_conf,
            "eligible_mask": eligible_mask,
            "threshold": threshold,
            "supervised_cells": confident_mask.sum().item(),
        }

    def self_training_loss(self, student_cls_logits: torch.Tensor, pseudo_labels: torch.Tensor,
                            confident_mask: torch.Tensor) -> torch.Tensor:
        """
        Class-balanced BCE restricted to confident, in-FoV cells only.

        BUG #4 FIX (minor, narrower than #1-#3): if the collaborator is
        fully out of range (or nothing clears the curriculum threshold),
        confident_mask sums to zero. A naive `loss_sum / supervised_cells`
        normalization would divide by zero there. class_balanced_bce_loss
        itself already guards this per-branch (an absent class contributes
        0, not NaN); here we additionally short-circuit to an explicit
        zero loss (that still participates in the autograd graph, so
        backward() stays valid) when there is nothing to supervise on at
        all, rather than relying on an empty-tensor edge case downstream.
        """
        num_supervised = confident_mask.sum()
        if num_supervised.item() == 0:
            return student_cls_logits.sum() * 0.0

        mask_bool = confident_mask.squeeze(1) > 0.5             # (B, H, W)
        logits_sel = student_cls_logits.squeeze(1)[mask_bool]    # (N,)
        labels_sel = pseudo_labels.squeeze(1)[mask_bool]         # (N,)
        return class_balanced_bce_loss(logits_sel, labels_sel)

    def forward(self, ego_bev: torch.Tensor) -> dict:
        return self.student(ego_bev)
