"""
Loss functions for LDE self-training.

class_balanced_bce_loss is the loss actually used by the training loop
(models/lde.py -> LDEModel.self_training_loss and train.py's supervised
pretraining phase). sigmoid_focal_loss is implemented for reference /
comparison only -- see README "Implementation notes" (bug #2) for why
plain mean-reduced BCE and even focal loss are NOT what is used here.
"""

import torch
import torch.nn.functional as F


def class_balanced_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Class-balanced binary cross-entropy.

    Computes per-cell BCE-with-logits, then averages the POSITIVE-cell
    losses and the NEGATIVE-cell losses SEPARATELY, and returns the SUM of
    those two averages (never a single mean over all cells combined).

    Why this matters (bug #2): on a large BEV grid with very few true
    objects (e.g. ~6 objects on a 32x32 = 1024-cell grid, ~99.4%
    background), a plain `F.binary_cross_entropy_with_logits(reduction=
    "mean")` is dominated by the ~1000 easy negative cells. Gradient
    descent can drive that single mean down just by staying confidently
    negative everywhere -- it never has to pay much to also become
    confident on the handful of positive cells, because they're diluted
    into the mean by a factor of ~150:1. Averaging positives and negatives
    separately, then summing, gives the rare positive cells equal weight
    to the whole mass of negative cells regardless of the imbalance ratio.

    Args:
        logits:  raw (pre-sigmoid) predictions, any shape
        targets: same shape, binary {0., 1.}
    Returns:
        scalar loss tensor (0-d), safe when one of the two classes is
        entirely absent from this batch (contributes 0, not NaN).
    """
    logits = logits.reshape(-1)
    targets = targets.reshape(-1).float()

    per_cell = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    pos_mask = targets > 0.5
    neg_mask = ~pos_mask

    pos_loss = per_cell[pos_mask].mean() if pos_mask.any() else per_cell.new_zeros(())
    neg_loss = per_cell[neg_mask].mean() if neg_mask.any() else per_cell.new_zeros(())

    return pos_loss + neg_loss


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Standard sigmoid focal loss (Lin et al., 2017, RetinaNet).

    Kept in this module for reference / comparison against
    class_balanced_bce_loss. NOT the loss used by LDEModel's training
    loop in this reconstruction (see README implementation notes).
    """
    logits = logits.reshape(-1)
    targets = targets.reshape(-1).float()

    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t).clamp(min=0.0) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss
