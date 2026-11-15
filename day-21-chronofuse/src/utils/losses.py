"""
CenterNet-style target construction and loss functions.

Targets (heatmap / size / offset) are built at stride 4 from a list of
per-sample object centers, sizes, and classes — using the standard
gaussian-splat heatmap convention (Law & Deng, CornerNet / Zhou et al.,
CenterNet) so that nearby-but-imperfect center predictions are still
rewarded during training, which matters a lot here since we're training
the model to predict a *future*, not-yet-observed position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 4


def gaussian_radius(size_wh: torch.Tensor, min_overlap: float = 0.7) -> float:
    """Standard CornerNet gaussian-radius heuristic, adapted for a single
    box, returning a radius (in stride-4 pixels) such that a same-class
    box shifted by that radius still has >= min_overlap IoU with the
    original box."""
    w, h = size_wh[0].item(), size_wh[1].item()
    w, h = max(w / STRIDE, 1.0), max(h / STRIDE, 1.0)

    a1, b1, c1 = 1, (w + h), w * h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = max(b1 ** 2 - 4 * a1 * c1, 0) ** 0.5
    r1 = (b1 + sq1) / 2

    return max(1.0, r1)


def draw_gaussian(heatmap: torch.Tensor, center_xy, radius: float):
    """Splats a 2D gaussian peak of the given radius onto `heatmap`
    (in-place, taking the elementwise max with any existing peak so
    overlapping objects don't cancel each other out)."""
    h, w = heatmap.shape
    cx, cy = center_xy
    sigma = radius / 3.0
    r_int = max(int(radius), 1)

    x0, x1 = max(0, int(cx) - r_int), min(w, int(cx) + r_int + 1)
    y0, y1 = max(0, int(cy) - r_int), min(h, int(cy) + r_int + 1)
    if x0 >= x1 or y0 >= y1:
        return

    yy, xx = torch.meshgrid(
        torch.arange(y0, y1, dtype=torch.float32),
        torch.arange(x0, x1, dtype=torch.float32),
        indexing="ij",
    )
    gauss = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2 + 1e-6))
    heatmap[y0:y1, x0:x1] = torch.maximum(heatmap[y0:y1, x0:x1], gauss)


def build_targets(centers_list, sizes_list, classes_list, out_h: int, out_w: int, num_classes: int):
    """Builds dense (heatmap, size, offset, positive-mask) targets for a
    batch from ragged per-sample lists of future centers/sizes/classes
    (in full-resolution pixel coordinates).

    Returns tensors of shape:
        heatmap: [B, num_classes, out_h, out_w]
        size:    [B, 2, out_h, out_w]
        offset:  [B, 2, out_h, out_w]
        pos_mask:[B, out_h, out_w]  bool, True at object-center pixels
    """
    b = len(centers_list)
    heatmap = torch.zeros(b, num_classes, out_h, out_w)
    size_t = torch.zeros(b, 2, out_h, out_w)
    offset_t = torch.zeros(b, 2, out_h, out_w)
    pos_mask = torch.zeros(b, out_h, out_w, dtype=torch.bool)

    for i in range(b):
        centers, sizes, classes = centers_list[i], sizes_list[i], classes_list[i]
        for j in range(centers.shape[0]):
            cx, cy = (centers[j] / STRIDE).tolist()
            if not (0 <= cx < out_w and 0 <= cy < out_h):
                continue
            cls = int(classes[j].item())
            radius = gaussian_radius(sizes[j])
            draw_gaussian(heatmap[i, cls], (cx, cy), radius)

            ix, iy = int(cx), int(cy)
            size_t[i, :, iy, ix] = sizes[j]
            offset_t[i, :, iy, ix] = torch.tensor([cx - ix, cy - iy])
            pos_mask[i, iy, ix] = True
            # BUGFIX (caught by an overfit-a-tiny-batch sanity check, see
            # daily-review doc): object centers are continuous, so the
            # gaussian splat above almost never lands exactly on 1.0 at a
            # discrete pixel -- `focal_loss`'s `target_heatmap.eq(1.0)`
            # positive mask would then select ~zero pixels almost always,
            # silently collapsing the heatmap loss to ~0 while predictions
            # stayed near-zero everywhere. Force the exact discretized
            # center pixel to 1.0 (the standard CenterNet convention) so
            # the positive mask is well-defined and matches `pos_mask`.
            heatmap[i, cls, iy, ix] = 1.0

    return heatmap, size_t, offset_t, pos_mask


def focal_loss(pred_logits: torch.Tensor, target_heatmap: torch.Tensor,
                alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Penalty-reduced pixelwise focal loss (CornerNet/CenterNet variant).

    pred_logits, target_heatmap: [B, num_classes, H, W]. `target_heatmap`
    values are in [0, 1] (gaussian-splatted, 1.0 exactly at true centers).
    """
    pred = torch.sigmoid(pred_logits).clamp(1e-4, 1 - 1e-4)
    pos_mask = target_heatmap.eq(1.0).float()
    neg_mask = 1.0 - pos_mask

    neg_weights = torch.pow(1 - target_heatmap, beta)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_mask

    num_pos = pos_mask.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


class DetectionLoss(nn.Module):
    """Combined heatmap focal loss + L1 size/offset regression loss
    (regression terms computed only at true object-center pixels)."""

    def __init__(self, size_weight: float = 0.1, offset_weight: float = 1.0):
        super().__init__()
        self.size_weight = size_weight
        self.offset_weight = offset_weight

    def forward(self, pred_heatmap, pred_size, pred_offset,
                target_heatmap, target_size, target_offset, pos_mask):
        hm_loss = focal_loss(pred_heatmap, target_heatmap)

        if pos_mask.any():
            mask = pos_mask.unsqueeze(1).expand_as(pred_size)  # [B, 2, H, W]
            size_loss = F.l1_loss(pred_size[mask], target_size[mask])
            offset_loss = F.l1_loss(pred_offset[mask], target_offset[mask])
        else:
            size_loss = pred_size.sum() * 0.0
            offset_loss = pred_offset.sum() * 0.0

        total = hm_loss + self.size_weight * size_loss + self.offset_weight * offset_loss
        return total, {
            "heatmap_loss": hm_loss.item(),
            "size_loss": float(size_loss.item() if torch.is_tensor(size_loss) else size_loss),
            "offset_loss": float(offset_loss.item() if torch.is_tensor(offset_loss) else offset_loss),
        }
