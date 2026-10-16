import torch
def occ_iou(pred, gt):            # class-agnostic occupied-vs-empty IoU
    p, g = pred > 0, gt > 0
    return ((p & g).sum() / ((p | g).sum().clamp(min=1))).item()
def class_ious(pred, gt, nc=5):
    out = []
    for c in range(1, nc):
        p, g = pred == c, gt == c
        u = (p | g).sum().item(); out.append(float('nan') if u == 0 else (p & g).sum().item() / u)
    return out
def weighted_miou(pred, gt, nc=5):  # weighted by class voxel frequency (excludes empty)
    num = den = 0.0
    for c in range(1, nc):
        p, g = pred == c, gt == c; u = (p | g).sum().item(); w = g.sum().item()
        if u: num += w * (p & g).sum().item() / u; den += w
    return num / max(den, 1)
def ebkl(pred, ref_pred, nc=5, eps=1e-6):
    """Environment-Based KL (our simplified version): KL(class histogram of adverse pred || normal pred), non-empty classes."""
    h = torch.stack([(pred == c).sum() for c in range(1, nc)]).float() + eps; h /= h.sum()
    r = torch.stack([(ref_pred == c).sum() for c in range(1, nc)]).float() + eps; r /= r.sum()
    return (h * (h / r).log()).sum().item()
