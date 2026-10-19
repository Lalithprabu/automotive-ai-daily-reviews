import torch


def confusion_miou(pred, gt, num_classes):
    """Pooled mIoU over classes present in gt∪pred. pred, gt: integer tensors (any shape)."""
    ious = []
    for c in range(num_classes):
        inter = ((pred == c) & (gt == c)).sum().item()
        union = ((pred == c) | (gt == c)).sum().item()
        if union > 0:
            ious.append(inter / union)
    return float(sum(ious) / max(len(ious), 1))


@torch.no_grad()
def evaluate(model, feats, occ, bs=16, gate=None):
    model.eval(); preds = []
    for i in range(0, len(feats), bs):
        preds.append(model(feats[i:i + bs], gate).argmax(-1))
    pred = torch.cat(preds).reshape(occ.shape)
    return confusion_miou(pred, occ, model.cfg["num_classes"])
