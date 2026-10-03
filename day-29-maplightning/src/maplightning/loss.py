import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .data import CLS_NONE, Y_HALF


@torch.no_grad()
def hungarian(logits, pts, gt_pts, gt_cls, present):
    """Per-sample optimal query<->GT assignment. Returns list of (query_idx, gt_idx) LongTensors."""
    prob = logits.softmax(-1)                                              # (B,Q,3)
    l1 = (pts[:, :, None] - gt_pts[:, None]).abs().mean((-1, -2)) / Y_HALF  # (B,Q,G)
    out = []
    for b in range(pts.shape[0]):
        g = present[b].nonzero().squeeze(-1)
        cost = l1[b][:, g] - prob[b][:, gt_cls[b][g]]
        qi, gi = linear_sum_assignment(cost.cpu().numpy())
        out.append((torch.as_tensor(qi), g[torch.as_tensor(gi)]))
    return out


def set_loss(logits, pts, batch, w_cls=1.0, w_pts=5.0, none_w=0.3):
    """DETR-style set loss: CE over all queries (unmatched -> 'none') + L1 on matched polylines."""
    match = hungarian(logits, pts, batch["polys"], batch["cls"], batch["present"])
    tgt = torch.full(logits.shape[:2], CLS_NONE, dtype=torch.long)
    l_pts, n = 0., 0
    for b, (qi, gi) in enumerate(match):
        tgt[b, qi] = batch["cls"][b, gi]
        l_pts = l_pts + (pts[b, qi] - batch["polys"][b, gi]).abs().mean((-1, -2)).sum() / Y_HALF
        n += len(qi)
    wt = torch.tensor([1., 1., none_w])
    l_cls = F.cross_entropy(logits.reshape(-1, 3), tgt.reshape(-1), weight=wt)
    l_pts = l_pts / max(n, 1)
    return w_cls * l_cls + w_pts * l_pts, dict(cls=float(l_cls.detach()), pts=float(l_pts.detach()))
