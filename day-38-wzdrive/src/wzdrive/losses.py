import torch
import torch.nn.functional as F


def min_ade_fde(pred, gt):
    """pred (B,K,T,2) metres, gt (B,T,2) -> per-sample minADE@K, minFDE@K (B,), best idx (B,)."""
    d = torch.linalg.norm(pred - gt[:, None], dim=-1)          # (B,K,T)
    ade = d.mean(-1)                                           # (B,K)
    fde = d[..., -1]                                           # (B,K)
    best = ade.argmin(1)
    return ade.min(1).values, fde.min(1).values, best


def wta_loss(pred, logits, gt, relax_eps=0.05, cls_weight=0.5):
    """Relaxed winner-take-all regression + CE on the winning mode.
    pred (B,K,T,2), logits (B,K), gt (B,T,2); all in normalised units."""
    K = pred.shape[1]
    l = F.smooth_l1_loss(pred, gt[:, None].expand_as(pred), reduction="none").mean((-1, -2))  # (B,K)
    best = l.argmin(1)
    w = torch.full_like(l, relax_eps / max(K - 1, 1))
    w.scatter_(1, best[:, None], 1.0 - relax_eps)
    reg = (w * l).sum(1).mean()
    cls = F.cross_entropy(logits, best)
    return reg + cls_weight * cls, reg.detach(), cls.detach()


def topk_min_ade(pred, logits, gt, k):
    """Ranked metric: keep only the k modes with the highest logits, then take the best ADE among them.
    k=1 -> ADE of the model's own top pick (does the classifier choose the right intent?);
    k=K -> ordinary minADE@K (ignores the classifier entirely)."""
    ade = torch.linalg.norm(pred - gt[:, None], dim=-1).mean(-1)          # (B,K)
    idx = logits.topk(k, dim=1).indices
    return ade.gather(1, idx).min(1).values
