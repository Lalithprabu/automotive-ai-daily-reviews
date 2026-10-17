import math, torch
import torch.nn.functional as F


def laplace_nll(mu, b, y):
    return (((mu - y).abs() / b) + torch.log(2 * b)).mean(dim=(-1, -2))


def futurelift_loss(refined, scale, logits, gt):
    """Laplace NLL on the winner-takes-all mode (winner picked on detached refined ADE) +
    endpoint Laplace-mixture NLL for the scores. Unit weights (as in the paper)."""
    with torch.no_grad():
        win = (refined - gt[:, None]).norm(dim=-1).mean(-1).argmin(1)                 # (B,)
    ar = torch.arange(len(win))
    reg = laplace_nll(refined[ar, win], scale[ar, win], gt).mean()
    mu_e, b_e = refined[:, :, -1], scale[:, :, -1]                                    # (B,K,2)
    lp = (-(mu_e - gt[:, None, -1]).abs() / b_e - torch.log(2 * b_e)).sum(-1)         # (B,K)
    cls = -torch.logsumexp(F.log_softmax(logits, -1) + lp, dim=-1).mean()
    return reg + cls, reg.detach(), cls.detach()
