"""Winner-takes-all (relaxed) Gaussian-mixture loss + trajectory metrics."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def per_mode_nll(mu, log_sigma, gt):
    """mu,log_sigma (B,K,T,2), gt (B,T,2) -> summed-over-time NLL per mode (B,K), diagonal Gaussian."""
    d = mu - gt[:, None]
    nll = 0.5 * (d / log_sigma.exp()) ** 2 + log_sigma + 0.5 * math.log(2 * math.pi)
    return nll.sum((-1, -2))


def gmm_loss(logits, mu, log_sigma, gt, relax_eps: float = 0.05):
    """Relaxed WTA: best mode (by ADE) gets the Gaussian NLL + the classification target; the other modes only get a
    small Huber pull (eps) so they do not collapse onto one mode but are not punished into inflating sigma."""
    d = mu - gt[:, None]
    ade = d.norm(dim=-1).mean(-1)                                 # (B,K)
    k_star = ade.argmin(1)
    idx = torch.arange(mu.shape[0], device=mu.device)
    nll = per_mode_nll(mu, log_sigma, gt) / mu.shape[2]           # mean per step
    reg_best = nll[idx, k_star].mean()
    others = F.smooth_l1_loss(mu, gt[:, None].expand_as(mu), reduction="none").mean((-1, -2))   # (B,K)
    mask = torch.ones_like(others).scatter_(1, k_star[:, None], 0.0)
    reg_other = (others * mask).sum(1).div(mu.shape[1] - 1).mean()
    cls = F.cross_entropy(logits, k_star)
    return reg_best + cls + relax_eps * reg_other, dict(nll=reg_best.item(), cls=cls.item())


@torch.no_grad()
def metrics(logits, mu, log_sigma, gt, miss_thr: float = 2.0):
    """minADE_K, minFDE_K, miss rate, top-1 (highest pi) ADE, mixture NLL per step; all averaged over the batch."""
    d = mu - gt[:, None]
    dist = d.norm(dim=-1)                                         # (B,K,T)
    ade, fde = dist.mean(-1), dist[..., -1]
    top1 = logits.argmax(1)
    idx = torch.arange(mu.shape[0])
    logp_k = -per_mode_nll(mu, log_sigma, gt)                     # (B,K) full-trajectory log-density
    mix = -torch.logsumexp(F.log_softmax(logits, 1) + logp_k, 1) / mu.shape[2]
    return dict(minADE=ade.min(1).values.mean().item(), minFDE=fde.min(1).values.mean().item(),
                MR=(fde.min(1).values > miss_thr).float().mean().item(),
                top1ADE=ade[idx, top1].mean().item(), mixNLL=mix.mean().item())
