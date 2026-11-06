"""Training losses: flow matching + world prediction (baseline) and the NegWAM-style negative terms."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .model import NegWAM


def training_loss(model: NegWAM, batch: dict, use_neg: bool, lam_rep: float = 1.0, lam_rider: float = 0.3,
                  margin: float = 0.02, lam_world: float = 0.5):
    """batch tensors (all normalised): scene (B,7) expert (B,T,2) neg_long (B,T,2) neg_lat (B,T,2) lead (B,T).
    Rectified-flow path:  x_t = (1-t) x0 + t x1,  x0 ~ N(0,I), x1 = expert,  target velocity = x1 - x0.
    Clean-sample estimate: x1_hat = x_t + (1-t) v_theta."""
    sc, ex = batch["scene"], batch["expert"]
    B = sc.shape[0]
    x0 = torch.randn_like(ex)
    t = torch.rand(B)
    tt = t[:, None, None]
    xt = (1 - tt) * x0 + tt * ex
    riders = torch.stack([ex, batch["neg_long"], batch["neg_lat"]], 1) if use_neg else None
    v, lead_pred, logit = model(sc, xt, t, riders)
    l_flow = F.mse_loss(v, ex - x0)
    l_world = F.mse_loss(lead_pred, batch["lead"])
    out = dict(flow=l_flow.detach(), world=l_world.detach())
    loss = l_flow + lam_world * l_world
    if use_neg:
        x1_hat = xt + (1 - tt) * v
        d_pos = ((x1_hat - ex) ** 2).mean((1, 2)).sqrt()                                   # distance to expert
        d_neg = torch.stack([((x1_hat - batch[k]) ** 2).mean((1, 2)).sqrt() for k in ("neg_long", "neg_lat")], 1)
        # Repulsive anchors: stay closer to the expert than to every near-expert negative, by a margin.
        # Weighted by t because x1_hat is only meaningful when the path is close to the data end.
        l_rep = (t[:, None] * F.relu(d_pos[:, None] - d_neg + margin)).mean()
        labels = torch.tensor([1.0, 0.0, 0.0]).expand(B, -1)
        l_rider = F.binary_cross_entropy_with_logits(logit, labels)
        loss = loss + lam_rep * l_rep + lam_rider * l_rider
        out.update(rep=l_rep.detach(), rider=l_rider.detach())
    out["total"] = loss.detach()
    return loss, out
