import torch
import torch.nn.functional as F


def det_loss(logits, heat):
    """Class-balanced BCE (positives and negatives averaged separately) -- Day 13/14 lesson."""
    pos = (heat >= 0.5).float()
    bce = F.binary_cross_entropy_with_logits(logits, heat, reduction="none")
    lp = (bce * pos).sum() / pos.sum().clamp_min(1)
    ln = (bce * (1 - pos)).sum() / (1 - pos).sum().clamp_min(1)
    return lp + ln


def objectness_loss(obj_logits, heat):
    return det_loss(obj_logits, heat)


def app_loss(B, dF, target, margin=5e-5, rho=1.5, w=(1.0, 0.25, 0.25, 0.1)):
    """Residual-application regulariser (4 terms named in the paper; formulas are this repo's reading).
    r = target - B is the residual the module *should* apply; dF is what it applied (after the gate)."""
    r = (target - B).detach()
    nr = r.norm(dim=1)                           # [B,H,W]
    nd = dF.norm(dim=1)
    active = (nr > 1e-3).float()
    cos = F.cosine_similarity(dF, r, dim=1, eps=1e-6)
    l_dir = ((1 - cos) * active).sum() / active.sum().clamp_min(1)
    l_mag = ((nd - nr).abs()).mean()
    err_before = (r ** 2).sum(1); err_after = ((B + dF - target) ** 2).sum(1)
    l_imp = F.relu(err_after - err_before + margin).mean()
    l_over = F.relu(nd - rho * nr).mean()
    return w[0] * l_dir + w[1] * l_mag + w[2] * l_imp + w[3] * l_over
