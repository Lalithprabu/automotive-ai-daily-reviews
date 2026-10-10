import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def occ_loss(probs, gt, w_bce=1.0, w_dice=1.0):
    """BCE + soft-Dice over all (t, voxel). probs/gt [B,T1,V]."""
    p = probs.clamp(1e-5, 1 - 1e-5)
    bce = F.binary_cross_entropy(p, gt)
    inter = (p * gt).sum(-1)
    dice = 1 - (2 * inter + 1) / (p.sum(-1) + gt.sum(-1) + 1)
    return w_bce * bce + w_dice * dice.mean(), dict(bce=bce.item(), dice=dice.mean().item())


@torch.no_grad()
def hungarian(out, batch):
    """Match queries to GT instances by t=0 BEV center distance + existence prior. Returns list of (qi, gi) per sample."""
    cen0 = out["centers"][:, 0, :, :2]                                    # [B,Q,2]
    res = []
    for b in range(cen0.shape[0]):
        gi = torch.nonzero(batch["valid"][b]).flatten()
        if len(gi) == 0:
            res.append(([], [])); continue
        d = torch.cdist(cen0[b], batch["cen"][b, gi, 0])                  # [Q,G]
        cost = d + 2.0 * (1 - out["exist"][b])[:, None]
        qi, gj = linear_sum_assignment(cost.cpu().numpy())
        res.append((list(qi), [int(gi[j]) for j in gj]))
    return res


def lighTROcc_loss(out, batch, cfg):
    t = cfg["train"]
    gt = batch["occ"]
    l_occ, parts = occ_loss(out["probs"], gt, t["w_occ_bce"], t["w_dice"])
    match = hungarian(out, batch)
    B, Q = out["exist"].shape
    tgt_exist = torch.zeros(B, Q)
    c_loss = y_loss = s_loss = d_loss = 0.0
    n = 0
    for b, (qi, gi) in enumerate(match):
        for q, g in zip(qi, gi):
            tgt_exist[b, q] = 1.0
            pc = out["centers"][b, :, q]                                   # [T1,3]
            c_loss = c_loss + F.smooth_l1_loss(pc[:, :2], batch["cen"][b, g]) + 0.3 * (pc[0, 2] - 0.8).abs()
            dy = out["yaws"][b, :, q] - batch["yaw"][b, g]
            y_loss = y_loss + (1 - torch.cos(dy)).mean()                   # wrap-safe yaw loss
            s_loss = s_loss + F.l1_loss(torch.log(out["size"][b, q]), torch.log(batch["size"][b, g]))
            d_loss = d_loss + (torch.log(out["depth"][b, q]) - torch.log(batch["cen"][b, g, 0, 0])).abs()
            n += 1
    n = max(n, 1)
    l_exist = F.binary_cross_entropy(out["exist"].clamp(1e-5, 1 - 1e-5), tgt_exist)
    total = l_occ + t["w_exist"] * l_exist + (t["w_center"] * c_loss + t["w_yaw"] * y_loss + t["w_size"] * s_loss
                                             + t["w_depth"] * d_loss) / n
    parts.update(exist=l_exist.item(), center=float(c_loss) / n)
    return total, parts
