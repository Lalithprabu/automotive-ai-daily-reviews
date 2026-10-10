import torch
import torch.nn.functional as F


def focal_loss(logits, target, alpha=2.0, beta=4.0):
    """CenterNet penalty-reduced focal loss. Positives = pixels with target == 1 (centre pixel is forced to 1.0)."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (target >= 0.999).float()
    neg = 1 - pos
    pl = -((1 - p) ** alpha) * torch.log(p) * pos
    nl = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * neg
    return (pl.sum() + nl.sum()) / pos.sum().clamp(min=1.0)


def total_loss(out, batch, tcfg):
    parts = {"det": focal_loss(out["logits"], batch["heat"])}
    if "recon" in out:
        full = batch["full"][:, 1]
        fgw = (batch["full"][:, 1].norm(dim=1, keepdim=True) > 0.4).float()
        err = ((out["recon_raw"] - full) ** 2).mean(1, keepdim=True)
        err0 = ((out["recon_prev_raw"] - batch["full"][:, 0]) ** 2).mean(1, keepdim=True)
        parts["recon"] = tcfg["lambda_recon"] * (((1 + 10 * fgw) * err).mean() + 0.5 * err0.mean())
    if "flow" in out:
        m = batch["flow_m"]
        parts["flow"] = tcfg["lambda_flow"] * ((out["flow"] - batch["flow_t"]).abs() * m).sum() / (2 * m.sum() + 1.0)
    if "shifted" in out and out["shifted"] is not None and tcfg.get("lambda_calib", 0) > 0 and out.get("offset") is not None:
        ego = batch["ego"]
        m = (ego.norm(dim=1, keepdim=True) > 0.5).float()
        parts["calib"] = tcfg["lambda_calib"] * (m * (out["shifted"] - ego) ** 2).sum() / (m.sum() * ego.shape[1] + 1.0)
    return sum(parts.values()), parts
