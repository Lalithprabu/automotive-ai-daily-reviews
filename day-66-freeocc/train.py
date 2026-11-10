"""Train FreeOcc on synthetic BEV worlds and report mIoU per horizon + ablations.
   python train.py --steps 1500            (CPU, a few minutes)
   python train.py --steps 1500 --no-warp  (ablation: no ego-motion warping of the future state)
"""
import argparse, json, os, time
import torch
import torch.nn.functional as F
from models.freeocc import FreeOcc
from src.geometry import warp_bev
from src.synthetic import make_batch, corrupt_history, H, HIST


def iou_per_horizon(pred_bin, gt):
    """pred_bin, gt: (B,H,G,G) {0,1}.  Dataset-level IoU per horizon -> (H,)"""
    inter = (pred_bin * gt).sum((0, 2, 3)); union = ((pred_bin + gt) > 0).float().sum((0, 2, 3))
    return inter / union.clamp(min=1)


def warp_only_baseline(batch, hist_noisy):
    """Static-world baseline: newest observed frame warped into every future ego frame (no learning).
    warp_bev wants the pose of the TARGET frame (future k) in the SOURCE frame (current) == fut_rel[:, k]."""
    last = hist_noisy[:, -1:]
    return torch.stack([warp_bev(last, batch["fut_rel"][:, k])[:, 0] for k in range(H)], 1)


@torch.no_grad()
def evaluate(model, n=300, seed=123, drop=0.0, jitter=0.0, bs=100):
    model.eval(); g = torch.Generator().manual_seed(seed)
    acc = {k: [] for k in ["warp_only", "decoder", "flow", "blend", "max", "mean"]}; gts = []
    for _ in range(n // bs):
        b = make_batch(bs, g, jitter=jitter); hn = corrupt_history(b["hist_gt"], g, drop=drop)
        out = model(hn, b["hist_rel"], b["fut_rel"], sample=True, gen=g)
        thr = lambda p: (p > 0.5).float()
        acc["warp_only"].append(thr(warp_only_baseline(b, hn))); acc["decoder"].append(thr(out["p_dec"]))
        acc["flow"].append(thr(out["p_ode"])); acc["blend"].append(thr(out["p"]))
        acc["max"].append(thr(torch.maximum(out["p_dec"], out["p_ode"]))); acc["mean"].append(thr(0.5 * (out["p_dec"] + out["p_ode"])))
        gts.append(b["fut_gt"])
    gt = torch.cat(gts)
    res = {k: iou_per_horizon(torch.cat(v), gt) for k, v in acc.items()}
    return {k: dict(per_horizon=[round(x, 4) for x in v.tolist()], mIoU=round(v.mean().item(), 4)) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--no-warp", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/freeocc.pt")
    a = ap.parse_args()
    torch.manual_seed(a.seed); g = torch.Generator().manual_seed(a.seed)
    model = FreeOcc(use_warp=not a.no_warp)
    print("params:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    t0 = time.time()
    for s in range(a.steps):
        model.train(); b = make_batch(a.bs, g); hn = corrupt_history(b["hist_gt"], g, drop=0.15)
        gt = b["fut_gt"]
        out = model(hn, b["hist_rel"], b["fut_rel"])
        # (1) heteroscedastic BCE: one MC sample of the logit with learned noise exp(logvar/2)
        noisy = out["logit"] + torch.exp(0.5 * out["logvar"]) * torch.randn_like(out["logit"])
        bce = F.binary_cross_entropy_with_logits(noisy, gt) + 0.5 * F.binary_cross_entropy_with_logits(out["logit"], gt)
        # (2) conditional flow matching on the SAME conditioning features
        B_ = gt.shape[0]; hi = torch.randint(0, H, (B_,), generator=g); ar = torch.arange(B_)      # one random horizon per scene
        f_sub = out["f"].view(B_, H, *out["f"].shape[1:])[ar, hi]                                  # (B,12,G,G)
        fm = model.flow_loss(f_sub, gt[ar, hi])
        loss = bce + fm
        # (3) blend parameters only: detached decoder + ODE sample, BCE on the blended probability
        with torch.no_grad():
            p_ode = model.sample_flow(f_sub.detach(), steps=3); pd = out["p_dec"][ar, hi].detach(); lv = out["logvar"][ar, hi].detach()
        w = model.blend_weight(lv); pb = ((1 - w) * pd + w * p_ode).clamp(1e-4, 1 - 1e-4)
        loss = loss + 0.2 * F.binary_cross_entropy(pb, gt[ar, hi])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if s % 100 == 0: print(f"step {s:4d} bce {bce.item():.3f} fm {fm.item():.3f} blend a={model.blend_a.item():.2f} b={model.blend_b.item():.2f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); torch.save(model.state_dict(), a.out)
    res = dict(clean=evaluate(model), robust=evaluate(model, drop=0.3, jitter=0.08), train_s=round(time.time() - t0))
    print(json.dumps({k: (v if k == "train_s" else {m: x["mIoU"] for m, x in v.items()}) for k, v in res.items()}, indent=1))
    json.dump(res, open(a.out.replace(".pt", "_metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
