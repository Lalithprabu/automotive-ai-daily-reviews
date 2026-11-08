"""Train MotoAnticipator on synthetic scenes and report the anticipation-vs-observation curve."""
import argparse, json, time, torch, torch.nn.functional as F
from models.motoanticipator import MotoAnticipator
from src.synthetic import make_batch, future_target, T_OBS, K, MANEUVERS


def apparent_cv(b, L, model):
    """Static/naive baseline: constant velocity from *apparent* (uncompensated) ego-frame motion."""
    from models.motoanticipator import EgoMotionCompensation
    v = EgoMotionCompensation(False)(b["agents_ego"], b["ego"])
    ar = torch.arange(L.shape[0]); p, vv = b["agents_ego"][ar, L - 1], v[ar, L - 1]
    return p[:, :, None] + vv[:, :, None] * model.tau[None, None, :, None]


@torch.no_grad()
def evaluate(model, n=2000, seed=123):
    model.eval(); g = torch.Generator().manual_seed(seed); b = make_batch(n, g)
    out = {}
    stress = b["ego"][..., 1].abs().mean(1); hi = stress > stress.quantile(0.67)   # high ego-motion tercile
    for f in (0.2, 0.4, 0.6, 0.8, 1.0):
        L = torch.full((n,), max(2, round(f * T_OBS)), dtype=torch.long)
        lg, pred, cv = model(b["ego"], b["agents_ego"], L)
        tgt = future_target(b, L)
        ade = lambda x: (x - tgt).norm(dim=-1).mean((1, 2))
        acc = (lg[torch.arange(n), L - 1].argmax(-1) == b["label"]).float()
        out[f] = dict(acc=acc.mean().item(), ade=ade(pred).mean().item(), ade_cv_comp=ade(cv).mean().item(),
                      ade_cv_apparent=ade(apparent_cv(b, L, model)).mean().item(),
                      ade_hi_ego=ade(pred)[hi].mean().item(), ade_hi_ego_cv_apparent=ade(apparent_cv(b, L, model))[hi].mean().item())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--no-comp", action="store_true"); ap.add_argument("--out", default="checkpoints/model.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); g = torch.Generator().manual_seed(a.seed)
    model = MotoAnticipator(use_comp=not a.no_comp)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    t0 = time.time()
    for s in range(a.steps):
        model.train(); b = make_batch(a.bs, g)
        L = torch.randint(6, T_OBS + 1, (a.bs,), generator=g)                     # random observation length
        lg, pred, _ = model(b["ego"], b["agents_ego"], L)
        ce = F.cross_entropy(lg.flatten(0, 1), b["label"][:, None].expand(-1, T_OBS).flatten())  # every frame
        ade = (pred - future_target(b, L)).norm(dim=-1).mean()
        loss = ce + 0.5 * ade
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if s % 100 == 0: print(f"step {s:4d} ce {ce.item():.3f} ade {ade.item():.3f}")
    import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(model.state_dict(), a.out)
    res = evaluate(model); print(json.dumps(res, indent=1)); print(f"{time.time()-t0:.0f}s")
    json.dump(res, open(a.out.replace(".pt", "_metrics.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
