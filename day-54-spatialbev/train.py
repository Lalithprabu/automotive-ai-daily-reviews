"""Train physics-residual dynamics models on synthetic off-road episodes and evaluate rollouts.
Usage: python train.py --iters 1500 --modes blind point pooled spatial
"""
import argparse, json, time
import torch

from spatialbev.models import SpatialBEVDynamics, physics_rollout
from spatialbev.terrain import make_terrain, sample_episodes

HIST, H = 5, 24


def loss_fn(pred, gt):
    pos = (pred[..., :2] - gt[..., :2]).norm(dim=-1)
    yaw = (pred[..., 2] - gt[..., 2]).abs()
    spd = (pred[..., 3] - gt[..., 3]).abs()
    huber = lambda e, d: torch.where(e < d, 0.5 * e ** 2 / d, e - 0.5 * d)
    return (huber(pos, 1.0) + 2 * huber(yaw, 0.3) + 0.3 * huber(spd, 1.0)).mean()


def metrics(pred, gt):
    pos = (pred[..., :2] - gt[..., :2]).norm(dim=-1)
    return {"ADE_m": pos.mean().item(), "FDE_m": pos[:, -1].mean().item(),
            "yaw_err_deg": (pred[..., 2] - gt[..., 2]).abs().mean().item() * 57.2958,
            "speed_err": (pred[..., 3] - gt[..., 3]).abs().mean().item()}


def make_test(bank, n, seed):
    gen = torch.Generator().manual_seed(seed)
    st, ac, idx = sample_episodes(bank, n, HIST, H, gen)
    return st, ac, idx


@torch.no_grad()
def evaluate(model, bank, test):
    st, ac, idx = test
    s0, a, gt = st[:, HIST], ac[:, HIST:], st[:, HIST + 1:]
    terr = bank[idx]
    pred = physics_rollout(s0, a) if model is None else torch.cat(
        [model.rollout(s0[i:i + 250], a[i:i + 250], terr[i:i + 250]) for i in range(0, len(s0), 250)])
    err = (pred[..., :2] - gt[..., :2]).norm(dim=-1)
    return pred, err


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--modes", nargs="+", default=["blind", "point", "pooled", "spatial"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints")
    a = ap.parse_args()
    torch.set_num_threads(2)
    train_bank, test_bank = make_terrain(160, 1), make_terrain(48, 2)    # disjoint maps
    test = make_test(test_bank, 1000, 123)
    _, perr = evaluate(None, test_bank, test)
    fde = perr[:, -1]
    hard = fde >= fde.quantile(0.8)                          # "mined" hard cases (physics FDE top 20%)
    fast = test[0][:, HIST, 3] >= 8.0
    subsets = {"all": torch.ones_like(hard), "hard20": hard, "fast": fast}
    results = {"physics": {k: metrics(*[x[m] for x in (evaluate(None, test_bank, test)[0], test[0][:, HIST + 1:])])
                           for k, m in subsets.items()}}
    results["_meta"] = {"iters": a.iters, "batch": a.batch, "seed": a.seed, "horizon_steps": H, "n_test": 1000}
    for mode in a.modes:
        torch.manual_seed(a.seed)
        gen = torch.Generator().manual_seed(a.seed + 7)
        model = SpatialBEVDynamics(mode)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
        t0 = time.time()
        for it in range(a.iters):
            st, ac, idx = sample_episodes(train_bank, a.batch, HIST, H, gen)
            pred = model.rollout(st[:, HIST], ac[:, HIST:], train_bank[idx])
            loss = loss_fn(pred, st[:, HIST + 1:])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if it % 250 == 0 or it == a.iters - 1:
                print(f"[{mode}] it {it:5d} loss {loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
        model.eval()
        pred, _ = evaluate(model, test_bank, test)
        gt = test[0][:, HIST + 1:]
        results[mode] = {k: metrics(pred[m], gt[m]) for k, m in subsets.items()}
        results[mode]["params"] = sum(p.numel() for p in model.parameters())
        torch.save(model.state_dict(), f"{a.out}/{mode}.pt")
        print(mode, json.dumps(results[mode]["all"]), flush=True)
    json.dump(results, open(f"{a.out}/results.json", "w"), indent=1)
    print(json.dumps({k: v["all"] if k != "_meta" else v for k, v in results.items()}, indent=1))
