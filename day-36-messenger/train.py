"""Train the old-way (K=1, no gate) and MESSENGER (K=3, gated) forecasters on identical data, then
evaluate extrapolation error at horizons 1..4 on held-out synthetic scenes.
Usage: python train.py --config config.yaml --variant both"""
import argparse, json, time, yaml, torch
from torch.utils.data import DataLoader
from messenger import PairFlowEstimator, MemoryForecaster, rollout, SceneFlowDataset, collate


class Pipeline(torch.nn.Module):
    def __init__(self, d, K, topk, use_gate):
        super().__init__()
        self.pair = PairFlowEstimator(d)
        self.fc = MemoryForecaster(d, K, topk, use_gate)
        self.K = K

    def observe(self, pts):
        """Observed pairwise flow s->s+1 for every consecutive frame pair. -> flows, states lists."""
        fl, hs = [], []
        for s in range(pts.shape[1] - 1):
            f, h = self.pair(pts[:, s], pts[:, s + 1])
            fl.append(f); hs.append(h)
        return fl, hs

    def entries_at(self, pts, fl, hs, t):
        lo = max(0, t - self.K)
        return [dict(P=pts[:, s], f=fl[s].detach(), h=hs[s].detach()) for s in range(lo, t)]


def run_loss(pipe, batch, cfg):
    pts, disp = batch["pts"], batch["disp"]
    fl, hs = pipe.observe(pts)
    l_pair = sum((fl[s] - disp[:, s, :, 0]).abs().mean() for s in range(len(fl))) / len(fl)
    Ht, l_f, nll, n = cfg["train"]["train_horizon"], 0.0, 0.0, 0
    for t in (4, 5):
        outs = rollout(pipe.fc, pts[:, t], pipe.entries_at(pts, fl, hs, t), Ht)
        for h, (cum, logsig, _) in enumerate(outs):
            err = (cum - disp[:, t, :, h]).norm(dim=-1)
            l_f = l_f + err.mean()
            sig = torch.exp(logsig.clamp(-4, 3))
            nll = nll + (err.detach() / sig + logsig.clamp(-4, 3)).mean()
            n += 1
    return l_pair + l_f / n, dict(pair=l_pair.item(), fore=(l_f / n).item(), nll=(nll / n).item()), nll / n


@torch.no_grad()
def evaluate(pipe, loader, H=4, t=5):
    pipe.eval()
    tot = {k: torch.zeros(H) for k in ("all", "obj", "bg", "cv")}
    cnt = {"all": 0, "obj": 0, "bg": 0}
    for batch in loader:
        pts, disp, lab = batch["pts"], batch["disp"], batch["lab"][:, t]
        fl, hs = pipe.observe(pts)
        ent = pipe.entries_at(pts, fl, hs, t)
        outs = rollout(pipe.fc, pts[:, t], ent, H)
        r1 = pipe.fc.retrieve(pts[:, t], ent)[0]["f"]                  # last pairwise flow, retrieved
        obj = lab >= 0
        for h in range(H):
            e = (outs[h][0] - disp[:, t, :, h]).norm(dim=-1)
            ecv = ((h + 1) * r1 - disp[:, t, :, h]).norm(dim=-1)       # non-learned constant-velocity carry
            tot["all"][h] += e.sum(); tot["obj"][h] += e[obj].sum(); tot["bg"][h] += e[~obj].sum()
            tot["cv"][h] += ecv[obj].sum()
        cnt["all"] += lab.numel(); cnt["obj"] += obj.sum().item(); cnt["bg"] += (~obj).sum().item()
    return {"epe_all": (tot["all"] / cnt["all"]).tolist(), "epe_moving": (tot["obj"] / cnt["obj"]).tolist(),
            "epe_static": (tot["bg"] / cnt["bg"]).tolist(), "epe_cv_carry_moving": (tot["cv"] / cnt["obj"]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml"); ap.add_argument("--variant", default="both")
    ap.add_argument("--steps", type=int, default=None); ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.steps: cfg["train"]["steps"] = a.steps
    torch.manual_seed(cfg["seed"])
    dc = cfg["data"]; kw = dict(T_obs=dc["T_obs"], H=dc["H"], N=dc["N"])
    tr = SceneFlowDataset(dc["n_train"], cfg["seed"] + 1, **kw)
    te = SceneFlowDataset(dc["n_test"], cfg["seed"] + 3, **kw)
    variants = {"old_way_K1": dict(K=1, use_gate=False), "messenger_K3": dict(K=cfg["model"]["K"], use_gate=True)}
    if a.variant != "both": variants = {k: v for k, v in variants.items() if a.variant in k}
    results = {}
    for name, v in variants.items():
        torch.manual_seed(cfg["seed"])
        pipe = Pipeline(cfg["model"]["d"], v["K"], cfg["model"]["topk"], v["use_gate"])
        npar = sum(p.numel() for p in pipe.parameters())
        opt = torch.optim.AdamW(pipe.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["train"]["steps"])
        dl = DataLoader(tr, batch_size=cfg["train"]["batch_size"], shuffle=True, collate_fn=collate, drop_last=True)
        step, t0, hist = 0, time.time(), []
        while step < cfg["train"]["steps"]:
            for batch in dl:
                pipe.train()
                loss, parts, nll = run_loss(pipe, batch, cfg)
                opt.zero_grad(); (loss + cfg["train"]["nll_weight"] * nll).backward()
                torch.nn.utils.clip_grad_norm_(pipe.parameters(), 5.0); opt.step(); sched.step(); step += 1
                hist.append(parts)
                if step % 50 == 0 or step == 1:
                    print(f"[{name}] step {step:4d} pair {parts['pair']:.3f} fore {parts['fore']:.3f} ({time.time()-t0:.0f}s)", flush=True)
                if step >= cfg["train"]["steps"]: break
        torch.save({"state": pipe.state_dict(), "K": v["K"], "use_gate": v["use_gate"], "cfg": cfg}, f"{name}.pt")
        ev = evaluate(pipe, DataLoader(te, batch_size=50, collate_fn=collate))
        ev.update(params=npar, train_seconds=time.time() - t0, first_loss=hist[0], last_loss=hist[-1])
        results[name] = ev
        print(name, json.dumps({k: ev[k] for k in ("epe_moving", "epe_cv_carry_moving", "epe_static")}), flush=True)
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
