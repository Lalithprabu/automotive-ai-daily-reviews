"""Train ManeuverFlow and the multi-query baseline on the synthetic data; writes checkpoints + results.json."""
import argparse, json, time, yaml
import torch, torch.nn.functional as F
from maneuverflow.data import make_dataset
from maneuverflow.model import ManeuverFlow, NORM, M, score_from_logits
from maneuverflow.baseline import MultiQueryPlanner
from maneuverflow.scoring import TF, factors, pdms, classify_maneuver


def prior_means(ds):
    x = ds["gt"] / NORM
    return torch.stack([x[ds["maneuver"] == m].mean(0).flatten() for m in range(M)])


def perturb(gt, n):
    """n perturbed copies of the GT path (lateral weave + speed scaling) -> scorer sees collisions/off-road."""
    B = gt.shape[0]
    out = []
    t = torch.linspace(0, 1, TF)[None, :]
    for _ in range(n):
        lat = (torch.rand(B, 1) * 2 - 1) * 4.5 * (t ** 1.5)
        sc = 0.35 + 0.8 * torch.rand(B, 1)
        out.append(torch.stack([gt[..., 0] * sc, gt[..., 1] + lat], -1))
    return torch.stack(out, 1)


@torch.no_grad()
def evaluate(predict, ds, bs=200, **kw):
    res = dict(pdms=0., nc=0., dac=0., ade1=0., minade=0., oracle_pdms=0., distinct=0., n=0)
    by_man = {m: [] for m in range(M)}
    for i in range(0, ds["hist"].shape[0], bs):
        sl = slice(i, i + bs)
        out = predict(ds["hist"][sl], ds["agents"][sl], **kw)
        valid = ds["agents"][sl][..., 5]
        f = factors(out["best"], ds["fut_agents"][sl], valid)
        p = pdms(f)
        fc = factors(out["cands"], ds["fut_agents"][sl][:, None], valid[:, None])
        pc = pdms(fc)
        gt = ds["gt"][sl]
        res["pdms"] += p.sum().item(); res["nc"] += f["nc"].sum().item(); res["dac"] += f["dac"].sum().item()
        ade1 = (out["best"] - gt).norm(dim=-1).mean(-1)
        res["ade1"] += ade1.sum().item()
        res["minade"] += (out["cands"] - gt[:, None]).norm(dim=-1).mean(-1).min(1).values.sum().item()
        res["oracle_pdms"] += pc.max(1).values.sum().item()
        lab = classify_maneuver(out["cands"])
        res["distinct"] += torch.stack([(lab == m).any(1) for m in range(M)], 1).sum(1).float().sum().item()
        res["n"] += gt.shape[0]
        for m in range(M):
            s = ds["maneuver"][sl] == m
            by_man[m] += p[s].tolist()
    n = res.pop("n")
    r = {k: v / n for k, v in res.items()}
    r["pdms_by_gt_maneuver"] = {str(m): (sum(v) / max(len(v), 1)) for m, v in by_man.items()}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.epochs: cfg["epochs"] = a.epochs
    torch.manual_seed(cfg["seed"])
    t0 = time.time()
    tr, va = make_dataset(cfg["n_train"], cfg["seed"]), make_dataset(cfg["n_val"], cfg["seed"] + 1)
    print(f"data built in {time.time()-t0:.1f}s; GT maneuver mix (train): {torch.bincount(tr['maneuver'], minlength=M).tolist()}")
    mf = ManeuverFlow(cfg["d_model"], cfg["topk"], cfg["per_man"], prior_means(tr), cfg["prior_sigma"])
    bl = MultiQueryPlanner(cfg["d_model"], cfg["topk"] * cfg["per_man"])
    print(f"params: ManeuverFlow {sum(p.numel() for p in mf.parameters()):,}  baseline {sum(p.numel() for p in bl.parameters()):,}")
    o1 = torch.optim.AdamW(mf.parameters(), cfg["lr"], weight_decay=cfg["weight_decay"])
    o2 = torch.optim.AdamW(bl.parameters(), cfg["lr"], weight_decay=cfg["weight_decay"])
    N, bs = cfg["n_train"], cfg["batch_size"]
    log = []
    for ep in range(cfg["epochs"]):
        mf.train(); bl.train()
        perm = torch.randperm(N)
        agg = dict(flow=0., router=0., world=0., score=0., refine=0., base=0.)
        nb = 0
        for i in range(0, N - bs + 1, bs):
            idx = perm[i:i + bs]
            h, ag, fa, gt, mn = (tr[k][idx] for k in ("hist", "agents", "fut_agents", "gt", "maneuver"))
            valid = ag[..., 5]
            # ---------- ManeuverFlow ----------
            cond, tok, vmask, fpos, ftok = mf.encode(h, ag)
            l_world = (((fpos - fa) / NORM) ** 2).mean(-1).mean(-1)[vmask].mean()
            lg_r = mf.router(cond)
            l_router = F.cross_entropy(lg_r, mn)
            l_flow = mf.meanflow_loss((gt / NORM).flatten(1), cond, mn)
            # candidates for the scorer: router top-k (GT forced in) -> one-step generation, + GT + perturbed GT
            with torch.no_grad():
                top = lg_r.detach().clone(); top[torch.arange(bs), mn] = 1e9
                sel = top.topk(cfg["topk"], -1).indices
                mm = sel[:, :, None].expand(-1, -1, cfg["per_man"]).reshape(bs, -1)
                cc = cond.detach()[:, None].expand(-1, mm.shape[1], -1).reshape(bs * mm.shape[1], -1)
                gen = mf._gen_one_step(mm.reshape(-1), cc).view(bs, -1, TF, 2)
                cand = torch.cat([gen, gt[:, None], perturb(gt, 3)], 1)
                tgt = factors(cand, fa[:, None], valid[:, None])
                tgt = torch.stack([tgt["nc"], tgt["dac"], tgt["ep"], tgt["comf"]], -1)
            use_future = bool(torch.rand(()) > cfg["future_dropout"])
            lg, emb = mf.scorer(cand, cond, tok, vmask, ftok, use_future)
            l_score = F.binary_cross_entropy_with_logits(lg[..., [0, 1, 3]], tgt[..., [0, 1, 3]]) + \
                F.binary_cross_entropy_with_logits(lg[..., 2], tgt[..., 2])
            # refiner: WTA L1 to the GT for the closest generated candidate; refined paths are also re-scored (trained)
            ref = mf.refiner(gen.detach(), emb[:, :gen.shape[1]].detach())
            near = (gen - gt[:, None]).norm(dim=-1).mean(-1).argmin(1)
            l_ref = ((ref[torch.arange(bs), near] - gt).abs() / NORM).mean() * 5
            lg_ref, _ = mf.scorer(ref.detach(), cond, tok, vmask, ftok, use_future)
            tr_ref = factors(ref.detach(), fa[:, None], valid[:, None])
            tr_ref = torch.stack([tr_ref["nc"], tr_ref["dac"], tr_ref["ep"], tr_ref["comf"]], -1)
            l_score = l_score + F.binary_cross_entropy_with_logits(lg_ref[..., [0, 1, 3]], tr_ref[..., [0, 1, 3]]) + \
                F.binary_cross_entropy_with_logits(lg_ref[..., 2], tr_ref[..., 2])
            loss = 8 * l_flow + l_router + 40 * l_world + l_score + l_ref
            o1.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(mf.parameters(), 2.0); o1.step()
            # ---------- baseline ----------
            traj, lgb = bl(h, ag)
            err = (traj - gt[:, None]).norm(dim=-1).mean(-1)
            w = err.argmin(1)
            lb = (((traj[torch.arange(bs), w] - gt).abs() / NORM).mean() * 5) + 0.3 * F.cross_entropy(lgb, w)
            o2.zero_grad(); lb.backward(); o2.step()
            for k, v in zip(agg, (l_flow, l_router, l_world, l_score, l_ref, lb)): agg[k] += v.item()
            nb += 1
        agg = {k: v / nb for k, v in agg.items()}
        log.append(agg)
        if ep % 3 == 0 or ep == cfg["epochs"] - 1:
            print(f"ep {ep:02d} " + " ".join(f"{k}={v:.4f}" for k, v in agg.items()) + f"  [{time.time()-t0:.0f}s]")
    mf.eval(); bl.eval()
    torch.save(dict(state=mf.state_dict(), cfg=cfg, prior=prior_means(tr)), "maneuverflow.pt")
    torch.save(dict(state=bl.state_dict(), cfg=cfg), "baseline.pt")
    gen = torch.Generator().manual_seed(123)
    R = {
        "old_way_multiquery": evaluate(bl.predict, va),
        "new_generate_only": evaluate(lambda h, a: mf.predict(h, a, use_future=False, refine=False, gen=gen), va),
        "new_evidence_current_only": evaluate(lambda h, a: mf.predict(h, a, use_future=False, refine=True, gen=gen), va),
        "new_full_maneuverflow": evaluate(lambda h, a: mf.predict(h, a, use_future=True, refine=True, gen=gen), va),
    }
    R["train_log"] = log
    fg = factors(va["gt"], va["fut_agents"], va["agents"][..., 5])
    R["human_gt_reference_pdms"] = pdms(fg).mean().item()
    R["human_gt_reference_nc"] = fg["nc"].mean().item()
    R["world_model_val_fut_err_m"] = None
    with torch.no_grad():
        cond, tok, vm, fpos, _ = mf.encode(va["hist"], va["agents"])
        R["world_model_val_fut_err_m"] = (fpos - va["fut_agents"]).norm(dim=-1).mean(-1)[vm].mean().item()
        cv = va["agents"][..., None, :2] + va["agents"][..., None, 2:4] * mf.world.taus[None, None, :, None]
        R["constant_velocity_fut_err_m"] = (cv - va["fut_agents"]).norm(dim=-1).mean(-1)[vm].mean().item()
    json.dump(R, open("results.json", "w"), indent=1)
    for k, v in R.items():
        if isinstance(v, dict) and "pdms" in v:
            print(f"{k:28s} PDMS={v['pdms']:.3f} NC={v['nc']:.3f} ADE1={v['ade1']:.2f} minADE={v['minade']:.2f} oracle={v['oracle_pdms']:.3f} distinct={v['distinct']:.2f}")
    print("world model future error (m):", R["world_model_val_fut_err_m"], " const-vel:", R["constant_velocity_fut_err_m"])


if __name__ == "__main__":
    main()
