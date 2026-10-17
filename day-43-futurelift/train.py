"""Stage 1: train the frozen map-blind backbone. Stage 2: train FutureLift variants on its frozen proposals.
Variants (each trained separately -- an enable/disable ablation needs both paths trained):
  bidirectional = FutureLift (later evidence can correct earlier segments)
  causal        = "sequential" old-way (segment r only sees segments <= r)
  none          = no temporal edges at all
"""
import argparse, json, time, yaml, torch
from futurelift.data import JunctionDataset
from futurelift.backbone import MapBlindBackbone, backbone_loss
from futurelift.model import FutureLift
from futurelift.losses import futurelift_loss
from futurelift.metrics import forecast_metrics


def proposals(bb, ds, bs=512):
    out = []
    with torch.no_grad():
        for b in ds.loader(bs, False):
            out.append(bb(b))
    return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])


@torch.no_grad()
def evaluate(model, ds, P, Lg, bs=128):
    model.eval(); R, S, G = [], [], []
    for i, b in enumerate(ds.loader(bs, False)):
        sl = slice(i * bs, i * bs + len(b["gt"]))
        r, _, lg = model(b, P[sl], Lg[sl]); R.append(r); S.append(lg); G.append(b["gt"])
    return forecast_metrics(torch.cat(R), torch.cat(S), torch.cat(G))


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path)); torch.manual_seed(cfg["seed"])
    tr = JunctionDataset(cfg["data"]["n_train"], 1); va = JunctionDataset(cfg["data"]["n_val"], 2)
    te = JunctionDataset(cfg["data"]["n_test"], 3)
    bc = cfg["backbone"]; bb = MapBlindBackbone(bc["K"], bc["hidden"])
    opt = torch.optim.Adam(bb.parameters(), lr=bc["lr"]); t0 = time.time()
    for ep in range(bc["epochs"]):
        for b in tr.loader(bc["batch_size"], True, seed=ep):
            tj, lg = bb(b); loss = backbone_loss(tj, lg, b["gt"]); opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == bc["epochs"] - 1: print(f"[backbone] ep {ep} loss {loss.item():.4f}")
    bb.eval()
    for p in bb.parameters(): p.requires_grad_(False)
    torch.save(bb.state_dict(), "checkpoints/backbone.pt")
    Ptr, Ltr = proposals(bb, tr); Pva, Lva = proposals(bb, va); Pte, Lte = proposals(bb, te)
    res = {"backbone_only": forecast_metrics(Pte, Lte, te.gt)}
    print("backbone-only test:", res["backbone_only"])
    rc = cfg["refiner"]
    for var in cfg["variants"]:
        torch.manual_seed(cfg["seed"])
        m = FutureLift(rc["d"], rc["heads"], rc["layers"], rc["R"], var)
        opt = torch.optim.Adam(m.parameters(), lr=rc["lr"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, rc["epochs"])
        best, best_sd = 1e9, None
        for ep in range(rc["epochs"]):
            m.train()
            order = torch.randperm(len(tr), generator=torch.Generator().manual_seed(100 + ep))
            for i in range(0, len(tr), rc["batch_size"]):
                idx = order[i:i + rc["batch_size"]]; b = tr.batch(idx)
                r, s, lg = m(b, Ptr[idx], Ltr[idx])
                loss, reg, cls = futurelift_loss(r, s, lg, b["gt"])
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
            sched.step()
            v = evaluate(m, va, Pva, Lva)
            if v["minADE"] < best: best, best_sd = v["minADE"], {k: x.clone() for k, x in m.state_dict().items()}
            print(f"[{var}] ep {ep} loss {loss.item():.3f} (reg {reg:.3f} cls {cls:.3f}) val minADE {v['minADE']:.3f}")
        m.load_state_dict(best_sd); torch.save(m.state_dict(), f"checkpoints/{var}.pt")
        res[var] = evaluate(m, te, Pte, Lte); print(var, "test:", res[var], f"[{time.time()-t0:.0f}s]")
    res["n_params"] = {v: sum(p.numel() for p in FutureLift(rc["d"], rc["heads"], rc["layers"], rc["R"], v).parameters())
                       for v in cfg["variants"]}
    json.dump(res, open("checkpoints/results.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
