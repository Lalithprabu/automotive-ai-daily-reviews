"""Train baseline (kinematics+scene) and WZDrive (+control +gaze) separately; evaluate minADE/minFDE@6."""
import argparse, json, time, yaml, torch, numpy as np
from torch.utils.data import DataLoader
from src.wzdrive import WZDrive, WorkZoneDataset, generate_dataset, wta_loss, min_ade_fde, topk_min_ade
from src.wzdrive.data import CLASS_NAMES


LOSS_SCALE = 5.0  # metres -> loss units; /20 under-weights sub-metre errors (found via overfit test)


def build(cfg, **kw):
    m = cfg["model"]
    return WZDrive(m["d_model"], m["n_heads"], m["n_modes"], m["n_dec_layers"], m["dropout"],
                   cfg["data"]["hist_steps"], cfg["data"]["fut_steps"], **kw)


def fit(model, tr, cfg, tag):
    t = cfg["train"]; opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    dl = DataLoader(WorkZoneDataset(tr), batch_size=t["batch_size"], shuffle=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, t["lr"], total_steps=t["epochs"] * len(dl))
    for ep in range(t["epochs"]):
        model.train(); tot = 0
        for b in dl:
            traj, lg = model(b["kin"], b["ctrl"], b["gaze"], b["scene"])
            loss, _, _ = wta_loss(traj / LOSS_SCALE, lg, b["fut"] / LOSS_SCALE, t["relax_eps"], t["cls_weight"])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item()
        if ep % 4 == 0 or ep == t["epochs"] - 1: print(f"[{tag}] epoch {ep:2d} loss {tot/len(dl):.4f}", flush=True)


@torch.no_grad()
def evaluate(model, d):
    model.eval(); traj, lg = model(d["kin"], d["ctrl"], d["gaze"], d["scene"])
    ade, fde, _ = min_ade_fde(traj, d["fut"])
    ade1 = topk_min_ade(traj, lg, d["fut"], 1)
    return ade, fde, ade1, topk_min_ade(traj, lg, d["fut"], 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    tr = generate_dataset(cfg["data"]["n_train"], cfg["seed"]); va = generate_dataset(cfg["data"]["n_val"], cfg["seed"] + 1)
    res = {}; t0 = time.time()
    for tag, kw in [("baseline", dict(use_gaze=False, use_control=False)), ("wzdrive", dict(use_gaze=True, use_control=True))]:
        m = build(cfg, **kw); print(tag, "params:", sum(p.numel() for p in m.parameters()))
        fit(m, tr, cfg, tag); ade, fde, ade1, ade3 = evaluate(m, va)
        res[tag] = dict(minADE6=ade.mean().item(), minFDE6=fde.mean().item(), top1_ADE=ade1.mean().item(), top3_minADE=ade3.mean().item(),
                        top1_ADE_per_class={CLASS_NAMES[c]: ade1[va['cls'] == c].mean().item() for c in range(4)},
                        per_class={CLASS_NAMES[c]: ade[va["cls"] == c].mean().item() for c in range(4)})
        torch.save(dict(state=m.state_dict(), kw=kw), f"checkpoints/{tag}.pt")
    res["reduction_top1_ADE_pct"] = 100 * (1 - res["wzdrive"]["top1_ADE"] / res["baseline"]["top1_ADE"])
    res["reduction_top3_minADE_pct"] = 100 * (1 - res["wzdrive"]["top3_minADE"] / res["baseline"]["top3_minADE"])
    res["reduction_minADE6_pct"] = 100 * (1 - res["wzdrive"]["minADE6"] / res["baseline"]["minADE6"])
    res["reduction_minFDE6_pct"] = 100 * (1 - res["wzdrive"]["minFDE6"] / res["baseline"]["minFDE6"])
    # constant-velocity reference
    cv = va["kin"][:, -1, 2:][:, None] * (torch.arange(1, 13) * 0.25)[None, :, None]
    res["const_velocity_ADE"] = torch.linalg.norm(cv - va["fut"], dim=-1).mean().item()
    res["train_seconds"] = time.time() - t0
    json.dump(res, open("results.json", "w"), indent=2); print(json.dumps(res, indent=2))


if __name__ == "__main__": main()
