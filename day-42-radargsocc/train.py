"""Train RadarGSOcc and the early-fusion baseline on identical synthetic data; save checkpoints + results.json."""
import yaml, json, time, sys, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, ".")
from models.radargsocc import RadarGSOcc, RadarOccBaseline, n_params
from src.data import RadarOccDataset
from src.metrics import occ_iou, weighted_miou, class_ious, ebkl


def run(cfg=None, path="config.yaml", out="checkpoints"):
    cfg = cfg or yaml.safe_load(open(path)); torch.manual_seed(cfg["seed"])
    tr = RadarOccDataset(cfg["train_scenes"], seed=1); va = RadarOccDataset(cfg["val_scenes"], seed=2)
    snow = RadarOccDataset(cfg["val_scenes"], seed=2, snow=0.8)          # same scenes, snowy radar
    w = torch.tensor(cfg["class_weights"])
    models = {"new": RadarGSOcc(cfg["feat_dim"], 5, cfg["attn_points"]), "old": RadarOccBaseline(cfg["baseline_dim"], 5)}
    res = {"params": {k: n_params(m) for k, m in models.items()}}
    print("params", res["params"]); import os; os.makedirs(out, exist_ok=True)
    for name, m in models.items():
        opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        dl = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True); t0 = time.time()
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["epochs"] * len(dl))
        for ep in range(cfg["epochs"]):
            m.train(); tot = 0
            for r, o in dl:
                loss = F.cross_entropy(m(r), o, weight=w); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
            if ep % 5 == 0 or ep == cfg["epochs"] - 1: print(f"{name} ep{ep} loss {tot/len(dl):.4f} ({time.time()-t0:.0f}s)", flush=True)
        torch.save(m.state_dict(), f"{out}/{name}.pt")
    for name, m in models.items():
        m.eval(); P = {}
        with torch.no_grad():
            for tag, ds in (("normal", va), ("snow", snow)):
                P[tag] = torch.stack([m(ds[i][0][None]).argmax(1)[0] for i in range(len(ds))])
        G = torch.stack([va[i][1] for i in range(len(va))])
        ci = class_ious(P["normal"], G)
        res[name] = {"occ_iou": occ_iou(P["normal"], G), "w_miou": weighted_miou(P["normal"], G), "vehicle_iou": ci[3],
                     "occ_iou_snow": occ_iou(P["snow"], G), "w_miou_snow": weighted_miou(P["snow"], G),
                     "ebkl_snow": ebkl(P["snow"], P["normal"])}
    json.dump(res, open(f"{out}/results.json", "w"), indent=2); print(json.dumps(res, indent=2)); return res


if __name__ == "__main__": run()
