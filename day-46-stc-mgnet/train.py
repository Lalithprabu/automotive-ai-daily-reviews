"""Train the 3-way old-vs-new ablation with identical budgets:
  baseline  : pose-only ST-GCN              (use_stc=False, use_modulation=False)
  pose+STC  : + STC attention               (use_stc=True,  use_modulation=False)
  STC-MGNet : + cross-modal modulation      (use_stc=True,  use_modulation=True)
Usage: python train.py --config config.yaml"""
import argparse, json, time, yaml, torch, torch.nn.functional as F
import numpy as np
from src.stcmgnet.model import STCMGNet
from src.stcmgnet.data import build_split

VARIANTS = {"baseline": (False, False), "pose_stc": (True, False), "stc_mgnet": (True, True)}


def make_model(cfg, name):
    s, m = VARIANTS[name]
    return STCMGNet(**{k: cfg["model"][k] for k in ("in_channels", "hidden", "visual_dim", "ego_dim",
                      "embed_dim", "temporal_kernel", "dropout")}, use_stc=s, use_modulation=m)


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    p, v, e, y = data
    pred = (torch.sigmoid(model(p, v, e)) > 0.5).float()
    return (pred == y).float().mean().item()


def train_variant(cfg, name, tr, va):
    torch.manual_seed(cfg["seed"])
    model = make_model(cfg, name)
    t = cfg["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, t["epochs"])
    n = tr[0].shape[0]; best, best_state, hist = -1, None, []
    for ep in range(t["epochs"]):
        model.train(); perm = torch.randperm(n); tot = 0
        for i in range(0, n, t["batch_size"]):
            idx = perm[i:i + t["batch_size"]]
            loss = F.binary_cross_entropy_with_logits(model(tr[0][idx], tr[1][idx], tr[2][idx]), tr[3][idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        sched.step(); acc = evaluate(model, va); hist.append((tot / n, acc))
        if acc > best: best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    d = cfg["data"]
    tr = build_split(d["n_train"], cfg["seed"], d["window"], d["scene_len"])
    va = build_split(d["n_val"], cfg["seed"] + 1, d["window"], d["scene_len"])
    te = build_split(d["n_test"], cfg["seed"] + 2, d["window"], d["scene_len"])
    results = {}
    for name in VARIANTS:
        t0 = time.time(); model, hist = train_variant(cfg, name, tr, va)
        nparams = sum(p.numel() for p in model.parameters())
        results[name] = dict(params=nparams, val_acc=max(h[1] for h in hist),
                             test_acc=evaluate(model, te), final_loss=hist[-1][0], secs=round(time.time() - t0, 1))
        torch.save({"state": model.state_dict(), "cfg": cfg, "variant": name}, f"{name}.pt")
        print(name, results[name], flush=True)
    json.dump(results, open("results.json", "w"), indent=2)
