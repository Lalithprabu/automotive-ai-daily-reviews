"""Held-out sweeps: AP vs transmission delay, keep ratio (bandwidth), and an ablation of each Sparse2comm module."""
import json, torch
from sparse2comm import ModelCfg, DataCfg, Sparse2comm
from sparse2comm.metrics import evaluate


def load(name, **override):
    ck = torch.load(f"checkpoints/{name}.pt", map_location="cpu")
    cfg = dict(ck["cfg"]); cfg.update(override)
    m = Sparse2comm(ModelCfg(**cfg)); m.load_state_dict(ck["state"]); return m.eval(), DataCfg(**ck["data"])


if __name__ == "__main__":
    n = 512; R = {}
    models = {k: load(k) for k in ("ego_only", "old_way", "sparse2comm")}
    R["delay"] = {k: [evaluate(m, dc, n, delay=d, radii=(2.0,))["AP@r2"] for d in range(4)] for k, (m, dc) in models.items()}
    R["keep"] = {k: [evaluate(m, dc, n, keep_ratio=r, radii=(2.0,))["AP@r2"] for r in (0.1, 0.2, 0.3, 0.45, 0.6)] for k, (m, dc) in models.items()}
    m, dc = models["sparse2comm"]
    R["bandwidth_pct"] = {}
    # ablation: switch off one restoration module of the trained full model at inference (no retraining)
    abl = {}
    for name, kw in (("full", {}), ("no_flow", dict(use_flow=False)), ("no_calib", dict(use_calib=False)), ("no_recon+flow+calib", dict(use_recon=False, use_flow=False, use_calib=False))):
        mm, _ = load("sparse2comm", **kw); abl[name] = evaluate(mm, dc, n, radii=(2.0,))
    R["inference_ablation"] = abl
    R["mixed"] = {k: evaluate(m_, dc_, 1024) for k, (m_, dc_) in models.items()}
    json.dump(R, open("checkpoints/sweeps.json", "w"), indent=1); print(json.dumps(R, indent=1))
