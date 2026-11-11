"""Seed-1 re-run for LOWP and the dense baseline (accuracy only)."""
import json
import torch
import yaml
from evaluate import metrics, predict
from lowp.data import rig_tensors
from lowp.world import make_rig
from train import build, load

cfg = yaml.safe_load(open("config.yaml")); test = load("test")
K, R, t = rig_tensors(make_rig(*cfg["data"]["img_hw"]))
res = {}
for name in ["lowp", "dense"]:
    for seed in (0, 1):
        m = build(name, cfg).eval(); m.load_state_dict(torch.load(f"checkpoints/{name}_s{seed}.pt"))
        res[f"{name}_s{seed}"] = r = metrics(predict(m, test, K, R, t), test)
        print(name, seed, {k: round(v, 3) for k, v in r.items()})
json.dump(res, open("checkpoints/test_metrics_seeds.json", "w"), indent=2)
