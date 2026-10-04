"""Re-evaluate saved checkpoints (same held-out scenes as train.py) and write results.json."""
import json, sys, yaml, torch
from egorefine import AsyncFusionNet, MODES
from train import evaluate
cfg = yaml.safe_load(open("config.yaml")); ck = sys.argv[1] if len(sys.argv) > 1 else "checkpoints"; res = {}
for m in MODES:
    d = torch.load(f"{ck}/{m}.pt", map_location="cpu"); net = AsyncFusionNet(m, **d["cfg"]); net.load_state_dict(d["state"])
    res[m] = evaluate(net, n=cfg["eval"]["n_scenes"], seed=cfg["eval"]["seed"])
    print(m, {k: round(v, 3) for k, v in res[m].items() if k != "AP@1.0_by_delay"}, {k: round(v, 3) for k, v in res[m]["AP@1.0_by_delay"].items()})
json.dump(res, open(f"{ck}/results.json", "w"), indent=1)
