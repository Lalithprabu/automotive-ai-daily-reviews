"""Params / FLOPs / CPU latency (batch 1) of both models. Writes checkpoints/efficiency.json."""
import json, os, sys, time, yaml, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from torch.utils.flop_counter import FlopCounterMode
from lightrocc import LighTROcc, DenseBEVOccForecaster

cfg = yaml.safe_load(open("config.yaml")); x = torch.randn(1, 12, cfg["camera"]["h"], cfg["camera"]["w"])
res = {}
for name, M in (("lightrocc", LighTROcc), ("dense", DenseBEVOccForecaster)):
    m = M(cfg).eval()
    with FlopCounterMode(display=False) as fc:      # forward FLOPs (grad tracking on so the counter's hooks are happy)
        m(x)
    with torch.no_grad():
        for _ in range(5): m(x)
        t0 = time.time()
        for _ in range(30): m(x)
    res[name] = dict(params=sum(p.numel() for p in m.parameters()), gflops=fc.get_total_flops() / 1e9, ms=(time.time() - t0) / 30 * 1000)
    print(name, res[name])
json.dump(res, open("checkpoints/efficiency.json", "w"), indent=2)
