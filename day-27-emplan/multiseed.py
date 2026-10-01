import json, subprocess, yaml, sys, numpy as np
rows = []
for sd in (1, 2, 3):
    cfg = yaml.safe_load(open("config.yaml")); cfg["seed"] = sd; yaml.safe_dump(cfg, open("/tmp/c.yaml", "w"))
    subprocess.run([sys.executable, "train.py", "--config", "/tmp/c.yaml"], capture_output=True)
    rows.append(json.load(open("results.json")))
for k in ("old_regression", "emplan_stage1_only", "emplan_stage2"):
    for m in ("collision_rate", "reward", "ade_vs_expert"):
        v = [r[k][m] for r in rows]; print(k, m, f"{np.mean(v):.3f} +- {np.std(v):.3f}", [round(x, 3) for x in v])
