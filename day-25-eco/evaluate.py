"""Same weights, same seeds, same noise: raw waypoints ('old way') vs ECO ('new way'). Writes results.json."""
import argparse, json, yaml, numpy as np, torch
from eco.policy import WaypointPolicy
from eco.eco_layer import EndpointConstrainedOptimizer
from eco.rollout import run_episode


def load(cfg, path):
    pol = WaypointPolicy(cfg["route_points"], cfg["history"], cfg["horizon"], cfg["hidden"])
    pol.load_state_dict(torch.load(path)); pol.eval()
    e = cfg["eco"]
    eco = EndpointConstrainedOptimizer(cfg["horizon"], cfg["history"], e["lambda_accel"], e["lambda_jerk"], e["lambda_fidelity"]).eval()
    return pol, eco


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--ckpt", default="policy.pt")
    ap.add_argument("--out", default="results.json"); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); pol, eco = load(cfg, a.ckpt)
    res = {"raw": [], "eco": []}
    for ep in range(cfg["eval"]["episodes"]):
        for k, use in (("raw", False), ("eco", True)):
            m = run_episode(pol, eco, cfg, 1000 + ep, use); m.pop("frames"); m.pop("speed_trace"); m.pop("lat_trace")
            res[k].append(m)
    summ = {}
    for k, ms in res.items():
        summ[k] = {key: float(np.mean([m[key] for m in ms])) for key in ("lat_rms", "jerk_rms", "steer_rate_rms", "progress")}
        summ[k]["off_road_rate"] = float(np.mean([m["off_road"] for m in ms]))
    summ["jerk_reduction_pct"] = 100 * (1 - summ["eco"]["jerk_rms"] / summ["raw"]["jerk_rms"])
    summ["progress_gain_pct_points"] = 100 * (summ["eco"]["progress"] - summ["raw"]["progress"])
    json.dump(summ, open(a.out, "w"), indent=2); print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
