"""Held-out F1 per prediction horizon (KG-Bayes vs MLP), missing-sensor robustness, and closed-loop Monte-Carlo."""
import json, sys, pathlib, numpy as np, torch, yaml
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from kglc.data import make_split, N_REL
from kglc.model import TransE, MLPBaseline, macro_f1
from kglc.decision import monte_carlo, lc_log_prior, false_alarm_rate


def load(path="checkpoints/model.pt"):
    ck = torch.load(path, map_location="cpu"); cfg = ck["cfg"]
    kg = TransE(cfg["model"]["dim"], cfg["model"]["gamma"], cfg["model"]["adv_temp"]); kg.load_state_dict(ck["kg"])
    mlp = MLPBaseline(cfg["mlp_baseline"]["hidden"]); mlp.load_state_dict(ck["mlp"])
    return kg.eval(), mlp.eval(), cfg


def main():
    kg, mlp, cfg = load(); rng = np.random.default_rng(cfg["seed"] + 99); out = {"horizon_f1": {}}
    for h in cfg["data"]["eval_horizons"]:
        s = make_split(cfg["data"]["n_test_per_horizon"], rng, fixed_h=h); y = torch.from_numpy(s.label)
        with torch.no_grad():
            f_kg = macro_f1(kg.log_posterior(torch.from_numpy(s.ents)).argmax(-1), y)
            f_ml = macro_f1(mlp(torch.from_numpy(s.feats)).argmax(-1), y)
            # drop both TTC-rear sensors (evidence 5,6 missing) -- KG just omits those factors
            obs = torch.ones(len(y), N_REL, dtype=torch.bool); obs[:, 5:] = False
            f_drop = macro_f1(kg.log_posterior(torch.from_numpy(s.ents), observed=obs).argmax(-1), y)
        out["horizon_f1"][h] = dict(kg=f_kg, mlp=f_ml, kg_rear_sensors_dropped=f_drop)
        print(f"h={h:.0f}s  KG-Bayes {f_kg:.3f} | MLP {f_ml:.3f} | KG w/o rear TTC {f_drop:.3f}")
    out["closed_loop"] = {}
    for g in cfg["decision"]["sweep_gaps_m"]:
        d = dict(cfg["decision"]); d["crossing_gap_m"] = g
        out["closed_loop"][g] = monte_carlo(kg, d, lc_log_prior(d))
        for k, v in out["closed_loop"][g].items(): print(f"gap@crossing={g:>4}m", k, v)
    out["false_alarm"] = false_alarm_rate(kg, cfg["decision"], lc_log_prior(cfg["decision"]))
    print("lane-keeping neighbour (false-alarm test)", out["false_alarm"])
    d0 = dict(cfg["decision"]); d0["prior_lane_change"] = 1 / 3
    out["false_alarm_balanced_prior"] = false_alarm_rate(kg, d0, lc_log_prior(d0))
    print("... same with balanced 1/3 prior", out["false_alarm_balanced_prior"])
    json.dump(out, open("checkpoints/eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
