"""Bucketed analysis: does SCOPE's gain concentrate where the scene-dependent uncertainty differs most from average?"""
import json, torch, numpy as np
from scope import *
from scope.io import load, nominal
from scope.evaluate import metrics, fit_alpha

cfg, model, head, ck = load()
data = make_data(cfg, cfg["seed"]); te = data["test"]
with torch.no_grad():
    a = nominal(model, cfg, te["cond"], te["mode_t"]); e = te["resid"] - a
    S = head.covariance(te["cond"], te["mode_t"], a)
    Sf = ck["S_fixed"].expand(len(e), D, D) * ck["al_fix"]
c = torch.tensor(te["c"]); out = {}
for name, m in [("low crowd density (c<0.33)", c < 0.33), ("mid (0.33-0.66)", (c >= 0.33) & (c < 0.66)), ("high crowd density (c>=0.66)", c >= 0.66)]:
    ms, mf = metrics(S[m], e[m]), metrics(Sf[m], e[m])
    out[name] = {"n": int(m.sum()), "scope_nll": ms["nll_per_step"], "fixed_nll": mf["nll_per_step"],
                 "scope_cov95": ms["cov95"], "fixed_cov95": mf["cov95"]}
    print(f"{name:32s} n={int(m.sum()):4d}  NLL scope {ms['nll_per_step']:.3f} vs fixed {mf['nll_per_step']:.3f} | cov95 scope {ms['cov95']:.3f} vs fixed {mf['cov95']:.3f}")
json.dump(out, open("results_buckets.json", "w"), indent=1)

# downstream planner-style decision: brake if max P(lane encroachment) > 0.05, vs the known-generator oracle (mode-0 scenes)
from scipy.stats import norm
from scope.evaluate import blocks
m = te["mode"] == 0
mu_n = (torch.tensor(te["cv"], dtype=torch.float32) + a).numpy()[:, 1::2]
mu_t = mean_future(te["v"], te["mode"])[:, 1::2]
vs = blocks(S)[:, :, 1, 1].numpy(); vf = blocks(Sf)[:, :, 1, 1].numpy()
vo = blocks(torch.tensor(true_cov(te["v"], te["c"]), dtype=torch.float32))[:, :, 1, 1].numpy()
r = lambda mu, v: (1 - norm.cdf((1.0 - mu) / np.sqrt(v))).max(1)
ro, rs, rf = r(mu_t, vo) > 0.05, r(mu_n, vs) > 0.05, r(mu_n, vf) > 0.05
dec = {"n_scenes": int(m.sum()), "oracle_brake_rate": float(ro[m].mean()),
       "agree_with_oracle_scope": float((rs == ro)[m].mean()), "agree_with_oracle_fixed": float((rf == ro)[m].mean()),
       "scope_misses": int((ro & ~rs)[m].sum()), "fixed_misses": int((ro & ~rf)[m].sum()),
       "scope_false_alarms": int((~ro & rs)[m].sum()), "fixed_false_alarms": int((~ro & rf)[m].sum())}
print("decision agreement:", dec)
out["_downstream_brake_decision"] = dec
json.dump(out, open("results_buckets.json", "w"), indent=1)
