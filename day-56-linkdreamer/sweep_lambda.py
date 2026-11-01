"""Sensitivity of the controller to the communication weight (reuses trained wm/det; retrains only the TS controller).
python sweep_lambda.py   -> checkpoints/sweep.json.  NOTE: evaluated on the same held-out episodes as train.py; the default 0.10 was fixed a priori."""
import json, numpy as np, torch
from linkdreamer import agent
from linkdreamer.synthetic import make_dataset, link_sequence
from linkdreamer.models import LatentWorldModel, FusionDetector, TSNeuroFuzzy, build_modes, decision_state, mode_mask
from linkdreamer.metrics import heat_loss
torch.manual_seed(0); torch.set_num_threads(2); rng = np.random.default_rng(5)
wm, det = LatentWorldModel(), FusionDetector()
wm.load_state_dict(torch.load("checkpoints/wm.pt")); det.load_state_dict(torch.load("checkpoints/det.pt")); wm.eval(); det.eval()
tr, te = make_dataset(100, 1), make_dataset(120, 2)
N = 6000; b = agent.sample_batch(tr, rng, N)
with torch.no_grad():
    coll, sig, wsig, reuse = build_modes(wm, b["col"], b["hist"], b["gaps"], b["ages"], b["has_mem"])
    risk = torch.stack([heat_loss(det(b["ego"], coll[:, m], sig[:, m], torch.full((N,), m)), b["tgt"]) for m in range(5)], 1)
    q = torch.tensor(rng.choice([0., .5, 1.], N), dtype=torch.float32); bud = torch.rand(N)
    state = decision_state(b["ego"], reuse, wsig, b["ages"], b["has_mem"], q, bud)
agent.RISK_REF["v"] = float(risk[:, 0].mean()); mask = mode_mask(q, b["has_mem"])
qs = link_sequence(np.random.default_rng(3), 120); out = {}
for lc in (0.0, 0.02, 0.05, 0.10, 0.20):
    agent.LAMBDA_C = lc; cost = agent.cost_table(risk, state)
    torch.manual_seed(0); ct = TSNeuroFuzzy(centers=state[torch.randperm(N)[:9]]); opt = torch.optim.Adam(ct.parameters(), 1e-2)
    for s in range(1500):
        i = torch.randint(0, N, (512,)); p = torch.softmax(ct(state[i]).masked_fill(~mask[i], -1e9), 1)
        loss = (p * cost[i]).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    o = agent.run(te, np.arange(120), qs, "ctrl", wm, det, ct.eval())
    out[str(lc)] = dict(ap_all=o["all"][0], ap_outage=o["outage"][0], comm_per_frame=o["comm_per_frame"], mode_share=[float((o["modes"][:, 2:] == m).float().mean()) for m in range(5)])
    print(lc, {k: (round(v, 3) if isinstance(v, float) else [round(x, 2) for x in v]) for k, v in out[str(lc)].items()}, flush=True)
json.dump(out, open("checkpoints/sweep.json", "w"), indent=1)
