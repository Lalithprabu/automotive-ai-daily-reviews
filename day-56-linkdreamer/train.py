"""3-stage recipe + closed-loop evaluation.   python train.py [--quick]
 stage 1: world model (Gaussian NLL on the missing collaborator BEV)   stage 2: detector on all five modes
 stage 3: Takagi-Sugeno controller minimising detection-risk + comm + latency + uncertainty cost (risk table from stage 2)."""
import argparse, json, time
import numpy as np, torch
import torch.nn.functional as F
from linkdreamer import agent
from linkdreamer.synthetic import make_dataset, link_sequence, T_EP
from linkdreamer.models import LatentWorldModel, FusionDetector, TSNeuroFuzzy, build_modes, decision_state, mode_mask
from linkdreamer.metrics import heat_loss

torch.manual_seed(0); torch.set_num_threads(2)


def main(a):
    t0 = time.time(); rng = np.random.default_rng(0)
    tr, te = make_dataset(a.n_train, 1), make_dataset(a.n_test, 2)
    print(f"data {time.time() - t0:.0f}s", flush=True)
    wm, det = LatentWorldModel(), FusionDetector()
    # ---- stage 1: world model
    opt = torch.optim.Adam(wm.parameters(), 2e-3)
    for s in range(a.wm_steps):
        b = agent.sample_batch(tr, rng, 32)
        mean, lv = wm(b["hist"], b["gaps"], b["ages"])
        nll = (0.5 * (lv + (b["col"] - mean) ** 2 / lv.exp())).mean()
        loss = nll + 2.0 * F.mse_loss(mean, b["col"])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0); opt.step()
        if s % 100 == 0: print(f"wm {s} nll {nll.item():.3f} mse {F.mse_loss(mean, b['col']).item():.4f} {time.time() - t0:.0f}s", flush=True)
    wm.eval()
    # ---- stage 2: detector trained on random modes
    opt = torch.optim.Adam(det.parameters(), 2e-3)
    for s in range(a.det_steps):
        b = agent.sample_batch(tr, rng, 32)
        with torch.no_grad():
            coll, sig, _, _ = build_modes(wm, b["col"], b["hist"], b["gaps"], b["ages"], b["has_mem"])
        mode = torch.tensor(rng.integers(0, 5, 32)); ar = torch.arange(32)
        loss = heat_loss(det(b["ego"], coll[ar, mode], sig[ar, mode], mode), b["tgt"], "mean")
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 200 == 0: print(f"det {s} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    det.eval()
    # ---- stage 3: controller from a risk table
    N = a.table; b = agent.sample_batch(tr, rng, N)
    with torch.no_grad():
        coll, sig, wsig, reuse = build_modes(wm, b["col"], b["hist"], b["gaps"], b["ages"], b["has_mem"])
        risk = torch.stack([heat_loss(det(b["ego"], coll[:, m], sig[:, m], torch.full((N,), m)), b["tgt"]) for m in range(5)], 1)
        q = torch.tensor(rng.choice([0., .5, 1.], N), dtype=torch.float32); bud = torch.rand(N)
        state = decision_state(b["ego"], reuse, wsig, b["ages"], b["has_mem"], q, bud)
    agent.RISK_REF["v"] = float(risk[:, 0].mean())
    print("mean risk per mode", [round(float(x), 4) for x in risk.mean(0)], "ref", agent.RISK_REF["v"], flush=True)
    cost, mask = agent.cost_table(risk, state), mode_mask(q, b["has_mem"])
    ctrl = TSNeuroFuzzy(centers=state[torch.randperm(N)[:9]])
    opt = torch.optim.Adam(ctrl.parameters(), 1e-2)
    for s in range(a.ctrl_steps):
        i = torch.randint(0, N, (512,))
        p = torch.softmax(ctrl(state[i]).masked_fill(~mask[i], -1e9), 1)
        loss = (p * cost[i]).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 300 == 0: print(f"ctrl {s} expected cost {loss.item():.4f}", flush=True)
    oracle = cost.masked_fill(~mask, 1e9).min(1).values.mean().item()
    print(f"controller cost {loss.item():.4f}  oracle {oracle:.4f}", flush=True)
    for n, m in (("wm", wm), ("det", det), ("ctrl", ctrl)): torch.save(m.state_dict(), f"checkpoints/{n}.pt")
    json.dump(dict(risk_ref=agent.RISK_REF["v"]), open("checkpoints/meta.json", "w"))
    # ---- closed-loop evaluation on held-out episodes
    idx = np.arange(a.n_test); q = link_sequence(np.random.default_rng(3), a.n_test)
    res = {}
    for name, pol in (("ego-only", "ego"), ("reuse (motion-comp.)", "reuse"), ("world-model imagine", "imagine"), ("LinkDreamer-style controller", "ctrl"), ("upper ref (no outage)", "upper")):
        o = agent.run(te, idx, q, pol, wm, det, ctrl)
        res[name] = dict(ap_outage=o["outage"][0], f1_outage=o["outage"][1], ap_all=o["all"][0], ap_degraded=o["degraded"][0], ap_up=o["up"][0],
                         comm_per_frame=o["comm_per_frame"], mode_share=[float((o["modes"][:, 2:] == m).float().mean()) for m in range(5)])
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items()}, flush=True)
    res["_outage_frame_share"] = float((q[:, 2:] == 0).mean())
    json.dump(res, open("checkpoints/results.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=300); ap.add_argument("--n_test", type=int, default=120)
    ap.add_argument("--wm_steps", type=int, default=1500); ap.add_argument("--det_steps", type=int, default=900)
    ap.add_argument("--ctrl_steps", type=int, default=1500); ap.add_argument("--table", type=int, default=6000)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick: a.n_train, a.n_test, a.wm_steps, a.det_steps, a.ctrl_steps, a.table = 40, 20, 60, 60, 100, 600
    main(a)
