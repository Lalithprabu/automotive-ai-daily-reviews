"""Cooperative-fusion evaluation shared by train.py, simulate.py and tests."""
from __future__ import annotations
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .data import simulate_scene, sample_agents, observe
from .fusion import DOP, fuse_agents


@torch.no_grad()
def agent_dops(model, rng, agent_pos, P, V, end, cfg, agent_id, learned=True, sigma_fixed=1.5):
    """Run the model on every visible object of one agent whose last frame is `end`; return DOPs + gt idx."""
    T = cfg["seq_len"]
    win = P[end - T + 1:end + 1]
    meas, vis, feats = observe(rng, agent_pos, win, cfg["sensor_range"])
    idx = np.where(vis)[0]
    if len(idx) == 0:
        return [], []
    out = model(torch.from_numpy(feats[:, idx].transpose(1, 0, 2).copy()))
    vel, lv = out["vel"].numpy(), out["vel_logvar"].exp().numpy()
    ps, pd = out["pos_sigma"].numpy() * 1.25, torch.sigmoid(out["dyn_logit"]).numpy()  # 1.25 ~ 1/E|z|
    dops = []
    for k, i in enumerate(idx):
        pv = ps[k] ** 2 if learned else np.full(3, sigma_fixed ** 2)
        vv = lv[k] if learned else np.full(3, 4.0)
        dops.append(DOP(meas[-1, i].astype(np.float64), vel[k].astype(np.float64), pv.astype(np.float64),
                        vv.astype(np.float64), float(pd[k]), agent_id))
    return dops, list(idx)


def score(tracks, gt_pos, match_gate=6.0):
    """Hungarian-match fused tracks to GT at t_now. Returns (mean err of matched, recall@2m, n_tracks)."""
    if not tracks:
        return np.nan, 0.0, 0
    T = np.array([t.mu for t in tracks]); C = np.linalg.norm(T[:, None] - gt_pos[None], axis=-1)
    r, c = linear_sum_assignment(C)
    errs = [C[i, j] for i, j in zip(r, c) if C[i, j] < match_gate]
    rec = sum(e < 2.0 for e in errs) / len(gt_pos)
    return (float(np.mean(errs)) if errs else np.nan), rec, len(tracks)


METHODS = {
    # name: (use_collab, compensate, learned_unc, bayesian)
    "ego_only":                  (False, False, True,  True),
    "late_fusion_stale":         (True,  False, False, False),   # OLD WAY: avg stale boxes
    "bayes_learned_no_latcomp":  (True,  False, True,  True),
    "bayes_fixed_cov_latcomp":   (True,  True,  False, True),
    "cocam4d_full":              (True,  True,  True,  True),     # NEW WAY
}


def run_eval(model, cfg, latencies_ms=(0, 100, 300, 500), n_scenes=60, seed=123):
    model.eval()
    res = {m: {L: {"err": [], "rec": []} for L in latencies_ms} for m in METHODS}
    T, dt = cfg["seq_len"], cfg["dt"]
    max_lag = int(round(max(latencies_ms) / 1000 / dt))
    for s in range(n_scenes):
        rng = np.random.default_rng(seed + s)
        P, V, _ = simulate_scene(rng, cfg["n_objects"], T + max_lag, dt)
        A = sample_agents(rng, cfg["n_agents"])
        now = P.shape[0] - 1
        for L in latencies_ms:
            lag = int(round(L / 1000 / dt))
            for name, (collab, comp, learned, bayes) in METHODS.items():
                per, deltas = [], []
                for k, a in enumerate(A):
                    if k > 0 and not collab:
                        continue
                    end = now if k == 0 else now - lag
                    # same rng stream per (scene, agent) so methods share identical observations
                    r2 = np.random.default_rng(seed * 7919 + s * 31 + k * 5 + lag)
                    d, _ = agent_dops(model, r2, a, P, V, end, cfg, k, learned=learned)
                    per.append(d); deltas.append(0.0 if k == 0 else lag * dt)
                tr = fuse_agents(per, deltas, compensate=comp, pose_sigma=cfg["pose_sigma"],
                                 bayesian=bayes)
                e, rec, _ = score(tr, P[now])
                res[name][L]["err"].append(e); res[name][L]["rec"].append(rec)
    return {m: {L: {"mean_pos_err_m": float(np.nanmean(v["err"])), "recall_2m": float(np.mean(v["rec"]))}
                for L, v in d.items()} for m, d in res.items()}
