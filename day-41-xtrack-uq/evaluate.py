"""Evaluate baseline / MC-dropout / deep-ensemble: accuracy, raw vs conformal coverage, interval width,
plus an exchangeability stress test (test set with 3x aleatoric noise the calibrator never saw)."""
import sys, json, numpy as np, torch
sys.path.insert(0, "src")
from xtrack_uq.data import make_dataset, to_tensors
from xtrack_uq.model import XTrackUQ
from xtrack_uq.uncertainty import propagate_samples, ConformalCalibrator

def ade_fde(mean, pos):
    e = (mean - pos).norm(dim=-1); return e.mean().item(), e[:, -1].mean().item()

def evaluate(cfg, D, base, mcd, ens):
    shifted = to_tensors(make_dataset(cfg["n_test"], cfg, cfg["seed"] + 4, noise_scale=3.0))
    res = {}
    for name, models, passes in [("baseline", [base], 1), ("mc_dropout", [mcd], cfg["mc_passes"]), ("deep_ensemble", ens, 1)]:
        torch.manual_seed(1)
        oc = propagate_samples(models, D["cal"], cfg["n_samples"], passes)
        cp = ConformalCalibrator(1 - cfg["coverage_target"]).fit(D["cal"]["pos"], oc["mean_traj"], oc["cov_total"])
        r = {}
        for tag, T in [("", D["test"]), ("_shift", shifted)]:
            ot = propagate_samples(models, T, cfg["n_samples"], passes)
            ade, fde = ade_fde(ot["mean_traj"], T["pos"])
            cov_raw = cp.coverage(T["pos"], ot["mean_traj"], ot["cov_total"], calibrated=False).mean().item()
            cov_cp = cp.coverage(T["pos"], ot["mean_traj"], ot["cov_total"], calibrated=True)
            sx = ot["cov_total"][..., 0, 0].sqrt()
            err = (ot["mean_traj"] - T["pos"]).norm(dim=-1).flatten()
            unc = ot["cov_total"].diagonal(dim1=-2, dim2=-1).sum(-1).sqrt().flatten()
            r.update({f"ADE{tag}": ade, f"FDE{tag}": fde, f"coverage_raw{tag}": cov_raw, f"coverage_cp{tag}": cov_cp.mean().item(),
                      f"coverage_cp_min_step{tag}": cov_cp.mean(0).min().item(),
                      f"mpiw_x_raw{tag}": (2 * 1.96 * sx).mean().item(), f"mpiw_x_cp{tag}": (2 * cp.q.view(1, -1) * sx).mean().item()})
            if tag == "":
                r["pearson_err_unc"] = float(np.corrcoef(err.numpy(), unc.numpy())[0, 1])
                tr = ot["cov_total"].diagonal(dim1=-2, dim2=-1).sum(-1)
                r["epi_share"] = (ot["cov_epi"].diagonal(dim1=-2, dim2=-1).sum(-1) / tr.clamp_min(1e-9)).mean().item()
        res[name] = r
        print(name, {k: round(v, 3) for k, v in r.items()}, flush=True)
    return res

if __name__ == "__main__":
    from xtrack_uq.inference import load_ensemble
    cfg, ens = load_ensemble(); ck = torch.load("checkpoint.pt"); kw = dict(t_pred=cfg["t_pred"], dt=cfg["dt"]); H = cfg["hidden"]
    base = XTrackUQ(H, **kw); base.load_state_dict(ck["base"]); mcd = XTrackUQ(H, p_drop=0.2, **kw); mcd.load_state_dict(ck["mcd"])
    D = {k: to_tensors(make_dataset(cfg["n_" + k], cfg, cfg["seed"] + i)) for i, k in [(2, "cal"), (3, "test")]}
    res = evaluate(cfg, D, base.eval(), mcd.eval(), ens); res["params_single_model"] = sum(p.numel() for p in base.parameters())
    json.dump(res, open("metrics.json", "w"), indent=2)
