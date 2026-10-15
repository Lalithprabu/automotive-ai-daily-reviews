import torch
from .data import make_dataset, to_tensors
from .model import XTrackUQ
from .uncertainty import propagate_samples, ConformalCalibrator

def load_ensemble(path="checkpoint.pt"):
    import yaml
    ck = torch.load(path, map_location="cpu"); cfg = {**ck["cfg"], **{k: v for k, v in yaml.safe_load(open("config.yaml")).items() if k.startswith("n_") and k != "n_samples" and k != "n_neighbors"}}; kw = dict(t_pred=cfg["t_pred"], dt=cfg["dt"])
    ens = []
    for v, sd in zip(ck["variants"], ck["ens"]):
        m = XTrackUQ(v["hidden"], heads=v["heads"], dec_layers=v["dec_layers"], **kw); m.load_state_dict(sd); m.eval(); ens.append(m)
    return cfg, ens

def fit_calibrator(cfg, ens):
    cal = to_tensors(make_dataset(cfg["n_cal"], cfg, cfg["seed"] + 2))     # same split train.py used
    o = propagate_samples(ens, cal, cfg["n_samples"], 1)
    return ConformalCalibrator(1 - cfg["coverage_target"]).fit(cal["pos"], o["mean_traj"], o["cov_total"])
