"""Evaluate every occupancy predictor on the synthetic test split.

Metrics (per method):
  coverage        mean fraction of the TRUE footprint (45-point grid) inside the predicted union, over agents x steps
  full_cov        fraction of agent-steps whose true footprint is 100% covered
  agent_full      fraction of agents covered at EVERY step of the 3 s horizon
  area            mean union area per step (m^2, Monte-Carlo, 2048 pts in the exact bounding box)
  completion      ego merge completed without collision (planner uses the set as hard constraints)
  collision       ego collided with the true target
  python evaluate.py [--tag s0] [--set-ckpt flocos_set_s0.pt]
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from flocos import data as D
from flocos import geometry as G
from flocos import planner as PL
from flocos import world as W
from flocos.model import DetReg, MTP, footprint_set
from train import CK, ROOT, build_flocos, get_data

torch.set_num_threads(2)


def chunked(fn, n, bs=500):
    outs = [fn(slice(i, i + bs)) for i in range(0, n, bs)]
    return [torch.cat(z) for z in zip(*outs)]


def fixed_noise(n, M, seed=7):
    return torch.randn(n, M, G.COEF_DIM, generator=torch.Generator().manual_seed(seed))


def metrics(c, s, te):
    cov = torch.cat([G.coverage(c[i:i + 500], s[i:i + 500], te["fut"][i:i + 500], te["head"][i:i + 500])
                     for i in range(0, len(c), 500)])                                  # (N, H)
    g = torch.Generator().manual_seed(0)
    area = torch.cat([G.union_area(c[i:i + 100], s[i:i + 100], 2048, gen=g) for i in range(0, len(c), 100)])
    pl = PL.evaluate_planner(c, s, te)
    intent = te["intent"]
    out = dict(coverage=cov.mean().item(), full_cov=(cov >= 0.999).float().mean().item(),
               agent_full=(cov >= 0.999).all(-1).float().mean().item(), area=area.mean().item(),
               completion=pl["completion"].mean().item(), collision=pl["collision"].mean().item(),
               merged=pl["merged"].mean().item(),
               cov_by_intent={W.INTENT_NAMES[k]: cov[intent == k].mean().item() for k in range(W.N_INTENT)},
               area_by_intent={W.INTENT_NAMES[k]: area[intent == k].mean().item() for k in range(W.N_INTENT)})
    return out, cov, area, pl


def predictors(cfg, set_ckpt, tag):
    mc = cfg["model"]
    fm = build_flocos(cfg); fm.load_state_dict(torch.load(CK / f"flocos_fm_{tag}.pt")); fm.eval()
    fs = build_flocos(cfg); fs.load_state_dict(torch.load(CK / set_ckpt)); fs.eval()
    det = DetReg(); det.load_state_dict(torch.load(CK / "detreg.pt")); det.eval()
    mtp = MTP(cfg["train"]["mtp_K"]); mtp.load_state_dict(torch.load(CK / "mtp.pt")); mtp.eval()
    return fm, fs, det, mtp, mc


@torch.no_grad()
def run_all(cfg, te, set_ckpt, tag, ablations=True):
    fm, fs, det, mtp, mc = predictors(cfg, set_ckpt, tag)
    n = len(te["v0"])
    M, nfe = mc["M"], mc["nfe"]
    noise = fixed_noise(n, M)
    sets = {}
    sets["oracle (true footprint)"] = (te["fut"][:, None], G.rect_support(te["head"])[:, None])
    sets["DetReg (1 footprint)"] = footprint_set(chunked(lambda i: (det(D.batch(te, i)),), n)[0])
    sets["MTP-6 (6 footprints)"] = footprint_set(chunked(lambda i: (mtp(D.batch(te, i))[0],), n)[0])
    c, ex = D.physics_reach_box(te)
    sets["Physics reach set"] = (c, G.rect_support(None, ex=ex[..., 0], ey=ex[..., 1]))
    sets[f"FloCOS FM-only (M={M})"] = chunked(lambda i: fm.predict_set(D.batch(te, i), M, nfe, noise[i]), n)
    sets[f"FloCOS full (M={M})"] = chunked(lambda i: fs.predict_set(D.batch(te, i), M, nfe, noise[i]), n)
    if ablations:
        for nf in (1, 2, 8):
            sets[f"FloCOS full NFE={nf}"] = chunked(lambda i: fs.predict_set(D.batch(te, i), M, nf, noise[i]), n)
        for m in (1, 3, 12):
            nz = fixed_noise(n, m)
            sets[f"FloCOS full M={m}"] = chunked(lambda i: fs.predict_set(D.batch(te, i), m, nfe, nz[i]), n)
        nz = fixed_noise(n, M, seed=99)
        sets[f"FloCOS full (M={M}, other noise)"] = chunked(lambda i: fs.predict_set(D.batch(te, i), M, nfe, nz[i]), n)
    return sets, (fm, fs, det, mtp)


def latency(models, te, M, nfe):
    fm, fs, det, mtp = models
    torch.set_num_threads(1)
    b = D.batch(te, slice(0, 1))
    res = {}
    for name, fn in (("DetReg", lambda: det(b)), ("MTP-6", lambda: mtp(b)),
                     (f"FloCOS M={M} NFE={nfe}", lambda: fs.predict_set(b, M, nfe))):
        with torch.no_grad():
            for _ in range(10):
                fn()
            t = time.perf_counter()
            for _ in range(100):
                fn()
        res[name] = (time.perf_counter() - t) * 10.0                                   # ms per call
    torch.set_num_threads(2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tag", default="s0")
    ap.add_argument("--set-ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-ablations", action="store_true")
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(ROOT / a.config))
    te = get_data(cfg, "test")
    set_ckpt = a.set_ckpt or f"flocos_set_{a.tag}.pt"
    sets, models = run_all(cfg, te, set_ckpt, a.tag, not a.no_ablations)
    res = {}
    for k, (c, s) in sets.items():
        res[k] = metrics(c, s, te)[0]
        r = res[k]
        print(f"{k:34s} cov {r['coverage']:.3f}  full {r['full_cov']:.3f}  agent {r['agent_full']:.3f}  "
              f"area {r['area']:6.1f}  compl {r['completion']:.3f}  coll {r['collision']:.3f}  merged {r['merged']:.3f}",
              flush=True)
    lat = latency(models, te, cfg["model"]["M"], cfg["model"]["nfe"])
    print("latency ms", lat)
    out = a.out or f"metrics_{a.tag}.json"
    json.dump(dict(n_test=len(te["v0"]), results=res, latency_ms=lat, set_ckpt=set_ckpt), open(CK / out, "w"), indent=1)


if __name__ == "__main__":
    main()
