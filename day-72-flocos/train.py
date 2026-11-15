"""Training.  python train.py --stage fm | set | detreg | mtp  [--config config.yaml] [--seed 0] [--tag s0]

  fm      stage 1: conditional flow matching on fitted coefficient tubes
  set     stage 2: + coverage loss + union-area loss back-propagated through the NFE-step Euler sampler
  detreg  baseline: deterministic single-trajectory regression (MSE)
  mtp     baseline: K-mode trajectory prediction (winner-takes-all + mode CE)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from flocos import data as D
from flocos.model import DetReg, FloCOS, MTP, footprint_set, set_losses

ROOT = Path(__file__).parent
CK = ROOT / "checkpoints"
torch.set_num_threads(2)


def get_data(cfg, split):
    dc = cfg["data"]
    return D.get(dc[f"n_{split}"], dc[f"seed_{split}"], ROOT / "data" / f"{split}.pt")


def build_flocos(cfg):
    m = cfg["model"]
    return FloCOS(m["d"], m["width"], m["blocks"])


def epochs(n, bs, ep, gen):
    for e in range(ep):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n - bs + 1, bs):
            yield e, perm[i:i + bs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fm", "set", "detreg", "mtp"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default="s0")
    ap.add_argument("--lambda-area", type=float, default=None)
    ap.add_argument("--fm-tag", default="s0", help="stage-1 checkpoint to start stage 2 from")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    tc, mc = cfg["train"], cfg["model"]
    seed = tc["seed"] if a.seed is None else a.seed
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    tr = get_data(cfg, "train")
    va = get_data(cfg, "val")
    n = len(tr["v0"])
    CK.mkdir(exist_ok=True)
    log, t0 = [], time.time()

    if a.stage in ("fm", "detreg", "mtp"):
        model = {"fm": lambda: build_flocos(cfg), "detreg": DetReg, "mtp": lambda: MTP(tc["mtp_K"])}[a.stage]()
        ep = tc["fm_epochs"] if a.stage == "fm" else tc["baseline_epochs"]
        opt = torch.optim.AdamW(model.parameters(), tc["lr"], weight_decay=1e-4)
        steps = ep * (n // tc["batch"])
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, tc["lr"], total_steps=steps, pct_start=0.05)
        last = -1
        for e, idx in epochs(n, tc["batch"], ep, gen):
            b = D.batch(tr, idx)
            if a.stage == "fm":
                loss = model.fm_loss(b)
            elif a.stage == "detreg":
                loss = F.smooth_l1_loss(model(b)[:, 0], b["fut"])
            else:
                traj, logit = model(b)
                err = (traj - b["fut"][:, None]).norm(dim=-1).mean(-1)                 # (B, K) ADE
                best = err.argmin(-1)
                loss = err[torch.arange(len(best)), best].mean() + 0.5 * F.cross_entropy(logit, best)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            if e != last:
                last = e
                with torch.no_grad():
                    vb = D.batch(va, slice(0, 1000))
                    if a.stage == "fm":
                        torch.manual_seed(123)
                        vl = model.fm_loss(vb).item()
                    elif a.stage == "detreg":
                        vl = (model(vb)[:, 0] - vb["fut"]).norm(dim=-1).mean().item()
                    else:
                        vl = (model(vb)[0] - vb["fut"][:, None]).norm(dim=-1).mean(-1).min(-1)[0].mean().item()
                log.append(dict(epoch=e, train=loss.item(), val=vl, t=time.time() - t0))
                print(log[-1], flush=True)
        name = {"fm": f"flocos_fm_{a.tag}.pt", "detreg": "detreg.pt", "mtp": "mtp.pt"}[a.stage]
        torch.save(model.state_dict(), CK / name)
    else:
        lam_a = tc["lambda_area"] if a.lambda_area is None else a.lambda_area
        model = build_flocos(cfg)
        model.load_state_dict(torch.load(CK / f"flocos_fm_{a.fm_tag}.pt"))
        opt = torch.optim.AdamW(model.parameters(), tc["set_lr"], weight_decay=1e-4)
        steps = tc["set_steps"]
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for e in range(steps):
            idx = torch.randint(0, n, (tc["set_batch"],), generator=gen)
            b = D.batch(tr, idx)
            c, s = model.predict_set(b, mc["M"], mc["nfe"])                            # differentiable sampler
            l_cov, area = set_losses(c, s, b["fut"], b["head"])
            l_fm = model.fm_loss(b)
            loss = l_fm + tc["lambda_cov"] * l_cov + lam_a * area
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            if e % 25 == 0 or e == steps - 1:
                log.append(dict(step=e, fm=l_fm.item(), cov=l_cov.item(), area=area.item(), t=time.time() - t0))
                print(log[-1], flush=True)
        suffix = "" if a.lambda_area is None else f"_la{lam_a:g}"
        torch.save(model.state_dict(), CK / f"flocos_set_{a.tag}{suffix}.pt")
    json.dump(log, open(CK / f"log_{a.stage}_{a.tag}.json", "w"), indent=1)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
