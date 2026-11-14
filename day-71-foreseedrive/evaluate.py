"""Evaluate base vs ForeseeDrive variants and token diagnostics on the synthetic test split.

    python evaluate.py --seed 0           -> checkpoints/metrics_s0.json + printed table
Metrics (NAVSIM-flavoured, vs the TRUE future): PDMS, NC, DAC, TTC(clearance > 0.5 m), Comfort, EP,
collision %, ADE/FDE to the expert, best-of-K ADE / PDMS (candidate SUPPORT), share of candidates degraded
vs their base counterpart, guard trigger rate, PDMS on scenes with a hidden future event vs quiet ones.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import yaml

from foresee import data as D
from foresee import losses as L
from foresee import world as W
from foresee.model import inject, n_params
from train import CFG, CK, batched, build_cond, get_data, load_base, load_pred


def pdm_t(traj: torch.Tensor, b: dict) -> dict:
    """Batched torch version of world.pdm_score.  traj (B, K, H, 2) -> dict of (B, K)."""
    m, _ = L.clearance_t(traj, b["fut"], b)
    nc = (m > 0).float()
    y = traj[..., 1]
    dac = ((y > W.Y_MIN - 0.25).all(-1) & (y < W.Y_MAX + 0.25).all(-1)).float()
    ttc = (m > 0.5).float()
    p = torch.cat([torch.zeros_like(traj[..., :1, :]), traj], -2)
    v = (p[..., 1:, :] - p[..., :-1, :]).norm(dim=-1) / W.DT
    v = torch.cat([b["v_ego"][:, None, None].expand(-1, traj.shape[1], 1), v], -1)
    acc = (v[..., 1:] - v[..., :-1]) / W.DT
    jerk = (acc[..., 1:] - acc[..., :-1]) / W.DT
    comf = ((acc.abs().amax(-1) <= 6.5) & (jerk.abs().amax(-1) <= 12.0)).float()
    ep = (traj[..., -1, 0] / b["expert"][:, None, -1, 0].clamp(min=1.0)).clamp(0, 1)
    pdms = nc * dac * (5 * ttc + 2 * comf + 5 * ep) / 12
    return dict(pdms=pdms, nc=nc, dac=dac, ttc=ttc, comf=comf, ep=ep, clear=m)


def summarize(traj, logit, b, tb, guard_bad=None):
    B = traj.shape[0]
    sel = logit.argmax(-1)
    ar = torch.arange(B)
    s = pdm_t(traj, b)
    sb = pdm_t(tb, b)
    st = traj[ar, sel]
    ade = (st - b["expert"]).norm(dim=-1).mean(-1)
    fde = (st[:, -1] - b["expert"][:, -1]).norm(dim=-1)
    ade_all = (traj - b["expert"][:, None]).norm(dim=-1).mean(-1)            # (B, K)
    hidden = (b["event"] > 0).any(-1)
    out = {k: s[k][ar, sel].mean().item() for k in ("pdms", "nc", "dac", "ttc", "comf", "ep")}
    out.update(collision=100 * (1 - s["nc"][ar, sel].mean().item()),
               ade=ade.mean().item(), fde=fde.mean().item(),
               best_of_k_ade=ade_all.min(-1).values.mean().item(),
               best_of_k_pdms=s["pdms"].max(-1).values.mean().item(),
               degraded=100 * (s["pdms"] < sb["pdms"] - 0.05).float().mean().item(),
               moved=100 * ((traj - tb).abs().amax((-1, -2)) > 1e-6).float().mean().item(),
               pdms_hidden=s["pdms"][ar, sel][hidden].mean().item(),
               pdms_quiet=s["pdms"][ar, sel][~hidden].mean().item(),
               coll_hidden=100 * (1 - s["nc"][ar, sel][hidden].mean().item()))
    if guard_bad is not None:
        out["guard_trigger"] = 100 * guard_bad.float().mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG)); tc = cfg["train"]
    torch.manual_seed(0)
    _, te = get_data(cfg)
    n = len(te["v_ego"])
    base, pred = load_base(cfg), load_pred(cfg)
    tb, lb = batched(lambda i: base(D.batch(te, i))[:2], n)
    Pe, Pc, Pr = batched(lambda i: pred(D.batch(te, i), tb[i]), n)
    tE, tC, tR = L.teacher_tokens(tb, te)
    v_pred = inject(Pe, Pc, Pr, from_logits=True)
    v_orc = inject(tE, tC, tR, from_logits=False)
    perm = torch.roll(torch.arange(n), 1)                                    # shuffled = tokens of another scene
    v_shuf = v_pred[perm]
    v_zero = torch.zeros_like(v_pred)

    # token prediction quality
    c_acc = ((torch.sigmoid(Pc) >= 0.5).float() == tC).float().mean().item()
    msk = tC > 0
    r_acc_risky = ((torch.sigmoid(Pr) >= 0.5) & (tR > 0) & msk).sum().item() / max(1, ((tR > 0) & msk).sum().item())
    e_mae = ((Pe - tE).abs().mean(-1)[msk]).mean().item()
    tok = dict(conf_acc=c_acc, risk_recall=r_acc_risky, e_mae=e_mae,
               live_tokens_pred=(v_pred.abs().sum(-1) > 0).float().sum(-1).mean().item(),
               live_tokens_teacher=(v_orc.abs().sum(-1) > 0).float().sum(-1).mean().item())

    res = {"base": summarize(tb, lb, te, tb)}
    cond = {}
    for var in ("full", "scorer", "naive"):
        fd = build_cond(cfg, base, var)
        fd.load_state_dict(torch.load(CK / f"cond_{var}_s{a.seed}.pt"), strict=False)
        fd.eval(); cond[var] = fd

    def run(var, v, guard):
        fd = cond[var]
        tf, lf = batched(lambda i: (lambda o: (o["traj"], o["logit"]))(fd(D.batch(te, i), v[i])), n)
        bad = None
        if guard:
            tf, bad = L.apply_guard(tf, tb, te, tc["eps_b"], tc["eps_g"])
        return tf, lf, bad

    t0 = time.time()
    for name, var, v, g in [("scorer_only_pred", "scorer", v_pred, False),
                            ("scorer_only_oracle", "scorer", v_orc, False),
                            ("foresee_pred", "full", v_pred, True),
                            ("foresee_pred_noguard", "full", v_pred, False),
                            ("foresee_shuffled", "full", v_shuf, True),
                            ("foresee_oracle", "full", v_orc, True),
                            ("naive_pred", "naive", v_pred, False),
                            ("naive_oracle", "naive", v_orc, False)]:
        tf, lf, bad = run(var, v, g)
        res[name] = summarize(tf, lf, te, tb, bad)
    # exact zero-token control
    tz, lz, _ = run("full", v_zero, False)
    zero = dict(max_abs_traj_diff=(tz - tb).abs().max().item(), max_abs_logit_diff=(lz - lb).abs().max().item(),
                same_selection=100 * (lz.argmax(-1) == lb.argmax(-1)).float().mean().item())
    params = dict(base=n_params(base), predictor=n_params(pred),
                  cond_full=n_params(cond["full"], trainable=True), cond_scorer=n_params(cond["scorer"], trainable=True))
    # CPU latency (batch 1, 1 thread)
    torch.set_num_threads(1)
    b1 = D.batch(te, torch.arange(1))
    def lat(fn, reps=30):
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); fn(); ts.append(1000 * (time.perf_counter() - t))
        return sorted(ts)[reps // 2]
    with torch.no_grad():
        ms_base = lat(lambda: base(b1))
        ms_fd = lat(lambda: cond["full"](b1, inject(*pred(b1, base(b1)[0]), from_logits=True)))
    out = dict(seed=a.seed, tokens=tok, zero_token=zero, params=params, cpu_ms=dict(base=ms_base, foresee=ms_fd),
               results=res, n_test=n, hidden_event_scenes=int((te["event"] > 0).any(-1).sum()))
    json.dump(out, open(CK / f"metrics_s{a.seed}.json", "w"), indent=1)
    cols = ["pdms", "nc", "ttc", "ep", "collision", "ade", "best_of_k_ade", "best_of_k_pdms", "degraded", "pdms_hidden", "pdms_quiet"]
    print(f"{'config':24s} " + " ".join(f"{c[:11]:>11s}" for c in cols))
    for k, r in res.items():
        print(f"{k:24s} " + " ".join(f"{r[c]:11.3f}" for c in cols) + (f"  guard {r['guard_trigger']:.2f}%" if "guard_trigger" in r else ""))
    print("tokens", tok); print("zero-token", zero); print("params", params, "cpu ms", out["cpu_ms"])


if __name__ == "__main__":
    main()
