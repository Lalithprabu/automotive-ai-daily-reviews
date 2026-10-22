"""Pretrain a generic base model, then compare six ways of adapting it:

  zero-shot | centralized FT | federated-from-scratch | federated full FT |
  LoRA-FedFT (F=1) | LoRA-FedFT (F=5)

Writes results/results.json, results/demo_trace.npz (per-round predictions for
simulate.py) and checkpoints.   Usage:  python train.py --config config.yaml
"""
from __future__ import annotations

import argparse, copy, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.lora_fedft import (TrajTransformer, apply_lora, payload_bytes, payload_numel, per_style_ade,
                            predict, run_federated, trajectory_metrics, uplink_seconds)
from src.lora_fedft.data import STYLES, concat, make_clients, make_pretrain_set


def fit(model, data, epochs, lr, bs, seed):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(data["x"]), generator=gen)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = F.smooth_l1_loss(model(data["x"][idx]), data["y"][idx], beta=0.05)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def rounds_to_target(hist, target):
    for r, a, b in zip(hist["round"], hist["ADE"], hist["uplink_bytes"]):
        if a <= target:
            return r, b
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.set_num_threads(2)
    out = Path(a.out); out.mkdir(exist_ok=True)
    t0 = time.time()
    mk = lambda: TrajTransformer(**cfg["model"])
    D, FD = cfg["data"], cfg["federated"]

    clients, val = make_clients(D["num_clients"], D["n_train"], D["n_val"], D["own_frac"], cfg["seed"])
    pooled = concat(clients)
    print(f"clients={len(clients)}  train/client={D['n_train']}  val={len(val['x'])}")

    # ---- 1. generic "foundation" base: straight walking only ------------------------
    base = mk()
    pre = make_pretrain_set(D["pretrain_n"], cfg["seed"] + 1)
    fit(base, pre, cfg["pretrain"]["epochs"], cfg["pretrain"]["lr"], cfg["pretrain"]["batch_size"], cfg["seed"])
    torch.save(base.state_dict(), out / "base.pt")
    res = {"config": cfg, "n_params_base": payload_numel(base)}
    ev = lambda m: (trajectory_metrics(predict(m, val["x"]), val["y"]),
                    per_style_ade(predict(m, val["x"]), val["y"], val["style"], STYLES))
    zs, zs_ps = ev(base)
    res["zero_shot"] = {**zs, "per_style_ADE": zs_ps}
    print("zero-shot", zs, zs_ps, f"[{time.time()-t0:.0f}s]")

    # ---- demo scenes for simulate.py (held-out, one per hard style + one crossing) --
    demo_idx = [int(np.where(val["style"] == s)[0][3]) for s in (1, 2, 3)]
    trace = {"idx": np.array(demo_idx), "world": val["world"][demo_idx], "R": val["R"][demo_idx],
             "origin": val["origin"][demo_idx], "style": val["style"][demo_idx]}
    xd = val["x"][demo_idx]
    trace["pred_round0"] = predict(base, xd).numpy()

    def tracer(key):
        buf = {}
        def cb(r, m, _):
            buf[r] = predict(m, xd).numpy()
        return buf, cb

    # ---- 2. centralized full fine-tuning (accuracy ceiling) -------------------------
    cen = copy.deepcopy(base)
    fit(cen, pooled, cfg["centralized"]["epochs"], cfg["centralized"]["lr"], cfg["centralized"]["batch_size"], cfg["seed"])
    cm, cps = ev(cen)
    res["centralized"] = {**cm, "per_style_ADE": cps}
    target = cm["ADE"] * cfg["target_ade_slack"]
    res["target_ade"] = target
    print("centralized", cm, f"[{time.time()-t0:.0f}s]")

    runs = {}
    # ---- 3. federated variants -----------------------------------------------------
    specs = [("lora_F%d" % F_, "lora", F_) for F_ in FD["local_epochs_lora"]] + \
            [("full_ft", "full", FD["local_epochs_full"]), ("full_ft_F%d" % max(FD["local_epochs_lora"]), "full", max(FD["local_epochs_lora"])), ("fed_scratch", "scratch", FD["local_epochs_full"])]
    for name, kind, F_ in specs:
        torch.manual_seed(cfg["seed"])
        m = mk() if kind == "scratch" else copy.deepcopy(base)
        if kind == "lora":
            apply_lora(m, **cfg["lora"])
        lr = FD["lr_lora"] if kind == "lora" else FD["lr_full"]
        buf, cb = tracer(name)
        hist = run_federated(m, clients, val, FD["rounds"], F_, lr, FD["batch_size"], cfg["seed"], on_round=cb)
        r_t, b_t = rounds_to_target(hist, target)
        m_final, ps = ev(m)
        info = {**m_final, "per_style_ADE": ps, "payload_bytes_per_upload": hist["payload_bytes_per_upload"],
                "trainable_params": payload_numel(m), "rounds_to_target": r_t, "uplink_bytes_to_target": b_t,
                "total_uplink_bytes": hist["uplink_bytes"][-1],
                "uplink_seconds_total_assumed": uplink_seconds(hist["uplink_bytes"][-1], cfg["comm"]["assumed_uplink_mbps"]),
                "history": hist}
        runs[name] = info
        trace[f"pred_{name}"] = np.stack([buf[r] for r in sorted(buf)])
        if kind == "lora":
            torch.save({k: v for k, v in m.state_dict().items() if "lora_" in k}, out / f"{name}_adapters.pt")
        print(f"{name:12s} ADE={m_final['ADE']:.3f} FDE={m_final['FDE']:.3f} MR={m_final['MR@2s/1m']:.3f} "
              f"payload={hist['payload_bytes_per_upload']/1e3:.0f}KB rounds->target={r_t} [{time.time()-t0:.0f}s]")
    res["runs"] = runs
    res["lora_to_full_payload_ratio"] = runs["lora_F1"]["payload_bytes_per_upload"] / runs["full_ft"]["payload_bytes_per_upload"]
    np.savez(out / "demo_trace.npz", **trace)
    json.dump(res, open(out / "results.json", "w"), indent=1, default=float)
    print("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
