"""Federated fine-tuning loop: vehicles train locally, an edge server averages.

The paper says only "server averaging" (algorithm not named).  This repo uses
sample-weighted FedAvg over exactly the tensors each vehicle uploads
(`trainable_state`): the A/B adapters for LoRA, every parameter for full FT.
NOTE: averaging A and B separately is not identical to averaging the products
B@A (the well-known LoRA-FedAvg inexactness); it is the simplest reading of
"server averaging" and is disclosed as such.
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from .lora import load_trainable_state, payload_bytes, trainable_state
from .metrics import predict, trajectory_metrics


def local_train(model, data, epochs: int, lr: float, batch_size: int, gen: torch.Generator) -> None:
    """`epochs` passes over one vehicle's private data (fresh Adam each round)."""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    n = len(data["x"])
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = F.smooth_l1_loss(model(data["x"][idx]), data["y"][idx], beta=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def fedavg(states: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    """Sample-weighted mean of the uploaded tensors."""
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()
    return {k: sum(wi * s[k] for wi, s in zip(w, states)) for k in states[0]}


def run_federated(
    model,                       # global model (LoRA-wrapped or plain)
    clients: List[dict],
    val: dict,
    rounds: int,
    local_epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    on_round: Optional[Callable[[int, torch.nn.Module, dict], None]] = None,
) -> dict:
    """Returns history with per-round val metrics + cumulative uplink bytes."""
    gen = torch.Generator().manual_seed(seed)
    worker = copy.deepcopy(model)
    per_upload = payload_bytes(model)
    hist = {"round": [], "ADE": [], "FDE": [], "MR@2s/1m": [], "uplink_bytes": [], "payload_bytes_per_upload": per_upload}

    def log(r, cum):
        m = trajectory_metrics(predict(model, val["x"]), val["y"])
        hist["round"].append(r); hist["uplink_bytes"].append(cum)
        for k in ("ADE", "FDE", "MR@2s/1m"):
            hist[k].append(m[k])
        if on_round:
            on_round(r, model, m)

    cum = 0
    log(0, cum)
    for r in range(1, rounds + 1):
        g_state = trainable_state(model)
        uploads, sizes = [], []
        for c in clients:
            load_trainable_state(worker, g_state)           # server -> vehicle (downlink not counted, as in the paper)
            local_train(worker, c, local_epochs, lr, batch_size, gen)
            uploads.append(trainable_state(worker))          # vehicle -> server (counted)
            sizes.append(len(c["x"]))
            cum += per_upload
        load_trainable_state(model, fedavg(uploads, sizes))
        log(r, cum)
    return hist


def uplink_seconds(total_bytes: float, rate_mbps: float, n_parallel: int = 1) -> float:
    """Sum of per-vehicle uplink occupancy (RaOT-style): bytes*8 / rate.

    rate_mbps is an ASSUMED per-vehicle uplink rate (the paper derives its rate
    from 3GPP TS 38.306; this repo does not reproduce that radio model).
    """
    return total_bytes * 8 / (rate_mbps * 1e6) / n_parallel
