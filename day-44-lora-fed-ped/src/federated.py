"""FedAvg over only the trainable tensors (full model or LoRA adapters) with communication accounting."""
import copy
import torch
import torch.nn.functional as F
from .lora import trainable_state, load_state, n_params


def ade_fde(pred, gt):
    d = (pred - gt).norm(dim=-1)  # (B,P)
    return d.mean().item(), d[:, -1].mean().item()


@torch.no_grad()
def evaluate(model, obs, fut):
    model.eval()
    return ade_fde(model(obs), fut)


def local_train(model, data, epochs, lr, bs, seed):
    obs, fut = data
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(obs), generator=g)
        for i in range(0, len(obs), bs):
            idx = perm[i:i + bs]
            loss = F.smooth_l1_loss(model(obs[idx]), fut[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return len(obs)


def fedavg(states, weights):
    tot = float(sum(weights))
    return {k: sum(w * s[k] for s, w in zip(states, weights)) / tot for k in states[0]}


def run_federated(global_model, clients, val, rounds, local_epochs, lr, bs, bits_per_param,
                  bits_per_rb, seed=0, on_round=None):
    """Returns per-round log: ADE, FDE, cumulative bits and resource blocks (up+down, all clients)."""
    log, cum_bits = [], 0
    for r in range(rounds):
        g_state = trainable_state(global_model)
        states, ws = [], []
        for cid, cd in enumerate(clients):
            local = copy.deepcopy(global_model)
            load_state(local, g_state)                                   # downlink
            ws.append(local_train(local, cd, local_epochs, lr, bs, seed + 100 * r + cid))
            states.append(trainable_state(local))                        # uplink
        load_state(global_model, fedavg(states, ws))
        bits_round = 2 * len(clients) * n_params(g_state) * bits_per_param   # down + up
        cum_bits += bits_round
        ade, fde = evaluate(global_model, *val)
        log.append(dict(round=r + 1, ade=ade, fde=fde, bits=cum_bits,
                        resource_blocks=cum_bits / bits_per_rb, payload_params=n_params(g_state)))
        if on_round: on_round(r, global_model, log[-1])
    return log
