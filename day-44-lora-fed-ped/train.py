"""Pretrain a base forecaster, then compare federated adaptation: (A) full-model FedAvg, (B) LoRA FedAvg
with more local epochs / fewer rounds. Saves checkpoints + results.json. Usage: python train.py --config config.yaml"""
import argparse, copy, json, time, yaml, torch, torch.nn.functional as F
from src.model import TrajTransformer
from src.lora import inject_lora, freeze_all_but_lora, trainable_state, n_params
from src.data import make_source, make_client, make_target_val
from src.federated import run_federated, evaluate, local_train


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--out", default="runs")
    a = ap.parse_args(); c = yaml.safe_load(open(a.config)); torch.manual_seed(c["seed"])
    import os; os.makedirs(a.out, exist_ok=True); t0 = time.time()
    O, P = c["obs_len"], c["pred_len"]
    mk = lambda: TrajTransformer(O, P, c["d_model"], c["n_heads"], c["n_layers"])

    # 1) Centralised pretraining on the generic source domain ("foundation" stand-in)
    base = mk(); so, sf, _ = make_source(c["pretrain_samples"], O, P, c["seed"])
    local_train(base, (so, sf), c["pretrain_epochs"], c["lr_pretrain"], c["batch_size"], c["seed"])
    torch.save(base.state_dict(), f"{a.out}/base.pt")

    clients = [make_client(i, c["n_clients"], c["client_samples"], O, P, c["seed"])[:2] for i in range(c["n_clients"])]
    vo, vf, _ = make_target_val(c["val_samples"], O, P, c["n_clients"], c["seed"]); val = (vo, vf)
    res = {"zero_shot": dict(zip(["ade", "fde"], evaluate(base, *val)))}
    zs_const = (vo[:, -1:] - vo[:, -2:-1]).expand(-1, P, -1).cumsum(1)          # constant-velocity baseline
    res["const_vel"] = dict(zip(["ade", "fde"], [(zs_const - vf).norm(dim=-1).mean().item(), (zs_const - vf).norm(dim=-1)[:, -1].mean().item()]))

    # 2a) Full-model FedAvg
    full = copy.deepcopy(base)
    for p in full.parameters(): p.requires_grad = True
    f = c["full_fl"]
    res["full_fl"] = run_federated(full, clients, val, f["rounds"], f["local_epochs"], c["lr_fl_full"], c["batch_size"],
                                   c["bits_per_param"], c["bits_per_resource_block"], c["seed"])
    res["full_payload_params"] = n_params(trainable_state(full))

    # 2b) LoRA FedAvg (adapters + tiny output head trainable)
    lora = copy.deepcopy(base); inject_lora(lora, set(c["lora_targets"]), c["lora_rank"], c["lora_alpha"])
    freeze_all_but_lora(lora); l = c["lora_fl"]
    res["lora_fl"] = run_federated(lora, clients, val, l["rounds"], l["local_epochs"], c["lr_fl_lora"], c["batch_size"],
                                   c["bits_per_param"], c["bits_per_resource_block"], c["seed"])
    res["lora_payload_params"] = n_params(trainable_state(lora))
    torch.save({"base": base.state_dict(), "lora": lora.state_dict(), "cfg": c}, f"{a.out}/lora_fl.pt")
    torch.save(full.state_dict(), f"{a.out}/full_fl.pt")
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)

    zf, ff, lf = res["zero_shot"], res["full_fl"][-1], res["lora_fl"][-1]
    print(f"zero-shot ADE {zf['ade']:.3f}  const-vel ADE {res['const_vel']['ade']:.3f}")
    print(f"full FedAvg : ADE {ff['ade']:.3f} FDE {ff['fde']:.3f}  payload {res['full_payload_params']} params  RB {ff['resource_blocks']:.0f}")
    print(f"LoRA FedAvg : ADE {lf['ade']:.3f} FDE {lf['fde']:.3f}  payload {res['lora_payload_params']} params  RB {lf['resource_blocks']:.0f}")
    print(f"comm reduction {100*(1-lf['bits']/ff['bits']):.1f}%  | {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
