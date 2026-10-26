"""Train SafeDriveVLA-Mini (and the no-dream ablation) on synthetic scenes; writes ckpt + results.json."""
import json, sys, yaml, argparse, numpy as np, torch
from src.synthetic import sample_scenes, build_batch, FOLLOW, executed_collision
from src.safedrivevla import SafeDriveVLAMini, compute_loss
from src import metrics as M


def to_t(b): return {k: torch.as_tensor(v) for k, v in b.items()}


def run(cfg, use_dream, tr, va, va_sc, tag):
    torch.manual_seed(cfg["seed"])
    model = SafeDriveVLAMini(cfg["dim"], cfg["heads"], cfg["enc_layers"], cfg["horizon"], use_dream)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    n = len(tr["nav"]); steps = cfg["epochs"] * ((n + cfg["batch_size"] - 1) // cfg["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=steps)
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(n); model.train()
        for i in range(0, n, cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            b = {k: v[idx] for k, v in tr.items()}
            out = model(b["agents"], b["ego"], b["nav"], forced_mode=b["mode"])      # teacher-forced mode for action head
            loss, _ = compute_loss(out, b, cfg["loss_w"])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if (ep + 1) % 10 == 0:
            print(f"[{tag}] epoch {ep+1} loss {loss.item():.3f}", flush=True)
    model.eval()
    with torch.no_grad():
        out = model(va["agents"], va["ego"], va["nav"])
    pred, true = out["mode"].numpy(), va["mode"].numpy()
    unsafe = true != FOLLOW
    naive = np.tile(np.array([0, .25, .6, 1, 1, 1, 1, 1], np.float32), (len(true), 1))   # always obey literally
    res = dict(mode_acc=M.mode_accuracy(pred, true), unsafe_recall=M.unsafe_recall(pred, true),
               false_refusal=M.false_refusal(pred, true),
               collision_rate_unsafe_instr_model=M.collision_rate(va_sc, out["profile"].numpy(), unsafe),
               collision_rate_unsafe_instr_naive=M.collision_rate(va_sc, naive, unsafe),
               params=sum(p.numel() for p in model.parameters()))
    if use_dream:
        res["dream_collision_acc"] = float(((out["risk"] > 0).float() == va["coll"]).float().mean())
    return model, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--epochs", type=int)
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config))
    if a.epochs: cfg["epochs"] = a.epochs
    rng = np.random.default_rng(cfg["seed"])
    tr_sc = sample_scenes(cfg["train_scenes"], rng, cfg["max_agents"]); va_sc = sample_scenes(cfg["val_scenes"], rng, cfg["max_agents"])
    tr, va = to_t(build_batch(tr_sc)), to_t(build_batch(va_sc))
    print("mode distribution (val):", np.bincount(va["mode"].numpy(), minlength=3) / len(va["mode"]))
    results = {}
    for tag, dream in (("dream", True), ("no_dream", False)):
        model, results[tag] = run(cfg, dream, tr, va, va_sc, tag)
        if dream: torch.save(model.state_dict(), "assets/safedrivevla_mini.pt")
        print(tag, json.dumps(results[tag], indent=1))
    json.dump(results, open("assets/results.json", "w"), indent=1)
