"""Train TransE lane-change KG + MLP baseline on synthetic highway data; calibrate tau; save checkpoint."""
import argparse, json, time, sys, pathlib
import numpy as np, torch, torch.nn as nn, yaml
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from kglc.data import make_split
from kglc.model import TransE, MLPBaseline, macro_f1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); torch.manual_seed(cfg["seed"]); rng = np.random.default_rng(cfg["seed"])
    d = cfg["data"]
    tr = make_split(d["n_train"], rng, d["train_horizons"]); va = make_split(d["n_val"], rng, d["train_horizons"])
    ents_tr, y_tr = torch.from_numpy(tr.ents), torch.from_numpy(tr.label)
    m = cfg["model"]; kg = TransE(m["dim"], m["gamma"], m["adv_temp"])
    opt = torch.optim.Adam(kg.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    t0 = time.time(); bs = cfg["train"]["batch_size"]; hist = []
    for ep in range(cfg["train"]["epochs"]):
        perm = torch.randperm(len(y_tr)); tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]; opt.zero_grad(); loss = kg.loss(ents_tr[idx], y_tr[idx]); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        hist.append(tot / len(perm))
        if ep % 10 == 0 or ep == cfg["train"]["epochs"] - 1: print(f"[kg] epoch {ep:3d} loss {hist[-1]:.4f}")
    # uniform prior (balanced classes) -- the precursor paper also picks the argmax posterior
    tau, nll = kg.calibrate_tau(torch.from_numpy(va.ents), torch.from_numpy(va.label))
    print(f"[kg] calibrated tau={tau}  val NLL={nll:.4f}")
    # ---- MLP baseline
    mlp = MLPBaseline(cfg["mlp_baseline"]["hidden"])
    X = torch.from_numpy(tr.feats); Xl = X.clone(); Xl[:, 2:] = torch.log(Xl[:, 2:].clamp(min=0.5))
    mlp.mu.copy_(Xl.mean(0)); mlp.sd.copy_(Xl.std(0) + 1e-6)
    o2 = torch.optim.Adam(mlp.parameters(), lr=0.003); ce = nn.CrossEntropyLoss()
    for ep in range(cfg["mlp_baseline"]["epochs"]):
        perm = torch.randperm(len(y_tr))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]; o2.zero_grad(); ce(mlp(X[idx]), y_tr[idx]).backward(); o2.step()
    with torch.no_grad():
        va_kg = macro_f1(kg.log_posterior(torch.from_numpy(va.ents)).argmax(-1), torch.from_numpy(va.label))
        va_mlp = macro_f1(mlp(torch.from_numpy(va.feats)).argmax(-1), torch.from_numpy(va.label))
    print(f"[val] macro-F1  KG-Bayes {va_kg:.4f}   MLP {va_mlp:.4f}   ({time.time()-t0:.1f}s, {sum(p.numel() for p in kg.parameters())} KG params)")
    pathlib.Path("checkpoints").mkdir(exist_ok=True)
    torch.save({"kg": kg.state_dict(), "mlp": mlp.state_dict(), "cfg": cfg}, "checkpoints/model.pt")
    json.dump({"loss_history": hist, "tau": tau, "val_nll": nll, "val_f1_kg": va_kg, "val_f1_mlp": va_mlp},
              open("checkpoints/train_log.json", "w"))


if __name__ == "__main__":
    main()
