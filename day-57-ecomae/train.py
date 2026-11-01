"""Pretrain (unlabelled) -> fine-tune (few labels) -> evaluate ID vs leave-one-environment-out OOD (fog).
python train.py [--quick]     methods: sup (no pretraining) | timae (reconstruction only) | ecomae (NIG + observability alignment)"""
import argparse, json, time, copy
import numpy as np, torch, torch.nn.functional as Fn
from ecomae.synthetic import make_dataset
from ecomae.model import eCoMAE, pretrain_loss
from ecomae.metrics import report

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--n_unl", type=int, default=6000); ap.add_argument("--n_lab", type=int, default=300)
ap.add_argument("--pre_epochs", type=int, default=25); ap.add_argument("--ft_epochs", type=int, default=60)
a = ap.parse_args()
if a.quick: a.seeds, a.n_unl, a.pre_epochs, a.ft_epochs = 1, 1500, 6, 20
torch.set_num_threads(2)


def batches(n, bs, g):
    p = torch.randperm(n, generator=g)
    for i in range(0, n, bs): yield p[i:i + bs]


def pretrain(model, D, use_evi, seed):
    g = torch.Generator().manual_seed(seed); opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-2)
    steps = a.pre_epochs * (len(D["X"]) // 128 + 1); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    for ep in range(a.pre_epochs):
        for ix in batches(len(D["X"]), 128, g):
            loss, t = pretrain_loss(model, D["X"][ix], D["valid"][ix], use_evi=use_evi)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        if ep % 5 == 0 or ep == a.pre_epochs - 1: print(f"    pre ep{ep:02d} " + " ".join(f"{k}={v:.3f}" for k, v in t.items()), flush=True)


def finetune(model, D, seed):
    g = torch.Generator().manual_seed(seed); opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-2)
    cw = 1.0 / torch.bincount(D["y"], minlength=4).float().clamp(min=1); cw = cw / cw.mean()
    for ep in range(a.ft_epochs):
        for ix in batches(len(D["y"]), 64, g):
            loss = Fn.cross_entropy(model.classify(D["X"][ix], D["valid"][ix]), D["y"][ix], weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def evaluate(model, D):
    model.eval(); p = torch.softmax(model.classify(D["X"], D["valid"]), -1).numpy(); return report(p, D["y"].numpy())


res = {m: {"ID": [], "OOD": []} for m in ("sup", "timae", "ecomae")}
t0 = time.time()
for s in range(a.seeds):
    train_envs = ["clear", "rain"]
    unl = make_dataset(a.n_unl, train_envs, 1000 + s); lab = make_dataset(a.n_lab, train_envs, 2000 + s)
    idt = make_dataset(1500, train_envs, 3000 + s); ood = make_dataset(1500, ["fog"], 4000 + s)
    for m in res:
        torch.manual_seed(s); model = eCoMAE()
        if m != "sup": print(f"seed {s} {m}: pretraining", flush=True); pretrain(model, unl, m == "ecomae", s)
        finetune(model, lab, s)
        r_id, r_ood = evaluate(model, idt), evaluate(model, ood); res[m]["ID"].append(r_id); res[m]["OOD"].append(r_ood)
        print(f"seed {s} {m:6s} ID {r_id}  OOD {r_ood}  [{time.time()-t0:.0f}s]", flush=True)
        if s == 0: torch.save(model.state_dict(), f"checkpoints/{m}.pt")
agg = {m: {sp: {k: (float(np.mean([r[k] for r in res[m][sp]])), float(np.std([r[k] for r in res[m][sp]]))) for k in ("acc", "ece", "aurc")} for sp in ("ID", "OOD")} for m in res}
json.dump(dict(args=vars(a), per_seed=res, mean_std=agg), open("checkpoints/results.json", "w"), indent=1)
print("\n| method | ID acc | ID ECE | ID AURC | OOD acc | OOD ECE | OOD AURC |\n|---|---|---|---|---|---|---|")
for m in agg:
    f = lambda sp, k: f"{agg[m][sp][k][0]:.3f}±{agg[m][sp][k][1]:.3f}"
    print(f"| {m} | {f('ID','acc')} | {f('ID','ece')} | {f('ID','aurc')} | {f('OOD','acc')} | {f('OOD','ece')} | {f('OOD','aurc')} |")
