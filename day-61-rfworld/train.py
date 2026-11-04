"""python train.py [--quick]  -> checkpoints/{rfworld,baseline}.pt + results.json"""
import argparse, json, time, numpy as np, torch
from rfworld.world import make_batch, ANCHORS
from rfworld.model import RFWorldToy, GlobalAnchorClassifier, loss_fn
from rfworld.evaluate import to_t, select_quality, closed_loop

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--no_aux", action="store_true")
a = ap.parse_args(); torch.set_num_threads(2); torch.manual_seed(0)
nscenes, epochs = (800, 4) if a.quick else (3000, 14)
t0 = time.time(); rng = np.random.default_rng(0)
data = make_batch(rng, nscenes); print(f"data {nscenes} scenes in {time.time()-t0:.0f}s")
A = torch.as_tensor(ANCHORS)[None]


def fit(model, aux, name):
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * (nscenes // 32))
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(nscenes); tot = 0
        for i in range(nscenes // 32):
            idx = perm[i * 32:(i + 1) * 32]; B = to_t({k: v[idx] for k, v in data.items()})
            out = model(B, A.expand(32, -1, -1, -1), with_aux=aux)
            l = loss_fn(out, B, B["fut"] if aux else None); opt.zero_grad(); l.backward(); opt.step(); sched.step(); tot += l.item()
        print(f"[{name}] epoch {ep+1}/{epochs} loss {tot/(nscenes//32):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return model.eval()


res = {}
rf = fit(RFWorldToy(), not a.no_aux, "rfworld"); torch.save(rf.state_dict(), "checkpoints/rfworld.pt")
bl = fit(GlobalAnchorClassifier(), False, "baseline"); torch.save(bl.state_dict(), "checkpoints/baseline.pt")
for n, m in (("rfworld_sparse_readout", rf), ("global_anchor_mlp_baseline", bl)):
    res[n] = dict(**select_quality(m), **closed_loop(m)); print(n, res[n])
json.dump(res, open("checkpoints/results.json", "w"), indent=1)
