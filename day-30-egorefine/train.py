"""Train the 4 ablation models (ego_only / naive / traf / egorefine) with identical budget, then evaluate."""
import argparse, json, time, yaml, torch
from egorefine import AsyncFusionNet, MODES, make_batch
from egorefine.metrics import focal_loss, extract_peaks, average_precision


@torch.no_grad()
def evaluate(model, n=400, seed=123, bs=100):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    peaks, pos, val, vis, dl = [], [], [], [], []
    for _ in range(n // bs):
        b = make_batch(bs, gen=gen)
        p = torch.sigmoid(model(b["ego"], b["col"]))
        peaks += extract_peaks(p); pos.append(b["pos"]); val.append(b["valid"]); vis.append(b["vis"]); dl.append(b["delta"])
    pos, val, vis, dl = map(torch.cat, (pos, val, vis, dl))
    r = {}
    for tau in (1.0, 2.0):
        r[f"AP@{tau}"] = average_precision(peaks, pos, val, tau)
        r[f"AP@{tau}_blind"] = average_precision(peaks, pos, val, tau, subset=~vis)
    r["AP@1.0_by_delay"] = {int(d): average_precision([peaks[i] for i in (dl == d).nonzero().squeeze(1).tolist()],
                            pos[dl == d], val[dl == d], 1.0) for d in range(7)}
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--steps", type=int); ap.add_argument("--modes", nargs="*", default=list(MODES))
    ap.add_argument("--out", default="checkpoints"); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); T = cfg["train"]; steps = a.steps or T["steps"]
    import os; os.makedirs(a.out, exist_ok=True); results = {}
    for mode in a.modes:
        torch.manual_seed(T["seed"])
        m = AsyncFusionNet(mode, **cfg["model"])
        opt = torch.optim.AdamW(m.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, T["lr"], total_steps=steps)
        gen = torch.Generator().manual_seed(T["seed"] + 1); t0 = time.time()
        print(f"== {mode}: {sum(p.numel() for p in m.parameters()):,} params")
        for s in range(steps):
            m.train(); b = make_batch(T["batch_size"], gen=gen)
            loss = focal_loss(m(b["ego"], b["col"]), b["hm"])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); sched.step()
            if s % 100 == 0 or s == steps - 1: print(f"  step {s:4d} loss {loss.item():.3f}  ({time.time()-t0:.0f}s)", flush=True)
        torch.save({"state": m.state_dict(), "cfg": cfg["model"], "mode": mode}, f"{a.out}/{mode}.pt")
        results[mode] = evaluate(m, **cfg["eval"] if False else dict(n=cfg["eval"]["n_scenes"], seed=cfg["eval"]["seed"]))
        print("  ", json.dumps({k: v for k, v in results[mode].items() if k != "AP@1.0_by_delay"}), flush=True)
    json.dump(results, open(f"{a.out}/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
