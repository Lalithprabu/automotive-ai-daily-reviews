"""Train the waypoint policy with plain per-waypoint L1 (no ECO in the loop: ECO is training-free)."""
import argparse, json, time, yaml, torch
from eco.policy import WaypointPolicy
from eco.data import make_dataset


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--out", default="policy.pt")
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config)); torch.manual_seed(cfg["seed"])
    Xtr, Ytr = make_dataset(cfg["train_samples"], cfg, cfg["seed"] + 1)
    Xva, Yva = make_dataset(cfg["val_samples"], cfg, cfg["seed"] + 2)
    pol = WaypointPolicy(cfg["route_points"], cfg["history"], cfg["horizon"], cfg["hidden"])
    opt = torch.optim.AdamW(pol.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n, bs, t0, g = len(Xtr), cfg["batch_size"], time.time(), torch.Generator().manual_seed(0)
    print(f"params={sum(p.numel() for p in pol.parameters()):,}")
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(n, generator=g); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            y = Ytr[idx] + cfg["label_noise"] * torch.randn_like(Ytr[idx])       # noisy demonstrations
            loss = (pol(Xtr[idx]) - y).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        if ep % 10 == 0 or ep == cfg["epochs"] - 1:
            with torch.no_grad(): v = (pol(Xva) - Yva).norm(dim=-1).mean().item()
            print(f"ep {ep:3d} train_l1={tot/n:.4f} val_ade={v:.3f}m")
    torch.save(pol.state_dict(), a.out); print(f"saved {a.out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
