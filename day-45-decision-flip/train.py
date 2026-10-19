"""Train the dense Gaussian-occupancy bank on synthetic scenes."""
import argparse, json, time, yaml, torch, torch.nn.functional as F
from decision_flip.model import GaussianOccModel
from decision_flip.data import make_split
from decision_flip.metrics import evaluate


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--steps", type=int, default=None); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); t = cfg["train"]; torch.manual_seed(t["seed"])
    grid = (cfg["grid"]["x"], cfg["grid"]["y"], cfg["grid"]["z"])
    tf, to = make_split(0, cfg["data"]["train_scenes"], grid)
    vf, vo = make_split(10000, cfg["data"]["val_scenes"], grid)
    model = GaussianOccModel(cfg); opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    steps = a.steps or t["steps"]; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    w = torch.tensor([0.5, 1.0, 3.0, 1.5, 2.0])  # class weights: empty dominates voxels
    t0 = time.time(); hist = []
    for s in range(steps):
        model.train(); idx = torch.randint(0, len(tf), (t["batch_size"],))
        logits = model(tf[idx]).reshape(-1, cfg["num_classes"])
        loss = F.cross_entropy(logits, to[idx].reshape(-1), weight=w)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); hist.append(loss.item())
        if s % 100 == 0 or s == steps - 1:
            print(f"step {s:4d} loss {loss.item():.4f}  val mIoU {evaluate(model, vf, vo):.4f}", flush=True)
    miou = evaluate(model, vf, vo)
    torch.save({"cfg": cfg, "state": model.state_dict()}, "checkpoints/dense.pt")
    json.dump({"loss_first": hist[0], "loss_last": hist[-1], "val_miou": miou, "params": sum(p.numel() for p in model.parameters()),
               "seconds": time.time() - t0}, open("checkpoints/train_stats.json", "w"), indent=1)
    print("done", miou)

if __name__ == "__main__":
    main()
