import argparse, os, yaml, torch
from torch.utils.data import DataLoader
from models.flow_model import FlowNet, flow_matching_loss
from src.data import build_dataset

def main(cfg_path="config.yaml"):
    cfg = yaml.safe_load(open(cfg_path)); torch.manual_seed(cfg["seed"])
    tr, _ = build_dataset(cfg["n_train"], cfg["hazard_frac"], cfg["seed"])
    va, _ = build_dataset(cfg["n_val"], cfg["hazard_frac"], cfg["seed"] + 1)
    net = FlowNet(cfg["horizon"], 7, cfg["hidden"], cfg["depth"])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    dl = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True)
    print(f"params: {sum(p.numel() for p in net.parameters()):,}")
    for ep in range(cfg["epochs"]):
        net.train(); tot = 0
        for x1, c in dl:
            loss = flow_matching_loss(net, x1, c); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(x1)
        sched.step()
        if ep % 5 == 0 or ep == cfg["epochs"] - 1:
            net.eval()
            with torch.no_grad(): vl = flow_matching_loss(net, *va.tensors)
            print(f"epoch {ep:3d} train {tot/len(tr):.5f} val {vl.item():.5f}")
    os.makedirs(os.path.dirname(cfg["checkpoint"]), exist_ok=True)
    torch.save(net.state_dict(), cfg["checkpoint"]); print("saved", cfg["checkpoint"])
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); main(ap.parse_args().config)
