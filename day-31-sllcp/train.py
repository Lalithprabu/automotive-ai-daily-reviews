"""Train the stand-in predictor, calibrate SLLCP, evaluate on held-out shifted test scenes."""
import argparse, json, yaml, torch
import torch.nn.functional as F
from sllcp.data import make_dataset
from sllcp.models import SafetyScorer
from evaluate import evaluate, format_table


def main(cfg_path="config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    torch.manual_seed(cfg["seed"])
    tr = make_dataset(cfg["n_train"], cfg["train_mix"], cfg["seed"] + 1)
    cal = make_dataset(cfg["n_cal"], cfg["cal_mix"], cfg["seed"] + 2)
    te = make_dataset(cfg["n_test"], cfg["test_mix"], cfg["seed"] + 3)
    model = SafetyScorer()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    n = tr["x"].shape[0]
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            loss = F.cross_entropy(model(tr["x"][idx]), tr["y"][idx])  # plain CE: overconfident toward 'safe'
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        if ep % 5 == 0 or ep == cfg["epochs"] - 1:
            print(f"epoch {ep:2d}  train CE {tot/n:.4f}")
    torch.save(model.state_dict(), "checkpoints/scorer.pt")
    res, *_ = evaluate(model, cal, te, cfg["alpha"], cfg["bandwidth"])
    print(format_table(res, cfg["alpha"]))
    print(f"mean SLLCP prediction-set size: {res['_set_size_sllcp']:.3f}")
    json.dump(res, open("checkpoints/results.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
