"""
Minimal evaluation script: runs SV-WAM in deployment mode
(predict_action_only=True) and reports Average/Final Displacement Error
(ADE/FDE) against ground-truth waypoints.

This is a starting point, not a NAVSIMv2/nuScenes-compliant closed-loop
harness -- plug this dataset/model into the official evaluation devkits
for paper-comparable PDMS/collision-rate numbers.
"""
import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.surround_view_dataset import SurroundViewDataset
from src.models.sv_wam import SVWAM


@torch.no_grad()
def evaluate(model, loader, device):
    total_ade, total_fde, n = 0.0, 0.0, 0
    for batch in loader:
        images = batch["images"].to(device)
        gt_trajectory = batch["trajectory"].to(device)      # [B, A, 3]

        pred_trajectory, _ = model(images, predict_action_only=True)  # [B, A, 3]

        displacement = torch.linalg.norm(
            pred_trajectory[..., :2] - gt_trajectory[..., :2], dim=-1
        )  # [B, A], Euclidean distance per waypoint in meters

        total_ade += displacement.mean(dim=-1).sum().item()   # avg over waypoints per sample
        total_fde += displacement[:, -1].sum().item()          # final-waypoint error per sample
        n += images.shape[0]

    return {"ADE_m": total_ade / n, "FDE_m": total_fde / n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sv_wam_base.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SVWAM(**cfg["model"], img_size=tuple(cfg["data"]["img_size"])).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dataset = SurroundViewDataset(
        n_cams=cfg["model"]["n_cams"],
        img_size=tuple(cfg["data"]["img_size"]),
        n_action_steps=cfg["model"]["n_action_steps"],
        synthetic=cfg["data"]["synthetic"],
        synthetic_length=32,
    )
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"])

    metrics = evaluate(model, loader, device)
    print(metrics)


if __name__ == "__main__":
    main()
