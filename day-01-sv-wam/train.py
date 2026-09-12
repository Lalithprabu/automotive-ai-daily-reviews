"""
Minimal training loop for SV-WAM.

Runs out of the box against the synthetic dataset (config data.synthetic:
true) so you can verify the pipeline end-to-end before wiring up real
NAVSIMv2/nuScenes clips. Swap `synthetic: false` and implement
SurroundViewDataset._load_sample once your manifest is ready.
"""
import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.surround_view_dataset import SurroundViewDataset
from src.losses.drivable_area_regularizer import DrivableAreaRegularizer
from src.losses.wam_losses import sv_wam_loss
from src.models.sv_wam import SVWAM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/sv_wam_base.yaml")
    parser.add_argument("--steps", type=int, default=None,
                         help="Override: run only this many optimizer steps (smoke-testing).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SVWAM(**cfg["model"], img_size=tuple(cfg["data"]["img_size"])).to(device)
    regularizer = DrivableAreaRegularizer(
        vehicle_length=cfg["loss"]["vehicle_length"],
        vehicle_width=cfg["loss"]["vehicle_width"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    dataset = SurroundViewDataset(
        n_cams=cfg["model"]["n_cams"],
        img_size=tuple(cfg["data"]["img_size"]),
        n_action_steps=cfg["model"]["n_action_steps"],
        n_video_tokens=cfg["model"]["n_video_tokens"],
        embed_dim=cfg["model"]["embed_dim"],
        synthetic=cfg["data"]["synthetic"],
    )
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True)

    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        for batch in loader:
            images = batch["images"].to(device)                       # [B, N, C, H, W]
            gt_trajectory = batch["trajectory"].to(device)             # [B, A, 3]
            gt_future_latents = batch["future_latents_target"].to(device)  # [B, V, D]

            pred_trajectory, pred_future_latents = model(images, predict_action_only=False)

            b = images.shape[0]
            # Synthetic/flat drivable-area mask for the smoke-test path: an
            # all-negative SDF (everywhere "inside") so the regularizer term
            # is well-defined even without a real BEV map wired up yet.
            sdf_map = -torch.ones(b, 1, 200, 200, device=device)
            map_origin = torch.zeros(b, 2, device=device)

            loss = sv_wam_loss(
                pred_trajectory, gt_trajectory,
                pred_future_latents, gt_future_latents,
                sdf_map, map_resolution=0.5, map_origin=map_origin,
                drivable_regularizer=regularizer,
                lambda_video=cfg["loss"]["lambda_video"],
                lambda_drivable=cfg["loss"]["lambda_drivable"],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 10 == 0 or (args.steps and step >= args.steps):
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            if args.steps and step >= args.steps:
                print("reached --steps limit, stopping (smoke test mode)")
                return


if __name__ == "__main__":
    main()
