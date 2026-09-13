"""Training-loop entrypoint for TrajFusionNet+ on the synthetic pedestrian-
crossing scene generator (src/utils/synthetic_scene.py). Not a substitute
for training on real PIE/JAAD footage (out of scope for this repo package)
but exercises every branch (SAM/VAM/GAM), the fusion trunk, both loss terms
(crossing classification + trajectory regression), and a full backward pass
end-to-end -- useful for smoke-testing the architecture and for CI.

Usage:
    python train.py --config config.yaml --steps 200
    python train.py --config tiny_trajfusionnetplus_cfg.yaml --steps 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.gam import GAMConfig
from src.models.sam import SAMConfig
from src.models.vam import VAMConfig
from src.models.trajfusionnet_plus import TrajFusionNetPlus, TrajFusionNetPlusConfig
from src.utils.synthetic_scene import generate_scene, scene_batch_to_tensors


def build_config(raw: dict) -> TrajFusionNetPlusConfig:
    return TrajFusionNetPlusConfig(
        sam=SAMConfig(**raw["sam"]),
        vam=VAMConfig(**raw["vam"]),
        gam=GAMConfig(**raw["gam"]),
        fusion_hidden_dim=raw["fusion_hidden_dim"],
        dropout=raw["dropout"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--steps", type=int, default=None, help="override train.steps from config")
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    train_cfg = raw["train"]
    steps = args.steps or train_cfg["steps"]
    torch.manual_seed(train_cfg["seed"])
    rng = np.random.default_rng(train_cfg["seed"])

    model_cfg = build_config(raw)
    model = TrajFusionNetPlus(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"TrajFusionNet+ initialized: {n_params:,} parameters "
          f"(SAM {sum(p.numel() for p in model.sam.parameters()):,} / "
          f"VAM {sum(p.numel() for p in model.vam.parameters()):,} / "
          f"GAM {sum(p.numel() for p in model.gam.parameters()):,})")

    frame_size = 32 if model_cfg.sam.d_model <= 32 else 128

    for step in range(1, steps + 1):
        scenes = [
            generate_scene(rng, past_len=model_cfg.sam.past_len,
                            future_len=model_cfg.sam.future_len,
                            max_nodes=model_cfg.gam.max_nodes)
            for _ in range(train_cfg["batch_size"])
        ]
        batch = scene_batch_to_tensors(scenes, frame_size=frame_size)

        out = model(
            past_traj=batch["past_traj"],
            observed_frame=batch["observed_frame"],
            predicted_frame=batch["predicted_frame"],
            node_positions=batch["node_positions"],
            node_classes=batch["node_classes"],
            node_areas=batch["node_areas"],
            node_valid_mask=batch["node_valid"],
        )

        loss_cls = F.cross_entropy(out["crossing_logits"], batch["will_cross"])
        loss_traj = F.mse_loss(out["predicted_trajectory"], batch["future_traj"])
        loss = train_cfg["cls_loss_weight"] * loss_cls + train_cfg["traj_loss_weight"] * loss_traj

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            with torch.no_grad():
                pred_label = out["crossing_logits"].argmax(dim=-1)
                acc = (pred_label == batch["will_cross"]).float().mean().item()
            print(f"step {step:4d}/{steps} | loss {loss.item():.4f} "
                  f"(cls {loss_cls.item():.4f}, traj {loss_traj.item():.4f}) | "
                  f"batch crossing-acc {acc:.2f}")

    print("Training smoke-test complete.")


if __name__ == "__main__":
    main()
