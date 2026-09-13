import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.gam import GAMConfig
from src.models.sam import SAMConfig
from src.models.vam import VAMConfig
from src.models.trajfusionnet_plus import TrajFusionNetPlus, TrajFusionNetPlusConfig
from src.utils.synthetic_scene import generate_scene, scene_batch_to_tensors


def _tiny_config() -> TrajFusionNetPlusConfig:
    return TrajFusionNetPlusConfig(
        sam=SAMConfig(past_len=6, future_len=8, d_model=32, traj_enc_layers=1, traj_dec_layers=1,
                      traj_heads=2, traj_ffn_dim=64, cls_layers=1, cls_heads=2, cls_ffn_dim=64, proj_dim=16),
        vam=VAMConfig(base_channels=8, num_stages=2, proj_dim=16),
        gam=GAMConfig(max_nodes=8, d_model=32, num_heads=2, num_layers=1, proj_dim=16),
        fusion_hidden_dim=24,
    )


def test_end_to_end_forward_backward():
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = TrajFusionNetPlus(cfg)

    rng = np.random.default_rng(0)
    scenes = [generate_scene(rng, past_len=cfg.sam.past_len, future_len=cfg.sam.future_len,
                              max_nodes=cfg.gam.max_nodes) for _ in range(4)]
    batch = scene_batch_to_tensors(scenes, frame_size=32)

    out = model(
        past_traj=batch["past_traj"],
        observed_frame=batch["observed_frame"],
        predicted_frame=batch["predicted_frame"],
        node_positions=batch["node_positions"],
        node_classes=batch["node_classes"],
        node_areas=batch["node_areas"],
        node_valid_mask=batch["node_valid"],
    )

    assert out["crossing_logits"].shape == (4, 2)
    assert out["predicted_trajectory"].shape == (4, cfg.sam.future_len, 5)
    assert torch.isfinite(out["crossing_logits"]).all()

    loss_cls = torch.nn.functional.cross_entropy(out["crossing_logits"], batch["will_cross"])
    loss_traj = torch.nn.functional.mse_loss(out["predicted_trajectory"], batch["future_traj"])
    total_loss = loss_cls + loss_traj
    total_loss.backward()

    n_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_params_total = sum(1 for _ in model.parameters())
    assert n_params_with_grad == n_params_total, "every parameter should receive a gradient"


def test_parameter_count_reported():
    cfg = _tiny_config()
    model = TrajFusionNetPlus(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
