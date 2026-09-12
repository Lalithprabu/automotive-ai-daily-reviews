"""
Shape + backward-pass smoke tests for SV-WAM.
Run with: pytest tests/test_sv_wam.py -v
"""
import torch

from src.losses.drivable_area_regularizer import DrivableAreaRegularizer
from src.losses.wam_losses import sv_wam_loss
from src.models.sv_wam import SVWAM


def _build_small_model():
    torch.manual_seed(0)
    return SVWAM(n_cams=6, embed_dim=64, n_layers=2, n_heads=4,
                 n_action_steps=6, n_video_tokens=16)


def test_train_forward_shapes():
    model = _build_small_model()
    images = torch.randn(2, 6, 3, 224, 400)
    traj, future_latents = model(images, predict_action_only=False)
    assert traj.shape == (2, 6, 3)
    assert future_latents.shape == (2, 16, 64)


def test_deploy_forward_skips_video_branch():
    model = _build_small_model()
    images = torch.randn(2, 6, 3, 224, 400)
    traj, video_out = model(images, predict_action_only=True)
    assert traj.shape == (2, 6, 3)
    assert video_out is None


def test_loss_and_backward():
    model = _build_small_model()
    images = torch.randn(2, 6, 3, 224, 400)
    traj, future_latents = model(images, predict_action_only=False)

    gt_traj = torch.randn(2, 6, 3)
    gt_latents = torch.randn_like(future_latents)
    sdf = torch.rand(2, 1, 200, 200) * 2 - 1
    origin = torch.zeros(2, 2)
    reg = DrivableAreaRegularizer()

    loss = sv_wam_loss(traj, gt_traj, future_latents, gt_latents,
                        sdf, map_resolution=0.5, map_origin=origin,
                        drivable_regularizer=reg)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())
