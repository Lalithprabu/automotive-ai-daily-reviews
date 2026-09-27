import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest

from src.geometry import make_camera_rig, project_points, normalize_pixels_for_grid_sample
from src.dataset import SyntheticDrivingSceneDataset, collate_scenes, FUT_LEN, HIST_LEN, N_CAM, IMG_SIZE
from src.model import S2Planner
from src.losses import coarse_to_fine_loss, ade_fde


def test_geometry_center_camera_projects_forward_point_to_image_center():
    rig = make_camera_rig(yaws_deg=(-32.0, 0.0, 32.0), focal_px=260.0, image_size=64, cam_height_m=1.4)
    # A point straight ahead at ground level, offset in z by cam height so it's at cam-height above ground level -> maps to (S/2, S/2) in the *center* camera only if z equals cam height. Use a point at camera height directly ahead.
    pt = torch.tensor([10.0, 0.0, 1.4])
    uv, valid = project_points(rig, pt)
    center_cam = 1
    assert valid[center_cam]
    assert abs(uv[center_cam, 0].item() - 32.0) < 1e-3
    assert abs(uv[center_cam, 1].item() - 32.0) < 1e-3


def test_geometry_lateral_point_moves_toward_correct_side_in_center_camera():
    rig = make_camera_rig(image_size=64)
    left_pt = torch.tensor([10.0, 2.0, 1.4])  # to the ego's left
    right_pt = torch.tensor([10.0, -2.0, 1.4])
    uv, valid = project_points(rig, torch.stack([left_pt, right_pt]))
    center_cam = 1
    u_left = uv[center_cam, 0, 0].item()
    u_right = uv[center_cam, 1, 0].item()
    # A point to the ego's left should land at a SMALLER image-u (more to the left of the frame).
    assert u_left < u_right


def test_geometry_point_behind_camera_is_invalid():
    rig = make_camera_rig(image_size=64)
    behind = torch.tensor([-5.0, 0.0, 1.4])
    _, valid = project_points(rig, behind)
    assert not valid[1].item()  # center camera


def test_dataset_shapes():
    ds = SyntheticDrivingSceneDataset(n_samples=4, seed=1)
    item = ds[0]
    assert item["cam_features"].shape == (N_CAM, 4, IMG_SIZE, IMG_SIZE)
    assert item["ego_history"].shape == (HIST_LEN, 2)
    assert item["command"].shape == (3,)
    assert item["future_gt"].shape == (FUT_LEN, 2)


def test_model_forward_shapes():
    torch.manual_seed(0)
    ds = SyntheticDrivingSceneDataset(n_samples=4, seed=2)
    batch = collate_scenes([ds[i] for i in range(4)])
    model = S2Planner(d_feat=16, n_scales=3, fut_len=FUT_LEN)
    out = model(batch["cam_features"], batch["ego_history"], batch["command"])
    assert out["trajectory"].shape == (4, FUT_LEN, 2)
    assert len(out["stage_trajectories"]) == 4  # init + 3 decoder layers
    assert out["last_cross_attn"].shape[0] == 4
    assert out["last_cross_attn"].shape[-1] == N_CAM


def test_cross_attention_gradient_flows_from_camera_features():
    """The core claim under test: predictions must actually depend on the
    sampled camera features (the paper's 'camera-projected cross-attention'),
    not just on the ego-conditioned initializer. A non-zero gradient into
    cam_features confirms the image branch participates in the forward pass."""
    torch.manual_seed(0)
    ds = SyntheticDrivingSceneDataset(n_samples=2, seed=3)
    batch = collate_scenes([ds[i] for i in range(2)])
    cam_features = batch["cam_features"].clone().requires_grad_(True)
    model = S2Planner(d_feat=16, n_scales=3, fut_len=FUT_LEN)
    out = model(cam_features, batch["ego_history"], batch["command"])
    loss = out["trajectory"].sum()
    loss.backward()
    assert cam_features.grad is not None
    assert cam_features.grad.abs().sum().item() > 0.0


def test_coarse_to_fine_refinement_improves_error_after_training():
    """Sanity/overfit check (this project's standing diagnostic): train briefly
    on a tiny fixed batch and confirm the FINAL stage ends up closer to ground
    truth than the initializer's own coarse guess -- i.e. the cross-attention
    refinement is actually contributing, not just along for the ride."""
    torch.manual_seed(0)
    ds = SyntheticDrivingSceneDataset(n_samples=6, seed=4)
    batch = collate_scenes([ds[i] for i in range(6)])
    model = S2Planner(d_feat=16, n_scales=3, fut_len=FUT_LEN)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    for _ in range(150):
        opt.zero_grad()
        out = model(batch["cam_features"], batch["ego_history"], batch["command"])
        loss, _ = coarse_to_fine_loss(out["stage_trajectories"], batch["future_gt"])
        loss.backward()
        opt.step()

    with torch.no_grad():
        out = model(batch["cam_features"], batch["ego_history"], batch["command"])
    init_ade, _ = ade_fde(out["stage_trajectories"][0], batch["future_gt"])
    final_ade, _ = ade_fde(out["trajectory"], batch["future_gt"])
    assert final_ade < init_ade * 0.6, f"expected clear improvement, got init={init_ade:.3f} final={final_ade:.3f}"
