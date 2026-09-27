import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import SyntheticSceneDataset, collate_scenes
from src.geometry import BEVGrid, PinholeCamera, default_camera_rig
from src.model import CameraBEVLifter, PriorMapBEVFusion, PriorMapBEVNet, SparseVoxelPriorEncoder
from train import balanced_bce, box_loss, compute_losses, focal_loss


def make_batch(n=3, seed=0, grid=None):
    grid = grid or BEVGrid()
    ds = SyntheticSceneDataset(n, grid, seed=seed)
    return collate_scenes([ds[i] for i in range(n)])


def test_camera_geometry_forward_point_is_centered():
    cam = PinholeCamera(name="front", yaw_deg=0.0)
    u, v, valid = cam.world_to_pixel(torch.tensor([[0.0, 10.0]]))
    assert valid.item()
    assert abs(u.item() - cam.img_w / 2) < 1e-4


def test_camera_geometry_left_point_has_smaller_u():
    cam = PinholeCamera(name="front", yaw_deg=0.0)
    u_left, _, v_left = cam.world_to_pixel(torch.tensor([[-3.0, 10.0]]))
    u_right, _, v_right = cam.world_to_pixel(torch.tensor([[3.0, 10.0]]))
    assert v_left.item() and v_right.item()
    assert u_left.item() < u_right.item()


def test_camera_geometry_behind_camera_is_invalid():
    cam = PinholeCamera(name="front", yaw_deg=0.0)
    _, _, valid = cam.world_to_pixel(torch.tensor([[0.0, -5.0]]))
    assert not valid.item()


def test_bev_grid_world_to_cell_roundtrip_center():
    grid = BEVGrid()
    centers = grid.cell_centers()
    rc = grid.world_to_cell(centers.view(-1, 2)).view(grid.grid_h, grid.grid_w, 2)
    rows = torch.arange(grid.grid_h).float().view(-1, 1).expand(grid.grid_h, grid.grid_w)
    cols = torch.arange(grid.grid_w).float().view(1, -1).expand(grid.grid_h, grid.grid_w)
    assert torch.allclose(rc[..., 0], rows, atol=1e-3)
    assert torch.allclose(rc[..., 1], cols, atol=1e-3)


def test_sparse_voxel_prior_encoder_output_shape():
    grid = BEVGrid()
    enc = SparseVoxelPriorEncoder(grid=grid, out_channels=16)
    batch = make_batch(2, grid=grid)
    out = enc(batch["prior_points"])
    assert out.shape == (2, 16, grid.grid_h, grid.grid_w)


def test_camera_bev_lifter_output_shape():
    grid = BEVGrid()
    cams = default_camera_rig()
    lifter = CameraBEVLifter(cams, out_channels=16, grid=grid)
    batch = make_batch(2, grid=grid)
    out = lifter(batch["camera_feats"])
    assert out.shape == (2, 16, grid.grid_h, grid.grid_w)


def test_fusion_disabled_zeros_prior_contribution_changes_output():
    fusion = PriorMapBEVFusion(channels=8)
    live = torch.randn(1, 8, 4, 4)
    prior = torch.randn(1, 8, 4, 4) * 5.0
    out_on = fusion(live, prior, prior_enabled=True)
    out_off = fusion(live, prior, prior_enabled=False)
    assert not torch.allclose(out_on, out_off)


def test_full_model_forward_shapes():
    grid = BEVGrid()
    model = PriorMapBEVNet(default_camera_rig(), grid=grid, channels=16)
    batch = make_batch(2, grid=grid)
    out = model(batch["prior_points"], batch["camera_feats"])
    assert out["det_heatmap"].shape == (2, 1, grid.grid_h, grid.grid_w)
    assert out["det_boxes"].shape == (2, 6, grid.grid_h, grid.grid_w)
    assert out["map_heatmap"].shape == (2, 2, grid.grid_h, grid.grid_w)


def test_gradient_flows_from_detection_loss_to_prior_encoder():
    grid = BEVGrid()
    model = PriorMapBEVNet(default_camera_rig(), grid=grid, channels=16)
    batch = make_batch(2, grid=grid)
    out = model(batch["prior_points"], batch["camera_feats"])
    loss = focal_loss(out["det_heatmap"], batch["det_heatmap"])
    loss.backward()
    grad_norm = sum(
        p.grad.abs().sum().item() for p in model.prior_encoder.parameters() if p.grad is not None
    )
    assert grad_norm > 0, "gradients should flow into the prior encoder from the detection loss"


def test_gradient_flows_from_map_loss_to_camera_lifter():
    grid = BEVGrid()
    model = PriorMapBEVNet(default_camera_rig(), grid=grid, channels=16)
    batch = make_batch(2, grid=grid)
    out = model(batch["prior_points"], batch["camera_feats"])
    loss = balanced_bce(out["map_heatmap"], batch["map_heatmap"])
    loss.backward()
    grad_norm = sum(
        p.grad.abs().sum().item() for p in model.cam_lifter.parameters() if p.grad is not None
    )
    assert grad_norm > 0, "gradients should flow into the camera lifter from the map loss"


def test_focal_loss_zero_for_perfect_prediction():
    gt = torch.zeros(1, 1, 4, 4)
    gt[0, 0, 2, 2] = 1.0
    pred = gt.clone().clamp(1e-4, 1 - 1e-4)
    loss = focal_loss(pred, gt)
    assert loss.item() < 1e-2


def test_balanced_bce_penalizes_missed_positive_more_when_positives_are_rare():
    gt = torch.zeros(1, 1, 10, 10)
    gt[0, 0, 0, 0] = 1.0  # 1 positive out of 100 cells
    all_zero_pred = torch.full_like(gt, 1e-3)
    correct_pred = gt.clone().clamp(1e-3, 1 - 1e-3)
    loss_wrong = balanced_bce(all_zero_pred, gt)
    loss_right = balanced_bce(correct_pred, gt)
    assert loss_wrong.item() > loss_right.item()


def test_box_loss_only_supervises_masked_cells():
    pred = torch.ones(1, 6, 4, 4)
    gt = torch.zeros(1, 6, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    loss_no_mask_hit = box_loss(pred, gt, mask)
    mask[0, 0, 1, 1] = 1.0
    loss_with_hit = box_loss(pred, gt, mask)
    assert loss_no_mask_hit.item() == 0.0
    assert loss_with_hit.item() > 0.0


def test_overfit_tiny_batch_reduces_loss_substantially():
    grid = BEVGrid()
    torch.manual_seed(0)
    model = PriorMapBEVNet(default_camera_rig(), grid=grid, channels=16)
    batch = make_batch(2, grid=grid)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    losses = []
    for _ in range(120):
        opt.zero_grad()
        out = model(batch["prior_points"], batch["camera_feats"])
        loss = compute_losses(out, batch, box_w=0.5, map_w=1.0)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, f"expected substantial loss decrease, got {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_prior_map_helps_mapping_more_than_detection_after_overfitting():
    """Regression test for this repo's core empirical claim (mirroring the
    paper's own qualitative pattern): once trained with heavy live-camera
    occlusion, disabling the prior should hurt map-heatmap reconstruction
    on the *training* scenes far more than detection, since the prior
    supplies exactly the occluded static geometry and never contains the
    dynamic objects the detector must find."""
    grid = BEVGrid()
    torch.manual_seed(1)
    model = PriorMapBEVNet(default_camera_rig(), grid=grid, channels=16)
    batch = make_batch(4, grid=grid, seed=7)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    for _ in range(150):
        opt.zero_grad()
        out_on = model(batch["prior_points"], batch["camera_feats"], prior_enabled=True)
        out_off = model(batch["prior_points"], batch["camera_feats"], prior_enabled=False)
        loss = compute_losses(out_on, batch, 0.5, 1.0) + compute_losses(out_off, batch, 0.5, 1.0)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        out_on = model(batch["prior_points"], batch["camera_feats"], prior_enabled=True)
        out_off = model(batch["prior_points"], batch["camera_feats"], prior_enabled=False)

    def map_iou(pred, gt, thresh=0.5):
        pred_b, gt_b = pred > thresh, gt > 0.5
        inter = (pred_b & gt_b).float().sum().item()
        union = (pred_b | gt_b).float().sum().item()
        return inter / union if union > 0 else 1.0

    def det_l1(pred_heat, gt_heat):
        return (pred_heat - gt_heat).abs().mean().item()

    map_iou_on = map_iou(out_on["map_heatmap"], batch["map_heatmap"])
    map_iou_off = map_iou(out_off["map_heatmap"], batch["map_heatmap"])
    det_err_on = det_l1(out_on["det_heatmap"], batch["det_heatmap"])
    det_err_off = det_l1(out_off["det_heatmap"], batch["det_heatmap"])

    map_gap = map_iou_on - map_iou_off
    det_gap = abs(det_err_off - det_err_on)

    assert map_gap > 0, f"prior should help mapping IoU on these occluded scenes, got on={map_iou_on:.3f} off={map_iou_off:.3f}"
    assert map_gap > det_gap, (
        f"expected the prior's benefit to mapping ({map_gap:.3f}) to exceed its "
        f"effect on detection ({det_gap:.3f}), mirroring the paper's own pattern "
        f"(small CDS delta, large mAP delta)"
    )
