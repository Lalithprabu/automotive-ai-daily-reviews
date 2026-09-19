"""Unit + regression tests for the MM-Future reconstruction.

Run with: pytest tests/ -v
"""
import numpy as np
import torch
import yaml

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import MMFutureDataset, extract_camera_crops, SyntheticDrivingScenes
from src.model import MMFuture
from src.action_prior import ActionPrior


def _tiny_cfg():
    with open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    # shrink further for fast tests
    cfg["data"]["num_train_scenes"] = 12
    cfg["data"]["num_val_scenes"] = 4
    cfg["model"]["num_proposals"] = 4
    cfg["model"]["n_flow_layers"] = 2
    cfg["model"]["d_model"] = 32
    cfg["model"]["patch_dim"] = 16
    cfg["model"]["chunk_tokens"] = 8
    cfg["model"]["register_tokens_per_camera"] = 2
    cfg["model"]["num_action_clusters"] = 4
    return cfg


def test_camera_crop_shape():
    frame = np.random.rand(2, 32, 32).astype(np.float32)
    crops = extract_camera_crops(frame, np.array([16.0, 16.0]), crop=16)
    assert crops.shape == (4, 2, 16, 16)


def test_scene_generation_bimodal():
    rng = np.random.default_rng(0)
    gen = SyntheticDrivingScenes(32, 4, 4, 8, rng)
    scenes = [gen.sample() for _ in range(50)]
    lefts = sum(1 for s in scenes if s["go_left"])
    assert 5 < lefts < 45  # roughly balanced, not degenerate
    assert scenes[0]["action_deltas"].shape == (8, 4)
    assert scenes[0]["gt_trajectory"].shape == (8, 2)


def test_dataset_item_shapes():
    cfg = _tiny_cfg()
    d = cfg["data"]
    ds = MMFutureDataset(6, d["grid_size"], d["num_agents"], d["history_frames"],
                          d["future_frames"], d["camera_crop"], seed=1)
    item = ds[0]
    assert item["history_cameras"].shape == (d["history_frames"], 4, 2, d["camera_crop"], d["camera_crop"])
    assert item["future_cameras"].shape == (d["future_frames"], 4, 2, d["camera_crop"], d["camera_crop"])
    assert item["action_deltas"].shape == (d["future_frames"], 4)


def test_action_prior_fit_and_sample():
    prior = ActionPrior(num_clusters=4, future_frames=8, action_dim=4)
    deltas = np.random.randn(30, 8, 4).astype(np.float32)
    prior.fit(deltas)
    z0 = prior.sample(batch_size=2, num_proposals=4, device=torch.device("cpu"))
    assert z0.shape == (2, 4, 8, 4)
    assert torch.isfinite(z0).all()


def test_forward_train_finite_and_shapes():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    d = cfg["data"]
    ds = MMFutureDataset(8, d["grid_size"], d["num_agents"], d["history_frames"],
                          d["future_frames"], d["camera_crop"], seed=2)
    model = MMFuture(cfg)
    all_deltas = np.stack([s["action_deltas"] for s in ds.scenes])
    model.action_prior.fit(all_deltas)

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    out = model.forward_train(batch, torch.device("cpu"))

    for key in ["loss", "action_loss", "scene_loss", "score_loss", "bev_loss", "ade", "top1_acc"]:
        assert torch.isfinite(out[key]), f"{key} is not finite: {out[key]}"
    assert out["loss"].item() > 0


def test_backward_pass_updates_weights():
    cfg = _tiny_cfg()
    torch.manual_seed(1)
    d = cfg["data"]
    ds = MMFutureDataset(8, d["grid_size"], d["num_agents"], d["history_frames"],
                          d["future_frames"], d["camera_crop"], seed=3)
    model = MMFuture(cfg)
    all_deltas = np.stack([s["action_deltas"] for s in ds.scenes])
    model.action_prior.fit(all_deltas)

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    before = model.flow_transformer.blocks[0].hist_ffn.net[0].weight.clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    out = model.forward_train(batch, torch.device("cpu"))
    out["loss"].backward()
    opt.step()
    after = model.flow_transformer.blocks[0].hist_ffn.net[0].weight
    assert not torch.allclose(before, after), "weights did not update after a backward+step"


def test_generate_no_nan_and_expected_shapes():
    cfg = _tiny_cfg()
    torch.manual_seed(2)
    d = cfg["data"]
    ds = MMFutureDataset(4, d["grid_size"], d["num_agents"], d["history_frames"],
                          d["future_frames"], d["camera_crop"], seed=4)
    model = MMFuture(cfg)
    all_deltas = np.stack([s["action_deltas"] for s in ds.scenes])
    model.action_prior.fit(all_deltas)
    model.eval()

    item = ds[0]
    history_cameras = item["history_cameras"].unsqueeze(0)
    ego_last = item["ego_history"][-1].unsqueeze(0)
    result = model.generate(history_cameras, ego_last, torch.device("cpu"), ode_steps=3)

    m = cfg["model"]["num_proposals"]
    ta = d["future_frames"]
    assert result["trajectories"].shape == (1, m, ta, 2)
    assert result["scores"].shape == (1, m)
    assert torch.isfinite(result["trajectories"]).all()
    assert torch.isfinite(result["scores"]).all()
    assert torch.isfinite(result["bev_pred"]).all()
    assert result["bev_pred"].min() >= 0 and result["bev_pred"].max() <= 1


def test_asymmetric_mask_blocks_future_leakage_into_history():
    from src.flow_transformer import build_asymmetric_mask
    mask = build_asymmetric_mask(hist_len=3, action_len=2, scene_len=2, device=torch.device("cpu"))
    # history rows (0..2) must not attend to future cols (3..6)
    assert mask[:3, 3:].all()
    # future rows may attend everywhere
    assert not mask[3:, :].any()


def test_best_of_many_selects_closest_proposal():
    from src.losses import select_best_of_many
    gt = torch.zeros(1, 4, 2)
    traj = torch.stack([
        torch.ones(4, 2) * 5.0,
        torch.zeros(4, 2) + 0.01,
        torch.ones(4, 2) * -3.0,
    ]).unsqueeze(0)  # [1, 3, 4, 2]
    m_star = select_best_of_many(traj, gt)
    assert m_star.item() == 1
