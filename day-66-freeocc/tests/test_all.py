import torch
from models.freeocc import FreeOcc
from src.geometry import warp_bev, rel_pose, inv_pose, G
from src.synthetic import make_batch, make_scene, render, ego_pose, HIST, H, DT


def test_pose_inverse():
    a = torch.tensor([[1.0, 2.0, 0.7]]); b = torch.tensor([[4.0, -1.0, 1.1]])
    p = rel_pose(a, b); q = rel_pose(b, a)
    assert torch.allclose(q, inv_pose(p), atol=1e-5)


def test_warp_matches_renderer():
    """Warping the TRUE occupancy of frame A into frame B must reproduce the true occupancy of B for STATIC content
    (checks the affine_grid axis conventions end to end)."""
    g = torch.Generator().manual_seed(0); sc = make_scene(6, g); sc["mov_vel"] = sc["mov_vel"] * 0   # freeze movers
    t = torch.tensor([0.0, 1.0]); occ = render(sc, t); pose = ego_pose(sc, t)
    warped = warp_bev(occ[:, 0:1], rel_pose(pose[:, 0], pose[:, 1]), mode="nearest")[:, 0]
    inner = (slice(None), slice(8, G - 8), slice(8, G - 8))                      # ignore cells that left the window
    iou = ((warped[inner] > 0.5) & (occ[:, 1][inner] > 0.5)).sum() / ((warped[inner] > 0.5) | (occ[:, 1][inner] > 0.5)).sum()
    assert iou > 0.7, iou
    # and a wrong convention (negated yaw) must be clearly worse
    bad = warp_bev(occ[:, 0:1], rel_pose(pose[:, 0], pose[:, 1]) * torch.tensor([1.0, 1.0, -1.0]), mode="nearest")[:, 0]
    iou_bad = ((bad[inner] > 0.5) & (occ[:, 1][inner] > 0.5)).sum() / ((bad[inner] > 0.5) | (occ[:, 1][inner] > 0.5)).sum()
    assert iou_bad < iou - 0.3


def test_shapes_and_sampling():
    b = make_batch(2, torch.Generator().manual_seed(1)); m = FreeOcc().eval()
    out = m(b["hist_gt"], b["hist_rel"], b["fut_rel"], sample=True)
    for k in ("p_dec", "p_ode", "p", "w", "logvar"):
        assert out[k].shape == (2, H, G, G), k
    assert ((out["p"] >= 0) & (out["p"] <= 1)).all()


def test_zero_init_is_pure_warp():
    """res_proj is zero-initialised -> at init the future state is exactly the warped history state."""
    b = make_batch(2, torch.Generator().manual_seed(2)); m = FreeOcc().eval()
    h = m.encode_state(b["hist_gt"], b["hist_rel"]); hs = m.propagate(h, b["fut_rel"])
    ref = h
    prev = torch.zeros_like(b["fut_rel"][:, 0])
    for k in range(H):
        ref = warp_bev(ref, rel_pose(prev, b["fut_rel"][:, k])); prev = b["fut_rel"][:, k]
        assert torch.allclose(hs[:, k], ref, atol=1e-5)


def test_history_is_causal_in_future_inputs():
    """Forecast must not depend on future OCCUPANCY (only on future poses): perturbing fut_gt changes nothing."""
    b = make_batch(2, torch.Generator().manual_seed(3)); m = FreeOcc().eval()
    a = m(b["hist_gt"], b["hist_rel"], b["fut_rel"])["p_dec"]
    b["fut_gt"] = 1 - b["fut_gt"]
    c = m(b["hist_gt"], b["hist_rel"], b["fut_rel"])["p_dec"]
    assert torch.allclose(a, c)


def test_flow_loss_backward():
    b = make_batch(2, torch.Generator().manual_seed(4)); m = FreeOcc()
    out = m(b["hist_gt"], b["hist_rel"], b["fut_rel"]); loss = m.flow_loss(out["f"], b["fut_gt"].flatten(0, 1))
    loss.backward(); assert torch.isfinite(loss) and m.flow.out.weight.grad is not None


def test_flow_sampler_deterministic_with_generator():
    b = make_batch(1, torch.Generator().manual_seed(5)); m = FreeOcc().eval()
    o1 = m(b["hist_gt"], b["hist_rel"], b["fut_rel"], sample=True, gen=torch.Generator().manual_seed(9))["p_ode"]
    o2 = m(b["hist_gt"], b["hist_rel"], b["fut_rel"], sample=True, gen=torch.Generator().manual_seed(9))["p_ode"]
    assert torch.allclose(o1, o2)
