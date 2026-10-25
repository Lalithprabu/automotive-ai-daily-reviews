import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from src.synthetic import build_dataset, render_frame, bev_targets, MAX_D, BEV_N
from src.foundation4d import Foundation4DMini, compute_loss, attention_pairs
from src.metrics import det_f1


def _scene(pos, half=(1.0, 2.2)):
    return dict(pos0=torch.tensor([[pos]]), vel=torch.zeros(1, 1, 2), half=torch.tensor([[half]]), mask=torch.ones(1, 1, dtype=torch.bool))


def test_raycast_depth_front_camera():
    # box centred 10 m ahead, half-extent 2.2 in y -> front face at y=7.8; centre pixel ray (fwd only, z=0 height 1.5<1.6) hits at 7.8
    rgb, depth, _ = render_frame(_scene((0.0, 10.0)), 0.0, 24, 32, noise=0.0)
    assert abs(depth[0, 0, 12, 16].item() - 7.8) < 0.15
    assert depth[0, 2].min() > 1.0 and (depth[0, 2, 0] == MAX_D).all()      # rear camera sees only ground/sky


def test_ground_depth_matches_geometry():
    _, depth, _ = render_frame(_scene((0.0, -50.0)), 0.0, 24, 32, noise=0.0)
    f = 16.0; v = (23 + 0.5 - 12) / f                                     # bottom row
    assert abs(depth[0, 0, 23, 16].item() - 1.5 / v) < 1e-3


def test_targets_centre_cell_is_one_and_occupancy_layers():
    pos = torch.tensor([[[3.1, -5.2]]])
    heat, occ = bev_targets(pos, torch.tensor([[[1.0, 2.2]]]), torch.ones(1, 1, dtype=torch.bool))
    assert heat.max() == 1.0 and (heat == 1.0).sum() == 1
    assert occ[0, 0].sum() > 0 and occ[0, 3].sum() == 0                    # 1.6 m box fills layers 0-2, not layer 3 (z>=1.8)


def test_shapes_and_positive_depth():
    net = Foundation4DMini(dim=32, heads=2, depth=1)
    d = build_dataset(2, 3)
    o = net(d["rgb"])
    assert o["depth"].shape == (2, 4, 2, 24, 32) and (o["depth"] > 0).all()
    assert o["det"].shape == (2, 1, BEV_N, BEV_N) and o["occ"].shape[1] == 4
    assert o["gs"]["rgb"].shape == (2, 4, 2, 3, 24, 32)


def test_cross_view_and_temporal_information_flow():
    torch.manual_seed(0)
    net = Foundation4DMini(dim=32, heads=2, depth=2).eval()
    x = torch.rand(1, 4, 2, 3, 24, 32)
    base = net(x)["depth"]
    x2 = x.clone(); x2[:, 0] += 0.5                                         # perturb camera 0 only
    d2 = net(x2)["depth"]
    assert (d2[:, 1] - base[:, 1]).abs().max() > 1e-6                       # cross-view attention reaches camera 1
    x3 = x.clone(); x3[:, :, 0] += 0.5                                      # perturb frame 0 only
    assert (net(x3)["depth"][:, :, 1] - base[:, :, 1]).abs().max() > 1e-6   # temporal attention reaches frame 1


def test_factorized_cheaper_than_dense():
    assert attention_pairs(4, 2, 48, True) < attention_pairs(4, 2, 48, False)
    assert attention_pairs(6, 8, 200, True) * 5 < attention_pairs(6, 8, 200, False)


def test_gradient_flows_to_all_heads():
    net = Foundation4DMini(dim=32, heads=2, depth=1)
    d = build_dataset(2, 4)
    loss, parts = compute_loss(net(d["rgb"]), d)
    loss.backward()
    for n in ("depth_head", "gs_head", "det_head", "occ_head", "bev_proj", "embed"):
        p = [q for k, q in net.named_parameters() if k.startswith(n)]
        assert p and all(q.grad is not None and q.grad.abs().sum() > 0 for q in p), n


def test_overfit_tiny_batch():
    torch.manual_seed(0)
    net = Foundation4DMini(dim=48, heads=4, depth=2)
    d = build_dataset(4, 5)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    first = None
    for i in range(120):
        loss, _ = compute_loss(net(d["rgb"]), d); opt.zero_grad(); loss.backward(); opt.step()
        first = first or loss.item()
    assert loss.item() < 0.5 * first
    net.eval()
    with torch.no_grad():
        p, r, f1 = det_f1(net(d["rgb"])["det"], d["heat"], 0.3)
    assert f1 > 0.5
