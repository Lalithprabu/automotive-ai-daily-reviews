import math, os, yaml, numpy as np, torch
from lightrocc import LighTROcc, DenseBEVOccForecaster, gaussians_to_occupancy
from lightrocc.geometry import project, unproject, voxel_centers
from lightrocc.data import SceneSet, collate, make_scene
from lightrocc.losses import lighTROcc_loss
from lightrocc.metrics import instance_metrics, voxel_iou

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config.yaml")))
VOX = voxel_centers(CFG)


def _batch(n=4):
    ds = SceneSet(n, 123, CFG)
    return collate([ds[i] for i in range(n)])


def test_project_unproject_roundtrip():
    p = torch.tensor([[12.0, 3.0, 0.8], [25.0, -6.0, 1.1]])
    u, v, d = project(p, CFG)
    assert torch.allclose(unproject(u, v, d, CFG), p, atol=1e-4)


def test_voxel_grid_shape_and_range():
    assert VOX.shape == (32 * 32 * 4, 3)
    assert VOX[:, 0].min() > 2.0 and VOX[:, 0].max() < 34.0 and VOX[:, 2].max() < 2.0


def test_single_gaussian_occupancy_peaks_at_center_and_exist_gates():
    c = torch.tensor([[[[16.5, 0.5, 0.75]]]])                      # B=1,T1=1,Q=1
    kw = dict(yaws=torch.zeros(1, 1, 1), offsets=torch.zeros(1, 1, 1, 3), sigmas=torch.full((1, 1, 1, 3), 1.0),
              amps=torch.full((1, 1, 1), 0.99), vox=VOX)
    p = gaussians_to_occupancy(c, exist=torch.ones(1, 1), **kw)[0, 0]
    ctr = c[0, 0, 0]
    near = ((VOX - ctr).norm(dim=-1) < 0.8).nonzero().flatten()
    far = ((VOX - ctr).norm(dim=-1) > 6).nonzero().flatten()
    assert p[near].min() > 0.5 and p[far].max() < 1e-3
    assert gaussians_to_occupancy(c, exist=torch.zeros(1, 1), **kw).max() < 1e-4


def test_rotation_moves_offset_gaussian_rigidly():
    c = torch.tensor([[[[16.5, 0.5, 0.75]]]])
    off = torch.tensor([[[[2.0, 0.0, 0.0]]]])
    kw = dict(offsets=off, sigmas=torch.full((1, 1, 1, 3), 0.8), amps=torch.full((1, 1, 1), 0.99), exist=torch.ones(1, 1), vox=VOX)
    p0 = gaussians_to_occupancy(c, torch.zeros(1, 1, 1), **kw)[0, 0]
    p90 = gaussians_to_occupancy(c, torch.full((1, 1, 1), math.pi / 2), **kw)[0, 0]
    com = lambda p: (p[:, None] * VOX).sum(0) / p.sum()
    assert abs(com(p0)[0] - 18.5) < 0.3 and abs(com(p90)[1] - 2.5) < 0.3     # offset follows heading


def test_model_shapes_and_range():
    m = LighTROcc(CFG).eval(); b = _batch(3)
    out = m(b["img"], return_instances=True)
    assert out["probs"].shape == (3, 5, 4096) and out["centers"].shape == (3, 5, 6, 3)
    assert 0 <= out["probs"].min() and out["probs"].max() <= 1
    d = DenseBEVOccForecaster(CFG).eval()(b["img"])["probs"]
    assert d.shape == (3, 5, 4096)


def test_zero_disp_freezes_instances():
    m = LighTROcc(CFG).eval(); b = _batch(2)
    with torch.no_grad():
        p = m(b["img"], zero_disp=True)["probs"]
    assert torch.allclose(p[:, 0], p[:, -1], atol=1e-5)


def test_depth_head_receives_gradient_via_forward_lifting():
    m = LighTROcc(CFG); b = _batch(2)
    out = m(b["img"], return_instances=True)
    out["probs"].sum().backward()
    g = m.head[-1].weight.grad
    assert g[3].abs().sum() > 0                                     # depth output row gets gradient (zero-init head)
    torch.nn.init.normal_(m.head[-1].weight, std=0.05)              # head is zero-init: randomize it so gradient can reach the encoder
    m.zero_grad(); m(b["img"], return_instances=True)["probs"].sum().backward()
    assert m.encoder.net[0].weight.grad.abs().sum() > 0


def test_gt_voxelization_matches_box_volume():
    s = make_scene(5, CFG, VOX)
    n_veh = int(s["valid"].sum())
    vol = s["occ"][0].sum().item()                                  # voxel COUNT: footprint cells x 3 voxel layers (z<1.6 m)
    expect = sum(float(L * W) * 3.0 for L, W, _ in s["size"][: n_veh].tolist())
    assert 0.5 * expect < vol < 1.5 * expect


def test_loss_finite_and_backward():
    m = LighTROcc(CFG); b = _batch(3)
    loss, parts = lighTROcc_loss(m(b["img"], return_instances=True), b, CFG)
    assert torch.isfinite(loss); loss.backward()


def test_instance_metrics_perfect_and_miss():
    b = _batch(2)
    perfect = [[b["cen"][i, :, t].numpy()[b["valid"][i].numpy()] for t in range(5)] for i in range(2)]
    f1, ade = instance_metrics(perfect, b, CFG)
    assert f1[0] > 0.99 and ade[0] < 1e-6
    f1m, _ = instance_metrics([[np.zeros((0, 2))] * 5] * 2, b, CFG)
    assert f1m[0] == 0.0


def test_voxel_iou_identity():
    b = _batch(2)
    assert torch.allclose(voxel_iou(b["occ"], b["occ"]), torch.ones(5))


def test_overfit_tiny_batch_reduces_loss():
    torch.manual_seed(0)
    m = LighTROcc(CFG); b = _batch(4)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = last = None
    for i in range(40):
        loss, _ = lighTROcc_loss(m(b["img"], return_instances=True), b, CFG)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first or loss.item(); last = loss.item()
    assert last < 0.8 * first
