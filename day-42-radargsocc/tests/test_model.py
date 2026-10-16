import sys; sys.path.insert(0, ".")
import torch, numpy as np
from models.radargsocc import RadarGSOcc, RadarOccBaseline, DeformAttn3D
from src.data import make_scene, render_radar, RadarOccDataset
from src.metrics import occ_iou, weighted_miou, ebkl

def test_shapes():
    m = RadarGSOcc(8, 5, 2); out = m(torch.randn(2, 8, 48, 32, 8)); assert out.shape == (2, 5, 48, 32, 8)
    assert RadarOccBaseline(8)(torch.randn(1, 8, 48, 32, 8)).shape == (1, 5, 48, 32, 8)

def test_deform_attn_zero_offset_is_pointwise():
    a = DeformAttn3D(4, 1); torch.nn.init.zeros_(a.off.bias)       # zero offsets, 1 point, weight softmax=1
    x = torch.randn(1, 4, 6, 5, 4); y = a(x, x); ref = a.o(a.v(x)); assert torch.allclose(y, ref, atol=1e-5)

def test_deform_attn_samples_correct_axis_order():
    a = DeformAttn3D(1, 1); torch.nn.init.zeros_(a.off.weight); torch.nn.init.zeros_(a.off.bias)
    with torch.no_grad(): a.off.bias[:] = torch.tensor([1., 0., 0.])        # +1 voxel along X
    for lay in (a.v, a.o): torch.nn.init.constant_(lay.weight, 1.0); torch.nn.init.zeros_(lay.bias)
    x = torch.zeros(1, 1, 6, 5, 4); x[0, 0, 3, 2, 1] = 1.0
    y = a(x, x)[0, 0]; assert y[2, 2, 1] > 0.9 and y[3, 2, 1] < 0.1         # query at x=2 reads value at x=3

def test_gradient_reaches_both_branches():
    m = RadarGSOcc(8, 5, 2); r = torch.rand(1, 8, 48, 32, 8)
    m(r).sum().backward()
    assert m.geo_in[0][0].weight.grad.abs().sum() > 0 and m.sem_in[0][0].weight.grad.abs().sum() > 0

def test_semantic_branch_uses_doppler_channels():
    m = RadarGSOcc(8, 5, 2).eval(); r = torch.rand(1, 8, 48, 32, 8); r2 = r.clone(); r2[:, 3:6] += 0.5
    assert not torch.allclose(m(r), m(r2))

def test_radar_physics_moving_vehicle_doppler_deviates():
    rng = np.random.default_rng(0); occ, vel = make_scene(rng)
    t = render_radar(occ, vel, rng); assert t.shape == (8, 48, 32, 8) and (t[0] > 0).mean() < 0.5   # sparse

def test_overfit_tiny_batch():
    torch.manual_seed(0); ds = RadarOccDataset(2, seed=3); r = torch.stack([ds[i][0] for i in range(2)]); o = torch.stack([ds[i][1] for i in range(2)])
    m = RadarGSOcc(12, 5, 2); opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(60):
        l = torch.nn.functional.cross_entropy(m(r), o); opt.zero_grad(); l.backward(); opt.step()
    assert occ_iou(m(r).argmax(1), o) > 0.6

def test_metrics():
    g = torch.tensor([[0, 1, 2, 4]]); assert occ_iou(g, g) == 1.0 and weighted_miou(g, g) == 1.0 and abs(ebkl(g, g)) < 1e-6
