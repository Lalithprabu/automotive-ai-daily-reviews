import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch, torch.nn.functional as F
from src.stcmgnet.model import STCMGNet, STCAttention, CrossModalChannelGate, NUM_JOINTS, normalized_adjacency
from src.stcmgnet.data import make_scene, window_tensors, build_split


def batch(B=4):
    return torch.randn(B, 4, 16, NUM_JOINTS), torch.randn(B, 16, 6), torch.randn(B, 16, 2)


def test_output_and_attention_shapes():
    m = STCMGNet().eval(); logit, attn = m(*batch(), return_attn=True)
    assert logit.shape == (4,) and attn["joint"].shape == (4, NUM_JOINTS) and attn["frame"].shape == (4, 16)


def test_stc_fusion_weights_init_and_sum():
    a = STCAttention(32)
    assert torch.allclose(a.fusion_weights(), torch.ones(3))
    a.theta.data = torch.tensor([2., 0., -1.]); assert abs(a.fusion_weights().sum().item() - 3) < 1e-5


def test_attention_maps_in_unit_interval():
    a = STCAttention(16); _, info = a(torch.randn(2, 16, 8, NUM_JOINTS))
    for k in ("joint", "frame"): assert (info[k] >= 0).all() and (info[k] <= 1).all()


def test_modulation_is_identity_at_init_then_active():
    g = CrossModalChannelGate(8, 5); x = torch.randn(2, 8, 4, 3)
    assert torch.allclose(g(x, torch.randn(2, 5), torch.randn(2, 5)), x)
    g.g_vis.weight.data.normal_(); v = torch.randn(2, 5)
    assert not torch.allclose(g(x, v, torch.randn(2, 5)), x)


def test_visual_and_ego_inputs_change_output_only_when_modulated():
    p, v, e = batch(); torch.manual_seed(0)
    base = STCMGNet(use_modulation=False).eval()
    assert torch.allclose(base(p, v, e), base(p, v * 0, e * 0))      # old way ignores context
    full = STCMGNet().eval()
    for g in full.gates: g.g_vis.weight.data.normal_(); g.g_ego.weight.data.normal_()
    assert not torch.allclose(full(p, v, e), full(p, v * 0, e * 0))


def test_gradient_reaches_every_parameter_when_gates_nonzero():
    m = STCMGNet(dropout=0.0)
    for g in m.gates: g.g_vis.weight.data.normal_(0, .1); g.g_ego.weight.data.normal_(0, .1)
    F.binary_cross_entropy_with_logits(m(*batch(8)), torch.ones(8)).backward()
    missing = [n for n, q in m.named_parameters() if q.grad is None or not torch.isfinite(q.grad).all()]
    assert not missing, missing


def test_adjacency_symmetric():
    A = normalized_adjacency(); assert torch.allclose(A, A.T)


def test_window_pose_has_no_absolute_curb_leak():
    s = make_scene(np.random.default_rng(0), 1)
    p, _, _ = window_tensors(s, 30); assert abs(p[0][:, [8, 11]].mean()) < 1e-4


def test_data_deterministic_and_balanced():
    a = build_split(40, 3); b = build_split(40, 3)
    assert torch.equal(a[0], b[0]) and a[3].mean().item() == 0.5


def test_overfit_tiny_batch():
    p, v, e, y = build_split(16, 1); torch.manual_seed(0)
    m = STCMGNet(dropout=0.0); opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(150):
        loss = F.binary_cross_entropy_with_logits(m(p, v, e), y); opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.2
