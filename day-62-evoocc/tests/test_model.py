import math, torch
from src.warp import warp_bev
from src.evoocc import EvoOcc
from src import data


def test_identity_warp():
    x = torch.randn(2, 4, 48, 48)
    assert torch.allclose(warp_bev(x, torch.zeros(2, 3), 48.0), x, atol=1e-4)


def test_translation_warp_moves_content():
    x = torch.zeros(1, 1, 48, 48); x[0, 0, 24, 30] = 1.0          # blob 6 cells ahead (+x)
    y = warp_bev(x, torch.tensor([[3.0, 0.0, 0.0]]), 48.0)         # ego advanced 3 m -> blob now 3 cells ahead
    assert y[0, 0].argmax().item() == 24 * 48 + 27


def test_rotation_warp_roundtrip_center():
    x = torch.zeros(1, 1, 48, 48); x[0, 0, 22:26, 22:26] = 1.0
    y = warp_bev(x, torch.tensor([[0., 0., math.pi / 6]]), 48.0)
    assert abs(y.sum().item() - x.sum().item()) < 1.0


def test_forward_shapes_and_zero_init_identity():
    m = EvoOcc(hidden=16); b = data.make_batch(2, T=3, seed=0)
    out = m(b["slow0"], b["fast"][:, :4], b["ego"][:, :3], b["dts"][:, :3])
    assert out.shape == (2, 4, 3, 48, 48)
    assert torch.allclose(out, b["fast"][:, :4])                   # zero-init decoder == fast logits


def test_gradients_flow_to_all_modules():
    m = EvoOcc(hidden=16); b = data.make_batch(2, T=3, seed=1)
    m.dyn.out.weight.data.normal_(0, .1); m.decoder[-1].weight.data.normal_(0, .1)
    out = m(b["slow0"], b["fast"][:, :4], b["ego"][:, :3], b["dts"][:, :3])
    torch.nn.functional.cross_entropy(out.flatten(0, 1), b["gt"][:, :4].flatten(0, 1)).backward()
    for n, p in m.named_parameters():
        if n.startswith("time_ctrl"): continue
        assert p.grad is not None and p.grad.abs().sum() > 0, n


def test_irregular_dt_changes_output():
    m = EvoOcc(hidden=16); m.dyn.out.weight.data.normal_(0, .2); m.decoder[-1].weight.data.normal_(0, .2)
    b = data.make_batch(1, T=2, seed=2)
    a = m(b["slow0"], b["fast"][:, :3], b["ego"][:, :2], torch.full((1, 2), .25))
    c = m(b["slow0"], b["fast"][:, :3], b["ego"][:, :2], torch.full((1, 2), .75))
    assert not torch.allclose(a, c)


def test_time_only_ablation_runs():
    m = EvoOcc(hidden=16, use_control=False); b = data.make_batch(1, T=2, seed=3)
    assert m(b["slow0"], b["fast"][:, :3], b["ego"][:, :2], b["dts"][:, :2]).shape[1] == 3


def test_data_labels_and_ego_consistency():
    b = data.make_batch(4, T=4, seed=4)
    assert set(b["gt"].unique().tolist()) <= {0, 1, 2} and (b["gt"] == 2).any() and (b["gt"] == 1).any()
    # warping GT static cells by the true ego motion should match next frame's static cells well
    g0 = (b["gt"][:, 0] == 1).float()[:, None]; g1 = (b["gt"][:, 1] == 1).float()[:, None]
    w = warp_bev(g0, b["ego"][:, 0], 48.0, mode="nearest")
    inter = ((w > .5) & (g1 > .5)).sum().float(); union = ((w > .5) | (g1 > .5)).sum().float()
    assert (inter / union) > 0.6
