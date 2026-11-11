"""Unit tests for the LOWP toy: geometry, reader invariants, gradients, mixing rule."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lowp.data import rig_tensors                                     # noqa: E402
from lowp.geometry import project_points, ray_points                 # noqa: E402
from lowp.model import LOWP, DeformableAnchorReader, GlobalRayReader, DenseIPMPlanner  # noqa: E402
from lowp.world import make_rig                                      # noqa: E402

HW = (40, 80)


def rig(B=2):
    K, R, t = rig_tensors(make_rig(*HW))
    return [x[None].expand(B, *x.shape).contiguous() for x in (K, R, t)]


def batch(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(B, 4, 3, *HW, generator=g)
    st = torch.rand(B, 2, generator=g)
    cmd = torch.tensor([0, 1] * (B // 2) + [2] * (B % 2))
    return img, st, cmd


def test_point_straight_ahead_projects_to_front_principal_column():
    K, R, t = rig(1)
    p = torch.tensor([[[30.0, 0.0, 1.6]]])                             # at camera height, dead ahead
    grid, valid, z = project_points(p, K, R, t, HW)
    assert valid[0, 0, 0] and not valid[0, 3, 0]                       # front sees it, back does not
    assert abs(grid[0, 0, 0, 0].item()) < 1e-4                         # centre column
    assert abs(z[0, 0, 0].item() - 28.5 * math.cos(math.radians(5))) < 0.1


def test_ray_points_reproject_to_their_own_cell():
    K, R, t = rig(1)
    depths = torch.tensor([1.0, 10.0, 60.0])
    pts = ray_points(K, R, t, HW, (5, 10), depths)                    # (1, 4, 5, 10, 3, 3)
    n, i, j = 1, 2, 7
    grid, valid, z = project_points(pts[0, n, i, j][None], K, R, t, HW)
    u = (grid[0, n, :, 0] + 1) / 2 * HW[1]
    v = (grid[0, n, :, 1] + 1) / 2 * HW[0]
    assert torch.allclose(u, torch.full_like(u, (j + 0.5) * 8), atol=1e-3)
    assert torch.allclose(v, torch.full_like(v, (i + 0.5) * 8), atol=1e-3)
    assert torch.allclose(z[0, n], depths, atol=1e-4)


def test_candidates_are_4_heights_by_8_ring_points():
    r = DeformableAnchorReader(32)
    anchor = torch.zeros(1, 16, 2)
    c = r.candidates(anchor, torch.zeros(1, 16, 8, 2))
    assert c.shape == (1, 16, 32, 3)
    assert torch.allclose(torch.unique(c[..., 2]), torch.linspace(0, 3, 4))
    assert torch.allclose(c[0, 0, :, :2].norm(dim=-1), torch.full((32,), 1.5), atol=1e-5)


def test_anchor_step_is_bounded():
    torch.manual_seed(0)
    r = DeformableAnchorReader(32)
    for l in r.layers:                                                 # force large raw steps
        torch.nn.init.constant_(l["step"].bias, 50.0)
    feat = torch.randn(2, 4, 32, 5, 10)
    K, R, t = rig(2)
    _, diag = r(feat, K, R, t, HW, torch.zeros(2, 32))
    for a0, a1 in zip(diag["anchors"][:-1], diag["anchors"][1:]):
        assert (a1 - a0).abs().max() <= r.max_step + 1e-5


def test_invisible_anchor_is_reseeded():
    r = DeformableAnchorReader(32)
    torch.nn.init.constant_(r.init_shift.bias, -50.0)                  # push all anchors far behind/right
    r.max_init_shift = 200.0
    feat = torch.randn(1, 1, 32, 5, 10)
    K, R, t = rig(1)
    K, R, t = K[:, :1], R[:, :1], t[:, :1]                             # front camera only
    _, diag = r(feat, K, R, t, HW, torch.zeros(1, 32))
    assert diag["n_reseeded"].item() == 16
    assert torch.allclose(diag["anchors"][0][0], r.anchor_grid)


def test_null_candidate_keeps_unseen_queries_finite():
    r = DeformableAnchorReader(32)
    feat = torch.randn(1, 1, 32, 5, 10)
    K, R, t = rig(1)
    K, R, t = K[:, 3:], R[:, 3:], t[:, 3:]                             # back camera only: grid ahead unseen
    q, diag = r(feat, K, R, t, HW, torch.zeros(1, 32))
    assert torch.isfinite(q).all()
    assert torch.allclose(diag["attn"][-1][..., 0], torch.ones(1, 16), atol=1e-5)   # all weight on NULL


def test_global_reader_attention_covers_every_cell_of_every_camera():
    g = GlobalRayReader(32)
    feat = torch.randn(2, 4, 32, 5, 10)
    K, R, t = rig(2)
    q, diag = g(feat, K, R, t, HW, torch.zeros(2, 32))
    assert q.shape == (2, 16, 32) and diag["attn"].shape == (2, 16, 4, 5, 10)
    assert torch.allclose(diag["attn"].sum((2, 3, 4)), torch.ones(2, 16), atol=1e-4)


def test_mixing_rule_matches_paper_alphas():
    torch.manual_seed(0)
    m = LOWP(32, img_hw=HW).eval()
    img, st, cmd = batch(3)
    K, R, t = rig(3)
    o = m(img, K, R, t, st, cmd)
    a = torch.tensor([0.5, 0.8, 0.8])[:, None, None]
    assert torch.allclose(o["tau"], a * o["tau_def"] + (1 - a) * o["tau_glob"], atol=1e-6)


def test_backward_reaches_both_experts_and_trunk():
    m = LOWP(32, img_hw=HW)
    img, st, cmd = batch(2)
    K, R, t = rig(2)
    o = m(img, K, R, t, st, cmd)
    m.loss(o, torch.randn(2, 6, 2)).backward()
    assert m.trunk.net[0][0].weight.grad.abs().sum() > 0
    assert m.defo.query.weight.grad.abs().sum() > 0
    assert m.glob.latents.grad.abs().sum() > 0


def test_lowp_builds_no_dense_bev_and_dense_baseline_runs():
    m = LOWP(32, img_hw=HW)
    assert not any("bev" in n for n, _ in m.named_parameters())
    d = DenseIPMPlanner(32, img_hw=HW)
    img, st, cmd = batch(2)
    K, R, t = rig(2)
    assert d(img, K, R, t, st, cmd)["tau"].shape == (2, 6, 2)
