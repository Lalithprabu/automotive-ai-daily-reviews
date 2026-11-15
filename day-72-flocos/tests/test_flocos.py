import math

import numpy as np
import torch

from flocos import data as D
from flocos import geometry as G
from flocos import planner as PL
from flocos import world as W
from flocos.model import FloCOS, set_losses


def test_rect_support_is_tight_enclosure():
    """The octagon with rect_support offsets encloses the rectangle tightly: corners on the boundary, far points out."""
    psi = torch.tensor([0.3])
    c = torch.tensor([[1.0, -2.0]])
    s = G.rect_support(psi)
    corners = torch.as_tensor(W.footprint_corners(c.numpy(), psi.numpy()), dtype=torch.float32)  # (1, 4, 2)
    v = G.violation(corners, c, s)
    assert v.abs().max() < 1e-4
    far = c[:, None] + torch.tensor([[[3.0, 0.0]]])
    assert (G.violation(far, c, s) > 0).all()


def test_polygon_vertices_area_matches_mc():
    c = np.array([0.5, 0.2]); s = G.rect_support(torch.tensor(0.0)).numpy()
    poly = G.polygon_vertices(c, s)
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    assert abs(area - W.VEH_L * W.VEH_W) < 1e-3
    mc = G.union_area(torch.tensor(c, dtype=torch.float32)[None, None, None],
                      torch.tensor(s, dtype=torch.float32)[None, None, None], 20000,
                      gen=torch.Generator().manual_seed(0))
    assert abs(mc.item() - area) / area < 0.03


def test_bernstein_partition_of_unity_and_fit_roundtrip():
    tau = torch.linspace(0, 1, 7)
    assert torch.allclose(G.bernstein(tau).sum(-1), torch.ones(7), atol=1e-5)
    d = D.generate(200, 0)
    th = G.encode_tube(d["fut"], d["head"])
    c, s = G.decode(th[:, None])
    # straight-line keep-lane futures should be fitted (almost) exactly
    keep = d["intent"] == 0
    err = (c[keep, 0] - d["fut"][keep]).norm(dim=-1).max()
    assert err < 0.3
    assert G.coverage(c, s, d["fut"], d["head"]).mean() > 0.9


def test_union_is_min_over_components():
    p = torch.tensor([[[[0.0, 0.0], [10.0, 0.0]]]])                       # (1, 1, 2, 2)
    c = torch.tensor([[[[0.0, 0.0]], [[10.0, 0.0]]]])                     # (1, 2, 1, 2)
    s = G.rect_support(torch.zeros(1, 2, 1))
    assert (G.union_violation(p, c, s) <= 0).all()
    assert (G.violation(p[:, None], c, s)[0, 0, 0, 1] > 0)                # second point not in first comp


def test_union_area_of_disjoint_and_identical_components():
    s1 = G.rect_support(torch.zeros(1, 1, 1))
    c_same = torch.zeros(1, 2, 1, 2)
    c_apart = torch.tensor([[[[0.0, 0.0]], [[20.0, 0.0]]]])
    g = torch.Generator().manual_seed(0)
    a_same = G.union_area(c_same, s1.expand(1, 2, 1, -1), 40000, gen=g).item()
    a_apart = G.union_area(c_apart, s1.expand(1, 2, 1, -1), 40000, gen=g).item()
    rect = W.VEH_L * W.VEH_W
    assert abs(a_same - rect) / rect < 0.03 and abs(a_apart - 2 * rect) / (2 * rect) < 0.05


def test_set_losses_differentiable_and_flow_shapes():
    torch.manual_seed(0)
    d = D.generate(16, 1)
    m = FloCOS(32, 64, 2)
    c, s = m.predict_set(d, M=3, nfe=2)
    assert c.shape == (16, 3, W.H, 2) and s.shape == (16, 3, W.H, G.N_H) and (s >= 0).all()
    l_cov, area = set_losses(c, s, d["fut"], d["head"])
    (l_cov + 0.01 * area + m.fm_loss(d)).backward()
    g = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(g) > 0 and all(torch.isfinite(x).all() for x in g)


def test_coverage_loss_zero_when_covered():
    d = D.generate(8, 2)
    c = d["fut"][:, None]
    s = G.rect_support(d["head"])[:, None] + 0.2                          # true footprint, inflated
    l_cov, _ = set_losses(c, s, d["fut"], d["head"])
    assert l_cov.item() < 1e-6


def test_planner_true_set_never_collides():
    d = D.generate(400, 3)
    c, s = d["fut"][:, None], G.rect_support(d["head"])[:, None]
    r = PL.evaluate_planner(c, s, d)
    # planning against the true footprint with a margin can only collide if every candidate was infeasible
    infeasible_all = r["n_feasible_merge"] == 0
    assert r["collision"][~infeasible_all].mean() < 0.02


def test_reach_set_covers_almost_everything():
    d = D.generate(500, 4)
    c, ex = D.physics_reach_box(d)
    s = G.rect_support(None, ex=ex[..., 0], ey=ex[..., 1])
    assert G.coverage(c, s, d["fut"], d["head"]).mean() > 0.99
