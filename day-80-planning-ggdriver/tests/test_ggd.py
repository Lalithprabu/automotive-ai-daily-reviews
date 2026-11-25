import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ggd.model import MODES, CrossAttn, GlanceGazePlanner, corridor_mask, random_mask_like  # noqa: E402
from ggd.scene import (ANCHORS, GW, make_dataset, make_scene, render, scene_targets, score_traj,  # noqa: E402
                       traj_points)


def test_traj_kinematics():
    p = traj_points(torch.tensor([[[3.5, 0.0], [0.0, 16.0]]]), torch.tensor([[10.0]]))
    assert p.shape == (1, 2, 16, 2)
    z_brake, z_fast = p[0, 0, :, 1], p[0, 1, :, 1]
    assert torch.all(z_brake[1:] >= z_brake[:-1]) and z_brake[-1] < 10.0 * 4 / 2 + 1     # stops: <= v0^2/12 + margin
    assert z_fast[-1] > 40 * 1.0
    assert torch.allclose(p[0, 1, :, 0], torch.zeros(16))


def test_score_collision_and_offroad():
    pts = traj_points(torch.tensor([[[0.0, 10.0], [7.0, 10.0]]]), torch.tensor([[10.0]]))
    obj = torch.tensor([[[0.0, 25.0, 0.8, 0.6]]])
    s = score_traj(pts, torch.tensor([[[0.0, 10.0], [7.0, 10.0]]]), torch.tensor([10.0]), obj,
                   torch.tensor([[True]]), torch.tensor([[-1.75, 5.25]]))
    assert s["nc"][0, 0] == 0 and s["score"][0, 0] == 0                      # drives through the debris
    assert s["nc"][0, 1] == 1 and s["dac"][0, 1] == 0                        # swerves off the road


def test_corridor_follows_the_path():
    A = torch.tensor([[[0.0, 8.0], [-7.0, 8.0], [7.0, 8.0]]])
    m = corridor_mask(traj_points(A, torch.tensor([[8.0]])))
    cols = [torch.nonzero(m[0, k]).squeeze(-1) % GW for k in range(3)]
    assert 0 < m[0, 0].sum() < 0.3 * 240
    assert cols[1].float().mean() < cols[0].float().mean() < cols[2].float().mean()


def test_random_mask_matches_count():
    m = corridor_mask(traj_points(torch.as_tensor(ANCHORS)[None], torch.tensor([[9.0]])))
    r = random_mask_like(m)
    assert torch.equal(r.sum(-1), m.sum(-1)) and not torch.equal(r, m)


def test_masked_attention_ignores_outside_tokens():
    torch.manual_seed(0)
    a = CrossAttn(32).eval()
    q, kv = torch.randn(1, 2, 32), torch.randn(1, 10, 32)
    mask = torch.zeros(1, 2, 10, dtype=torch.bool)
    mask[0, 0, :3] = True
    mask[0, 1, 5:] = True
    y = a(q, kv, mask)
    kv2 = kv.clone()
    kv2[0, 3:] += 5.0 * torch.randn(7, 32)  # perturb tokens outside query 0's mask
    y2 = a(q, kv2, mask)
    assert torch.allclose(y[0, 0], y2[0, 0], atol=1e-6) and not torch.allclose(y[0, 1], y2[0, 1])


def test_planner_all_modes():
    x, v0 = torch.rand(2, 3, 96, 160), torch.tensor([6.0, 12.0])
    for r in MODES:
        m = GlanceGazePlanner(r, r, d=32, n_reg=4)
        o = m(x, v0)
        assert o["score"].shape == (2, 45) and o["dv"].shape == (2, 45, 2)
        assert torch.all((o["score"] >= 0) & (o["score"] <= 1))
        o["score"].sum().backward()
        p = m.eval().plan(x, v0, 8)
        assert p["dv"].shape == (2, 2)


def test_glance_scorer_ignores_patches_but_gaze_does_not():
    torch.manual_seed(0)
    x, v0 = torch.rand(1, 3, 96, 160), torch.tensor([9.0])
    for mode in ("glance", "gaze"):
        m = GlanceGazePlanner("glance", mode, d=32, n_reg=4).eval()
        P = m.backbone(x)
        g = m.reg.expand(1, -1, -1)
        for a in m.reg_attn:
            g = a(g, P)
        G = torch.cat([g, m.ego(v0[:, None] / 16.0)[:, None]], 1)
        kv, mask, _ = m.evidence(mode, P, G, traj_points(m.anchors[None], v0[:, None]))
        assert (mask is None) == (mode == "glance") and (kv.shape[1] == 5) == (mode == "glance")  # 4 registers + ego token


def test_targets_refine_never_hurts():
    d = make_dataset(20, 3)
    assert np.all(d["best_score"] >= d["anchor_score"] - 1e-5)
    assert d["img"].shape == (20, 96, 160, 3)


def test_object_is_rendered():
    rng = np.random.default_rng(0)
    sc = make_scene(rng)
    a = render(sc)
    sc2 = dict(sc, typ=np.full_like(sc["typ"], -1))
    b = render(sc2)
    assert np.abs(a.astype(int) - b.astype(int)).sum() > 0
