import math, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from fastbev.synthetic import G, W_IMG, make_dataset, azimuth_map, heatmap
from fastbev.models import FastBEVFusion, UniformDepthProjector, FUSIONS, pos_enc_2d
from fastbev.metrics import decode, match


def test_shapes_all_fusions():
    L, I, H, _ = make_dataset(4, 0)
    assert L.shape == (4, 2, G, G) and I.shape == (4, 3, W_IMG) and H.shape == (4, 1, G, G)
    for f in FUSIONS:
        assert FastBEVFusion(f)(L, I).shape == (4, 1, G, G)


def test_projector_copies_feature_along_ray():
    """Cells with the same azimuth must receive the SAME image feature (uniform depth distribution)."""
    feat = torch.randn(1, 5, W_IMG); out = UniformDepthProjector()(feat)[0]
    az = azimuth_map(); i0, j0 = 10, 16; a0 = az[i0, j0]
    same = np.argwhere(np.abs(az - a0) < 0.004)
    assert len(same) >= 2
    for i, j in same:
        assert torch.allclose(out[:, i, j], out[:, i0, j0], atol=0.2 * feat.abs().max())   # near-identical along the ray


def test_projector_ray_exact_for_constant_ray():
    feat = torch.zeros(1, 1, W_IMG); feat[0, 0, W_IMG // 2] = 1.0
    out = UniformDepthProjector()(feat)[0, 0]
    assert out.max() > 0.5 and (out > 0.5).sum() >= 8          # a bright column lights a whole ray of BEV cells


def test_yaw_error_moves_the_ray():
    feat = torch.randn(1, 3, W_IMG); p = UniformDepthProjector()
    assert not torch.allclose(p(feat), p(feat, torch.tensor([math.radians(5)])))
    assert torch.allclose(p(feat), p(feat, torch.tensor([0.0])), atol=1e-6)


def test_cross_attention_rows_are_distributions():
    L, I, _, _ = make_dataset(2, 1); m = FastBEVFusion("xattn").eval()
    m(L, I, keep_attn=True); a = m.fuse.last_attn
    assert a.shape == (2, 256, 256) and torch.allclose(a.sum(-1), torch.ones(2, 256), atol=1e-4)


def test_gradients_reach_both_branches():
    L, I, H, _ = make_dataset(4, 2); m = FastBEVFusion("xattn"); m(L, I).sum().backward()
    assert m.lidar_enc[0].weight.grad.abs().sum() > 0 and m.img_enc[0].weight.grad.abs().sum() > 0


def test_decode_and_match():
    objs = np.array([[20.25, 0.75, .5], [39.75, 9.75, .5]])   # cell centres (no ties)
    hm = torch.tensor(heatmap(objs))[None]; logits = torch.logit(hm.clamp(1e-3, 1 - 1e-3))
    dets = decode(logits, 0.5)[0]; r, miss = match(dets, objs)
    assert len(dets) == 2 and all(t for _, t in r) and miss == []


def test_lidar_sparser_with_range():
    rng = np.random.default_rng(0); from fastbev.synthetic import render
    near = np.mean([render(np.array([[8.0, 0, .5]]), rng, n_clutter=0, n_posters=0, n_poles=0)[0][0].sum() for _ in range(50)])
    far = np.mean([render(np.array([[44.0, 0, .5]]), rng, n_clutter=0, n_posters=0, n_poles=0)[0][0].sum() for _ in range(50)])
    assert near > 3 * far
