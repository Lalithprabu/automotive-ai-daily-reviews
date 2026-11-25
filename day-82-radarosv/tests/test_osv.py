import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osv.model import K_CAND, PCG, RadarOSVNet, VelocityReadout, losses  # noqa: E402
from osv.radar import AI, DI, DS, RI, RS, AS, VI, _cube, render, wrap  # noqa: E402
from osv.scene import FOV_MASK, GR, GTH, G, R_MAX, make_scene, rasterize  # noqa: E402


def test_wrap():
    v = np.array([0.0, 2.4, 2.6, -2.6, 7.0, -12.0])
    w = wrap(v)
    assert np.all((w >= -VI) & (w < VI))
    assert np.allclose(np.round((v - w) / (2 * VI)) * 2 * VI + w, v)
    assert np.isclose(w[2], -2.4) and np.isclose(w[4], 2.0) and np.isclose(w[5], -2.0)


def test_static_cells_have_minus_ego_cos():
    s = make_scene(np.random.default_rng(3), v_ego=6.0)
    gt = rasterize(s)
    st = (gt["sem"] == 1)
    assert st.any()
    assert np.allclose(gt["vrel"][st], -6.0 * np.cos(GTH[st]), atol=1e-4)
    assert np.allclose(gt["vgr"][st], 0.0)


def test_imaging_folds_singlechip_does_not():
    rng = np.random.default_rng(0)
    ret = np.array([[10.0, 0.0, 4.0, 50.0]])                                # one strong return, +4 m/s
    img = _cube(ret, RI, AI, DI, R_MAX / RI, VI, (0.1, 0.1, 0.1), 1e-6, rng, 0, 0, fold=True).astype(np.float32)
    sc = _cube(ret, RS, AS, DS, R_MAX / RS, 24.0, (0.1, 0.1, 0.1), 1e-6, rng, 0, 0, fold=False).astype(np.float32)
    v_img = (img.max((1, 2)).argmax() + 0.5) / DI * 2 * VI - VI
    v_sc = (sc.max((1, 2)).argmax() + 0.5) / DS * 48 - 24
    assert abs(v_img - (-1.0)) < 0.35                                         # 4 m/s aliased to -1 m/s
    assert abs(v_sc - 4.0) < 0.8


def test_render_shapes():
    rng = np.random.default_rng(1)
    img, sc = render(make_scene(rng), rng)
    assert img.shape == (DI, RI, AI) and sc.shape == (DS, RS, AS)
    assert np.isfinite(img.astype(np.float32)).all()


def test_pcg_fixed_sampling_maps_range():
    pcg = PCG(1, deformable=False)
    rr = (torch.arange(RI, dtype=torch.float32) + 0.5) * R_MAX / RI
    f = rr[None, None, :, None].expand(1, 1, RI, AI).contiguous()           # polar map whose value is the range
    out = pcg(f)[0, 0].numpy()
    # zero padding attenuates the outermost half azimuth bin, so test one bin inside the field of view
    m = FOV_MASK & (GR < R_MAX - 0.5) & (np.abs(GTH) < np.deg2rad(60 - 1.875))
    assert np.abs(out[m] - GR[m]).max() < 0.25


def test_pcg_deformable_starts_as_fixed():
    f = torch.randn(2, 4, RI, AI)
    a, b = PCG(4, deformable=False), PCG(4, deformable=True)
    assert torch.allclose(b(f), 2 * a(f), atol=1e-5)                        # zero offsets, uniform weights


def test_velocity_decode_roundtrip():
    v = torch.tensor([[-17.3, -2.2, 0.4, 3.1, 9.9]])[..., None]               # (1, 5, 1)
    w = (v + VI) % (2 * VI) - VI
    phi = math.pi * w / VI
    ph = torch.stack([torch.cos(phi), torch.sin(phi)], 1)                     # (1, 2, 5, 1)
    k = torch.round((v - w) / (2 * VI)).long() + K_CAND
    logit = torch.nn.functional.one_hot(k, 2 * K_CAND + 1).permute(0, 3, 1, 2).float() * 10
    dec, _, _ = VelocityReadout.decode(ph, logit)
    assert torch.allclose(dec, v, atol=1e-4)


def test_losses_and_backward():
    m = RadarOSVNet(use_sc=True, ego="estimated")
    img, sc = torch.randn(2, DI, RI, AI), torch.randn(2, DS, RS, AS)
    sem = torch.randint(0, 6, (2, G, G))
    vrel = torch.randn(2, G, G) * 8
    out = m(img, sc)
    loss, parts = losses(out, sem, vrel, torch.tensor([3.0, 6.0]), torch.from_numpy(FOV_MASK), torch.ones(6))
    loss.backward()
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in parts.values())


def test_external_ego_is_used_and_deterministic():
    torch.manual_seed(0)
    m = RadarOSVNet(use_sc=False, ego="external").eval()
    img, sc = torch.randn(1, DI, RI, AI), torch.randn(1, DS, RS, AS)
    with torch.no_grad():
        o1 = m(img, sc, torch.tensor([2.0]))
        o2 = m(img, sc, torch.tensor([2.0]))
    assert torch.allclose(o1["v_used"], torch.tensor([2.0])) and torch.allclose(o1["sem"], o2["sem"])
