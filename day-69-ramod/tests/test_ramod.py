"""Unit tests: physics of the world, the ego estimator / suppression geometry, shapes, losses, decoding."""
import numpy as np
import torch

from ramod import world as W
from ramod.data import RadarSeq, geom, make_split
from ramod.losses import decode, ramod_loss
from ramod.model import EgoMotionHough, RaMOD, StaticDopplerSuppression, wrap_t


def test_wrap_matches_numpy():
    v = torch.linspace(-15, 15, 301)
    assert np.allclose(wrap_t(v, 4.0).numpy(), W.wrap_v(v.numpy()), atol=1e-5)
    assert (wrap_t(v, 4.0) >= -4).all() and (wrap_t(v, 4.0) < 4).all()


def test_static_radial_velocity_ignores_yaw_rate():
    """v_r(static) = -v_e cos(theta), independent of yaw rate, for a radar at the rotation centre."""
    rng = np.random.default_rng(1)
    sim = W.simulate_episode(rng, 5)
    st = W.frame_state(sim, 2)
    n = len(sim["static_xy"])
    p, v = st["p"][:n], st["v"][:n]
    r = np.linalg.norm(p, axis=1)
    vr = (p * v).sum(1) / r
    assert np.allclose(vr, -st["v_ego"] * p[:, 0] / r, atol=1e-6)


def test_wrap_labels_consistent():
    rng = np.random.default_rng(3)
    for _ in range(5):
        sim = W.simulate_episode(rng, 10)
        for k in range(10):
            for o in W.labels(W.frame_state(sim, k)):
                assert abs(o["v_meas"] + 2 * W.V_MAX * o["k"] - o["v_rad"]) < 1e-6
                assert -W.V_MAX <= o["v_meas"] < W.V_MAX


def test_suppression_mask_peaks_on_static_curve():
    g = geom()
    sup = StaticDopplerSuppression(torch.tensor(g["s_centers"]), torch.tensor(g["v_centers"]), g["vmax"])
    v_e = torch.tensor([6.0])                                 # > v_max -> wrapped clutter
    _, M = sup(torch.zeros(1, 40, 40, 32), v_e)
    v_s = W.wrap_v(-6.0 * np.sqrt(1 - g["s_centers"] ** 2))
    d_peak = M[0].argmax(-1).numpy()
    d_true = np.clip(((v_s + W.V_MAX) / W.V_RES).astype(int), 0, 31)
    assert (np.abs(d_peak - d_true) <= 1).mean() > 0.95


def test_hough_votes_peak_at_true_speed_with_oracle_evidence():
    """If the evidence map is exactly the static curve, the vote argmax must be the true speed (even > v_max)."""
    g = geom()
    est = EgoMotionHough(torch.tensor(g["s_centers"], dtype=torch.float32), g["vmax"])
    for v_true in (1.0, 3.5, 6.2, 7.8):
        v_s = W.wrap_v(-v_true * np.sqrt(1 - g["s_centers"] ** 2))
        E = np.zeros((40, 32), np.float32)
        E[np.arange(40), np.clip(((v_s + W.V_MAX) / W.V_RES).astype(int), 0, 31)] = 1.0
        Et = torch.from_numpy(E)[None, None]
        v_s_grid = wrap_t(-est.v_grid[:, None] * est.cos_a[None], est.vmax)
        grid = torch.stack([v_s_grid / est.vmax, ((torch.arange(40) + 0.5) / 40 * 2 - 1)[None].expand(86, -1)], -1)
        votes = torch.nn.functional.grid_sample(Et, grid[None], align_corners=False)[0, 0].mean(-1)
        assert abs(est.v_grid[votes.argmax()].item() - v_true) < 0.35


def _tiny_split():
    return make_split(2, seed=7, steps=8)


def test_forward_shapes_all_variants():
    split = _tiny_split()
    for kw in (dict(T=3), dict(T=3, suppress=False), dict(T=1, suppress=False, wrap_head=False)):
        m = RaMOD(geom(), c=16, **kw)
        ds = RadarSeq(split, T=m.T)
        b = [torch.stack([ds[i][j] for i in range(3)]) for j in range(5)]
        out = m(b[0])
        assert out["heat"].shape == (3, 3, 40, 40)
        assert out["v_raw"].shape == (3, 40, 40) and out["v_raw"].abs().max() <= 4.0
        for ab in (dict(), dict(ego_sup=False), dict(wrap_sup=False)):
            loss, parts = ramod_loss(out, *b[1:], m.vmax, **ab)
            assert torch.isfinite(loss)
            assert ("wrap" in parts) == (kw.get("wrap_head", True) and ab.get("wrap_sup", True))
        loss.backward()
        if kw.get("suppress", True):
            assert out["v_e"].shape == (3, 3) and out["mask"].shape == (3, 3, 40, 32)


def test_frame_diff_channels():
    m = RaMOD(geom(), c=8, T=3)
    assert m.backbone.enc.dop[0].in_channels == 3 * 32          # suppressed + mask + difference
    m1 = RaMOD(geom(), c=8, T=1, suppress=False, wrap_head=False)
    assert m1.backbone.enc.dop[0].in_channels == 32


def test_decode_uses_unwrapped_hypothesis_above_half():
    z = lambda: torch.zeros(1, 40, 40)
    out = dict(heat=torch.full((1, 3, 40, 40), -9.0), size=torch.zeros(1, 2, 40, 40), off=torch.zeros(1, 2, 40, 40),
               v_raw=z(), f_raw=z(), v_unw=z(), f_unw=z(), wrap=z())
    out["heat"][0, 2, 10, 20] = 5.0
    out["v_raw"][0, 10, 20], out["v_unw"][0, 10, 20] = 3.0, -5.0
    out["wrap"][0, 10, 20] = 3.0                               # p_wrap ~ 0.95 -> unwrapped
    d = decode(out, 4.0)[0]
    assert len(d) == 1 and d[0]["cls"] == 2 and d[0]["wrapped"] and d[0]["v_rad"] == -5.0
    out["wrap"][0, 10, 20] = -3.0                              # p_wrap ~ 0.05 -> raw
    d = decode(out, 4.0)[0]
    assert not d[0]["wrapped"] and abs(d[0]["v_rad"] - 3.0) < 1e-6


def test_targets_center_mask():
    split = _tiny_split()
    ds = RadarSeq(split, T=3)
    for i in range(len(ds)):
        _, heat, reg, mask, _, _ = ds[i]
        assert int(mask.sum()) <= len(ds.objects(i))
        assert heat.max() <= 1.0
