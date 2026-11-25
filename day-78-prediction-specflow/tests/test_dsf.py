import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsf.data import DT, VEH, agent_frame, local_to_world, make_dataset  # noqa: E402
from dsf.model import VARIANTS, Kinematics, build, dct_matrix  # noqa: E402

D = make_dataset(40, 3)
F = agent_frame(D["pos"], D["clean"], D["mask"], D["kind"])


def batch(n=8):
    keys = ["hist", "fut", "v0", "rel", "mask", "kind"]
    return {k: torch.as_tensor(F[k][:n]) for k in keys}


def test_dct_orthonormal():
    C = dct_matrix(12)
    assert torch.allclose(C @ C.T, torch.eye(12), atol=1e-5)


def test_integration_inverts_differencing():
    k = Kinematics()
    p = torch.randn(5, 12, 2).cumsum(1)
    v0 = torch.randn(5, 2)
    v, a = k.differentiate(p, v0)
    v2, p2 = k.integrate(k.spec_to_acc(k.acc_to_spec(a)), v0)
    assert torch.allclose(p, p2, atol=1e-4) and torch.allclose(v, v2, atol=1e-4)


def test_agent_frame_origin_and_heading():
    m = D["mask"]
    assert np.allclose(F["hist"][..., -1, :2][m], 0, atol=1e-5)
    moving = m & (np.linalg.norm(F["v0"], axis=-1) > 3)
    assert np.abs(F["v0"][moving][:, 1]).mean() < 0.3 * np.abs(F["v0"][moving][:, 0]).mean()


def test_local_to_world_roundtrip():
    s, a = np.argwhere(D["mask"])[3]
    w = local_to_world(F["fut_clean"][s, a], F["origin"][s, a], F["psi"][s, a])
    assert np.allclose(w, D["clean"][s, a, 8:], atol=1e-3)


def test_simulator_accel_limits():
    acc = np.diff(D["clean"], 2, axis=2) / DT ** 2
    veh = D["mask"] & (D["kind"] == VEH)
    assert np.percentile(np.linalg.norm(acc[veh], axis=-1), 99) < 6.0


def test_all_variants_train_step():
    b = batch()
    for nm in VARIANTS:
        m = build(nm)
        m.fit_norm(b["fut"], b["v0"], b["mask"])
        loss, _ = m.loss(b)
        loss.backward()
        assert torch.isfinite(loss)


def test_fit_norm_unit_variance():
    m = build("specflow")
    fut, v0, mk = (torch.as_tensor(F[k]) for k in ("fut", "v0", "mask"))
    m.fit_norm(fut, v0, mk)
    x = ((m.raw_state(fut, v0) - m.mu) / m.sd)[mk]
    assert abs(float((x ** 2).mean()) - 1) < 0.05


def test_sample_shapes_and_padding():
    b = batch(4)
    m = build("specflow")
    o = m.sample(b, H=7, steps=2)
    assert o["p"].shape == (4, 7, 6, 12, 2)
    assert torch.isfinite(o["p"]).all()


def test_stratified_endpoints_follow_confidence():
    b = batch(2)
    m = build("specflow")
    with torch.no_grad():
        m.eph.mlp[-1].bias.zero_(); m.eph.mlp[-1].weight.zero_()
        m.eph.mlp[-1].bias.view(6, 3)[:, 2] = torch.log(torch.tensor([0.5, 0.25, 0.25, 1e-6, 1e-6, 1e-6]))
        m.eph.mlp[-1].bias.view(6, 3)[:, 0] = torch.arange(6.0)            # candidate k has x = 20k m
    o = m.sample(b, H=20, steps=1, ep_mode="sampled")
    ks = (o["cond"][0, :, 0, 0] / 20).round().long()
    assert torch.bincount(ks, minlength=6).tolist()[:3] == [10, 5, 5]
