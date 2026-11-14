"""Unit tests: physics/metrics consistency, token semantics, exact zero-token no-op, guard rule, shapes."""
import numpy as np
import torch

from foresee import data as D
from foresee import losses as L
from foresee import tokens as TK
from foresee import world as W
from foresee.model import BaseGenerator, ForeseeDrive, TokenPool, TokenPredictor, ZeroGatedAdapter, inject
from evaluate import pdm_t

torch.manual_seed(0)
D_SMALL = D.to_torch(D.build(64, seed=123))
ANCH = torch.from_numpy(np.stack([W.rollout_profile(9.0, [a], [a], [lc])[1][0]
                                  for a in (-3, 0, 1) for lc in (0, 1)])).float()


def test_expert_is_safe():
    rng = np.random.default_rng(0)
    for _ in range(100):
        sc = W.sample_scene(rng)
        assert W.pdm_score(sc["expert"], sc)["nc"] == 1.0


def test_torch_pdm_matches_numpy():
    b = D_SMALL
    traj = b["expert"][:, None] + torch.randn(64, 1, W.H, 2) * 0.8
    s = pdm_t(traj, b)
    for i in range(16):
        ref = W.pdm_score(traj[i, 0].numpy(), D.scene_dict(b, i))
        assert abs(ref["pdms"] - s["pdms"][i, 0].item()) < 1e-4, (i, ref, s["pdms"][i, 0])


def test_lane_change_profile():
    _, c = W.rollout_profile(10.0, [0.0], [0.0], [1.0])
    assert abs(c[0, -1, 1] - W.LEFT_Y) < 1e-6 and abs(c[0, -1, 0] - 40.0) < 1e-6


def test_braking_lead_has_risk_token():
    rng = np.random.default_rng(3)
    found = False
    for _ in range(400):
        sc = W.sample_scene(rng)
        rank = TK.rank_actors(sc["hist"], sc["valid"])
        E, C, R = TK.scene_tokens(sc, rank)
        for slot, j in enumerate(rank):
            if j >= 0 and sc["event"][j] == 1 and sc["fut"][j, -1, 0] < sc["hist"][j, -1, 0] + sc["hist"][j, -1, 2] * 4 - 3:
                assert R[slot] == 1 and E[slot, 0] < -0.3
                found = True
    assert found


def test_zero_token_exact_noop():
    base = BaseGenerator(ANCH).eval()
    fd = ForeseeDrive(base).eval()
    for p in fd.alphas:                                       # make the adapters non-trivial
        torch.nn.init.normal_(p, std=1.0)
    b = D.batch(D_SMALL, torch.arange(8))
    v = torch.zeros(8, TK.N_SCENE_Q + base.K, 5)
    with torch.no_grad():
        o = fd(b, v)
        assert torch.equal(o["traj"], o["base_traj"]) and torch.equal(o["logit"], o["base_logit"])
        v[:, 3, 0] = 1.0                                      # one live token -> output must change
        o2 = fd(b, v)
    assert (o2["traj"] - o2["base_traj"]).abs().max() > 0


def test_adapter_and_pool_zero():
    a, pool = ZeroGatedAdapter(16), TokenPool(10, 16)
    assert torch.equal(a(torch.randn(2, 5, 16), torch.zeros(2, 16)), torch.zeros(2, 5, 16))
    assert torch.equal(pool(torch.zeros(2, 10, 5), torch.randn(2, 16)), torch.zeros(2, 16))


def test_inject_suppresses_low_confidence():
    e, c, r = torch.ones(1, 3, 4), torch.tensor([[-5.0, 0.0, 5.0]]), torch.zeros(1, 3)
    v = inject(e, c, r, from_logits=True)
    assert v[0, 0].abs().sum() == 0 and v[0, 1].abs().sum() > 0 and v[0, 2].abs().sum() > 0


def test_guard_reverts_safe_to_unsafe():
    b = D.batch(D_SMALL, torch.arange(64))
    tb = b["expert"][:, None].repeat(1, 2, 1, 1)
    cv = L.cv_projection(b)
    tf = tb.clone()
    tf[:, 1] = cv[:, 0]                                       # drive straight into actor 0's projected path
    out, bad = L.apply_guard(tf, tb, b)
    mb, _ = L.clearance_t(tb, cv, b)
    expect = (mb[:, 1] > 1.0) & b["valid"][:, 0]
    assert bad[:, 0].sum() == 0
    assert bad[expect, 1].all()
    assert torch.equal(out[bad], tb[bad])


def test_shapes():
    base = BaseGenerator(ANCH)
    pred = TokenPredictor(base.K)
    b = D.batch(D_SMALL, torch.arange(4))
    traj, logit, z, mem = base(b)
    assert traj.shape == (4, base.K, W.H, 2) and logit.shape == (4, base.K)
    e, c, r = pred(b, traj)
    Q = TK.N_SCENE_Q + base.K
    assert e.shape == (4, Q, TK.E_DIM) and c.shape == (4, Q) and r.shape == (4, Q)
    tE, tC, tR = L.teacher_tokens(traj, b)
    assert tE.shape == e.shape
    loss = L.distill_loss((e, c, r), (tE, tC, tR))
    loss.backward()
