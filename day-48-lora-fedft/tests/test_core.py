import copy, sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.lora_fedft import (LoRALinear, TrajTransformer, apply_lora, fedavg, load_trainable_state, payload_bytes,
                            payload_numel, predict, run_federated, trainable_state, trajectory_metrics)
from src.lora_fedft.data import (HOR, OBS, SCALE, canonicalize, make_clients, make_pretrain_set, make_split,
                                 to_canonical_future, to_world)

torch.manual_seed(0)
CFG = dict(d=32, heads=2, layers=1, ff=64)


def test_lora_is_identity_at_init():
    base = TrajTransformer(**CFG).eval()
    x = torch.randn(4, OBS, 4)
    ref = base(x)
    lora = apply_lora(copy.deepcopy(base), r=4, alpha=8.0).eval()
    assert torch.allclose(lora(x), ref, atol=1e-6)  # B = 0 -> exactly the pretrained model


def test_only_adapters_trainable_and_all_linears_wrapped():
    m = apply_lora(TrajTransformer(**CFG), r=4)
    n_lin = sum(isinstance(c, LoRALinear) for c in m.modules())
    assert n_lin == 4 + 2 + 1 + 1  # q,k,v,o + fc1,fc2 per block + inp + head
    assert all(("lora_" in n) == p.requires_grad for n, p in m.named_parameters())
    assert set(trainable_state(m)) == {n for n, _ in m.named_parameters() if "lora_" in n}


def test_gradients_reach_B_but_base_stays_frozen():
    m = apply_lora(TrajTransformer(**CFG), r=4)
    before = {n: p.clone() for n, p in m.named_parameters() if not p.requires_grad}
    loss = m(torch.randn(8, OBS, 4)).pow(2).mean()
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if n.endswith("lora_B"))
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1); opt.step()
    assert all(torch.equal(before[n], p) for n, p in m.named_parameters() if n in before)


def test_merged_weight_matches_forward():
    lin = torch.nn.Linear(16, 8)
    l = LoRALinear(lin, r=4, alpha=8.0)
    torch.nn.init.normal_(l.lora_B, std=0.1)
    x = torch.randn(5, 16)
    assert torch.allclose(l(x), x @ l.merged_weight().t() + lin.bias, atol=1e-5)


def test_payload_much_smaller_than_full_when_wide():
    wide = dict(d=128, heads=4, layers=2, ff=256)
    full = payload_bytes(TrajTransformer(**wide))
    lora = payload_bytes(apply_lora(TrajTransformer(**wide), r=4))
    assert lora < 0.1 * full


def test_fedavg_weighted_mean():
    s1, s2 = {"a": torch.tensor([0.0, 0.0])}, {"a": torch.tensor([4.0, 8.0])}
    out = fedavg([s1, s2], [1, 3])
    assert torch.allclose(out["a"], torch.tensor([3.0, 6.0]))


def test_canonicalize_roundtrip_and_orientation():
    d = make_split(32, {"curving": 1.0}, seed=1)
    feats, R, origin = canonicalize(d["world"][:, :OBS])
    # last motion points along +x in canonical frame
    rel = feats[..., :2] * SCALE  # positions relative to last point, canonical frame
    assert (rel[:, -3, 0] < 0).all() and np.abs(rel[:, -3, 1]).max() < 1e-3  # last 2-step motion is along +x
    fut_c = to_canonical_future(d["world"][:, OBS:], R, origin)
    assert np.allclose(to_world(fut_c, R, origin), d["world"][:, OBS:], atol=1e-4)


def test_styles_differ_from_straight_extrapolation():
    """The base domain (straight) must be genuinely wrong for the other styles."""
    errs = {}
    for s in ("straight", "curving", "stop_go", "crossing"):
        d = make_split(200, {s: 1.0}, seed=3)
        v = d["x"][:, -1, 2:4]                                   # last canonical velocity
        cv = v[:, None] * torch.arange(1, HOR + 1)[None, :, None]
        errs[s] = ((cv - d["y"]).norm(dim=-1) * SCALE).mean().item()
    assert errs["straight"] < 0.5 and min(errs["curving"], errs["stop_go"], errs["crossing"]) > 3 * errs["straight"]


def test_overfit_tiny_batch():
    d = make_split(16, {"curving": 1.0}, seed=5)
    m = TrajTransformer(**CFG)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(300):
        loss = torch.nn.functional.smooth_l1_loss(m(d["x"]), d["y"], beta=0.05)
        first = first or loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.2 * first


def test_core_claim_lora_fed_beats_zero_shot_with_tiny_payload():
    """Pretrain on straight walking, adapt with LoRA over federated rounds: error must drop."""
    base = TrajTransformer(**CFG)
    pre = make_pretrain_set(600, 0)
    opt = torch.optim.Adam(base.parameters(), lr=3e-3)
    for _ in range(40):
        loss = torch.nn.functional.smooth_l1_loss(base(pre["x"]), pre["y"], beta=0.05)
        opt.zero_grad(); loss.backward(); opt.step()
    clients, val = make_clients(8, 64, 32, 0.7, seed=2)
    zs = trajectory_metrics(predict(base, val["x"]), val["y"])["ADE"]
    m = apply_lora(copy.deepcopy(base), r=4, alpha=8.0)
    h = run_federated(m, clients, val, rounds=8, local_epochs=2, lr=3e-3, batch_size=32, seed=0)
    assert h["ADE"][-1] < 0.9 * zs
    assert h["uplink_bytes"][-1] == 8 * 8 * payload_bytes(m)  # 8 rounds x 8 vehicles x per-upload payload
