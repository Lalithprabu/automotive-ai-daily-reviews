import copy, torch, pytest
from src.model import TrajTransformer
from src.lora import inject_lora, freeze_all_but_lora, trainable_state, load_state, n_params, LoRALinear
from src.federated import fedavg, run_federated
from src.data import make_client, make_target_val

def mk(): torch.manual_seed(0); return TrajTransformer(8, 12, 32, 4, 2)

def test_lora_identity_at_init():
    m = mk(); x = torch.randn(5, 8, 2); y0 = m(x)
    inject_lora(m, {"qkv", "proj", "ff1", "ff2"}, 4, 8.0)
    assert torch.allclose(m(x), y0, atol=1e-6)

def test_only_adapters_and_head_trainable_and_small():
    m = mk(); total = sum(p.numel() for p in m.parameters())
    assert inject_lora(m, {"qkv", "proj", "ff1", "ff2"}, 4, 8.0) == 8
    freeze_all_but_lora(m); st = trainable_state(m)
    assert all(k.endswith((".A", ".B")) or "head" in k for k in st)
    assert n_params(st) < 0.5 * total

def test_frozen_base_unchanged_after_training():
    m = mk(); inject_lora(m, {"qkv", "proj", "ff1", "ff2"}, 4, 8.0); freeze_all_but_lora(m)
    before = copy.deepcopy(m.blocks[0].qkv.base.weight)
    o, f, _ = make_client(0, 4, 64, 8, 12, 0)
    from src.federated import local_train; local_train(m, (o, f), 2, 0.01, 32, 0)
    assert torch.equal(before, m.blocks[0].qkv.base.weight)
    assert m.blocks[0].qkv.B.abs().sum() > 0   # adapters did move

def test_fedavg_weighted_mean():
    a, b = {"w": torch.tensor([0.0])}, {"w": torch.tensor([4.0])}
    assert fedavg([a, b], [1, 3])["w"].item() == pytest.approx(3.0)

def test_comm_accounting_lora_less_than_full():
    base = mk(); cl = [make_client(i, 2, 32, 8, 12, 0)[:2] for i in range(2)]
    v = make_target_val(32, 8, 12, 2, 0)[:2]
    full = copy.deepcopy(base)
    lg = run_federated(full, cl, v, 1, 1, 1e-3, 32, 32, 1000)
    lo = copy.deepcopy(base); inject_lora(lo, {"qkv", "proj", "ff1", "ff2"}, 4, 8.0); freeze_all_but_lora(lo)
    lg2 = run_federated(lo, cl, v, 1, 1, 1e-3, 32, 32, 1000)
    assert lg2[0]["bits"] < lg[0]["bits"]
    assert lg[0]["bits"] == 2 * 2 * lg[0]["payload_params"] * 32

def test_federated_training_reduces_error():
    base = mk(); cl = [make_client(i, 2, 128, 8, 12, 0)[:2] for i in range(2)]
    v = make_target_val(64, 8, 12, 2, 0)[:2]
    from src.federated import evaluate
    e0 = evaluate(base, *v)[0]; lg = run_federated(base, cl, v, 3, 3, 3e-3, 32, 32, 1000)
    assert lg[-1]["ade"] < e0
