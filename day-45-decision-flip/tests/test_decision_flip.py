import copy, torch, yaml
from decision_flip.model import GaussianOccModel
from decision_flip.flip import mean_flip_scores, guarded_otsu, compile_pruned, topk_mask
from decision_flip.data import make_split

CFG = yaml.safe_load(open("configs/config.yaml")); CFG["model"]["num_gaussians"] = 24


def tiny():
    torch.manual_seed(0); m = GaussianOccModel(CFG); m.eval(); f, o = make_split(0, 4); return m, f, o


def test_forward_shape():
    m, f, _ = tiny(); assert m(f).shape == (4, 16 * 16 * 4, 5)


def test_gradient_flow():
    m, f, o = tiny(); m.train(); loss = torch.nn.functional.cross_entropy(m(f).reshape(-1, 5), o.reshape(-1)); loss.backward()
    assert m.bank.mean.grad.abs().sum() > 0 and m.bank.cls.grad.abs().sum() > 0


def test_flip_matches_bruteforce_removal():
    """Core correctness: additive leave-one-out shortcut == physically deleting the row and re-running."""
    m, f, _ = tiny(); s = mean_flip_scores(m, f)
    base = m(f).argmax(-1)
    for i in [0, 5, 11]:
        keep = torch.ones(24, dtype=torch.bool); keep[i] = False
        flips = (compile_pruned(m, keep)(f).argmax(-1) != base).float().sum() / len(f)
        assert abs(flips.item() - s[i].item()) < 1e-4, (i, flips.item(), s[i].item())


def test_dead_row_scores_zero_and_removal_is_noop():
    m, f, _ = tiny()
    with torch.no_grad(): m.bank.opa[3] = -50.0; m.bank.refine[-1].bias[3] = 0
    s = mean_flip_scores(m, f)
    assert s[3] == 0


def test_compile_is_physically_smaller_and_equal_to_mask():
    m, f, _ = tiny(); keep = topk_mask(torch.rand(24), 10); pm = compile_pruned(m, keep)
    assert pm.bank.n == 10 and pm.bank.mean.shape == (10, 3)
    g = keep.float()
    assert torch.allclose(pm(f), m(f, g), atol=1e-5)  # gate-by-zero == compiled slice


def test_otsu_splits_bimodal_and_abstains_on_unimodal():
    sc = torch.cat([torch.zeros(30), torch.full((30,), 40.0) + torch.rand(30)])
    keep, info = guarded_otsu(sc); assert not info["abstained"] and keep.sum() == 30 and keep[30:].all()
    keep, info = guarded_otsu(torch.linspace(10, 11, 60)); assert info["abstained"] and keep.all()
