import numpy as np, torch
from src.cwt import MorletCWT, ScaleCompression
from src.model import CWTRULNet
from src.synthetic import make_cells, split_idx, pick_cycles


def test_cwt_peak_scale():                       # a sinusoid should peak at the matching scale
    T = 128; cwt = MorletCWT(T); t = torch.arange(T)
    f = 8 / T; x = torch.sin(2 * np.pi * f * t)[None, None].float()
    w = cwt(x)[0, 0, :, 20:-20].mean(-1); s_exp = (6.0 + (2 + 36) ** 0.5 / 1) / (2 * np.pi * f)
    assert abs(float(cwt.scales[w.argmax()]) - 6.0 / (2 * np.pi * f)) / (6.0 / (2 * np.pi * f)) < 0.2


def test_shapes_and_compression():
    m = CWTRULNet(); x = torch.randn(5, 3, 3, 128)
    assert m.scalogram(x).shape == (15, 3, 8, 128) and m(x).shape == (5,) and m(x[:, :1]).shape == (5,)
    assert ScaleCompression(4)(torch.rand(2, 3, 32, 16)).shape == (2, 3, 8, 16)


def test_backward_finite():
    m = CWTRULNet(); loss = m(torch.randn(4, 2, 3, 128)).pow(2).mean(); loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_split_by_cell_and_cycles():
    a, b, c = split_idx(100, .15, .15); assert not (set(a) & set(b) | set(a) & set(c) | set(b) & set(c))
    assert list(pick_cycles(1)) == [0] and list(pick_cycles(50)) == [0, 16, 33, 49]


def test_signal_has_life_info():                 # linear probe on single-cycle features must beat mean baseline
    X, life, _ = make_cells(400, 5); y = np.log(life); F = X[:, 0].reshape(400, -1)[:, ::4]
    F = (F - F.mean(0)) / (F.std(0) + 1e-6); tr, te = slice(0, 300), slice(300, 400)
    w = np.linalg.solve(F[tr].T @ F[tr] + 50 * np.eye(F.shape[1]), F[tr].T @ (y[tr] - y[tr].mean()))
    assert ((F[te] @ w + y[tr].mean() - y[te]) ** 2).mean() < 0.9 * ((y[tr].mean() - y[te]) ** 2).mean()
