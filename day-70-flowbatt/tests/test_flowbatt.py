import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flowbatt import world as W                                   # noqa: E402
from flowbatt.data import Normaliser                              # noqa: E402
from flowbatt.model import DiTBlock, FlowBatt, DiffBattLite, DetTransformer, TrajectoryDiT  # noqa: E402

CFG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml"))
M = CFG["model"]


def test_soh_hits_eol_exactly():
    rng = np.random.default_rng(0)
    for _ in range(50):
        lat = W.sample_latents(rng)
        assert abs(W.soh_curve(lat, np.array([lat["n_eol"]]))[0] - 0.80) < 1e-6
        s = W.soh_curve(lat, W.GRID)
        assert np.all(np.diff(s) <= 1e-9)                         # monotone fade (noise-free)


def test_eol_cycle_interpolation():
    soh = np.linspace(1.0, 0.6, W.L)[None]                        # crosses 0.8 at the middle of the horizon
    assert abs(W.eol_cycle(soh)[0] - W.H_CYC / 2) < W.H_CYC / W.L
    assert W.eol_cycle(np.ones((1, W.L)))[0] == W.H_CYC


def test_cell_shapes_and_prefix():
    c = W.make_cell(np.random.default_rng(1))
    assert c["cap"].shape == (W.EARLY, W.NV) and c["soh"].shape == (W.L,) and c["prefix"].shape == (W.PREFIX,)
    assert np.allclose(c["prefix"], c["soh"][:W.PREFIX])


def test_early_data_is_informative_but_incomplete():
    f = W.make_fleet(300, 3)
    dq = f["cap"][:, 99] - f["cap"][:, 9]
    r = np.corrcoef(np.log(np.var(dq, 1)), np.log(f["n_eol"]))[0, 1]
    assert r < -0.5                                               # bigger Delta-Q variance -> shorter life
    assert abs(np.corrcoef(np.log(np.var(dq, 1)), f["hidden"])[0, 1]) < 0.35  # hidden defect: only weak leak via early fade


def test_normaliser_roundtrip():
    f = W.make_fleet(40, 4)
    n = Normaliser(f["soh"])
    assert np.allclose(n.inv(n.fwd(f["soh"])), f["soh"], atol=1e-5)


def test_dit_block_starts_as_identity():
    blk = DiTBlock(32, 4)
    x, c = torch.randn(2, 16, 32), torch.randn(2, 32)
    assert torch.allclose(blk(x, c), x)                           # adaLN-Zero gates


def test_dit_shapes():
    dit = TrajectoryDiT(256, 16, 32, 2, 4)
    assert dit(torch.randn(3, 256), torch.randn(3, 32)).shape == (3, 256)


def test_flowbatt_loss_and_sample_shapes():
    torch.manual_seed(0)
    m = FlowBatt(M, 256, 10)
    cap, pre, x1 = torch.randn(4, 100, 100), torch.ones(4, 10), torch.randn(4, 256)
    loss = m.loss(x1, cap, pre)
    loss.backward()
    assert torch.isfinite(loss)
    s, path = m.eval().sample(cap[:2], pre[:2], 5, steps=4, return_path=True)
    assert s.shape == (2, 5, 256) and len(path) == 5


def test_flow_sampler_is_exact_for_constant_velocity():
    """With v = x1 - x0 known exactly, one Euler step from x0 lands on x1 (the linear path is straight)."""
    m = FlowBatt(M, 256, 10).eval()
    target = torch.linspace(-1, 1, 256)

    class V(torch.nn.Module):
        def forward(self, x, c):                                  # v(x_t, t) = (x1 - x_t) / (1 - t) along the path
            return (target - x) / (1 - self.t)
    v = V()
    m.velocity = lambda x, t, c: (v.__setattr__("t", t[:, None]) or v(x, c))
    out = m.sample(torch.randn(1, 100, 100), torch.ones(1, 10), 3, steps=1)
    assert torch.allclose(out[0], target.expand(3, -1), atol=1e-5)


def test_baselines_run():
    cap, pre, x1 = torch.randn(2, 100, 100), torch.ones(2, 10), torch.randn(2, 256)
    for cls in (DiffBattLite, DetTransformer):
        m = cls(M, 256, 10)
        assert torch.isfinite(m.loss(x1, cap, pre))
    d = DiffBattLite(M, 256, 10, T=5).eval()
    assert d.sample(cap, pre, 2).shape == (2, 2, 256)
