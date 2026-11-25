import numpy as np
import torch

from twm.data import BEAM_ROWS, H, LIDAR, T_ALL, W, WEATHERS, make_clip, make_dataset, to_tiles
from twm.model import TILE_W, VideoWorldModel, tile_loss


def test_dataset_shapes_and_weathers():
    d = make_dataset(2, 0, keep_boxes=True)
    assert d["rgb"].shape == (12, T_ALL, H, W, 3)
    assert d["depth"].shape == (12, T_ALL, H, W)
    assert sorted(set(d["weather"].tolist())) == list(range(len(WEATHERS)))


def test_lidar_only_on_beam_rows_and_within_range():
    rng = np.random.default_rng(0)
    for wx in range(6):
        c = make_clip(rng, wx)
        rows = np.nonzero(c["valid"].any((0, 2)))[0]
        assert set(rows.tolist()) <= set(BEAM_ROWS.tolist())
        rmax = LIDAR[wx][0]
        dep = c["depth"][c["valid"]] / 100.0
        assert dep.max() < rmax + 0.5


def test_fog_shortens_lidar_and_snow_adds_near_returns():
    a = make_dataset(6, 3, keep_boxes=True)
    ret = [a["valid"][a["weather"] == w].sum() for w in range(6)]
    assert ret[4] < ret[0] and ret[5] < ret[0]
    far_max = lambda w: (a["depth"][a["weather"] == w][a["valid"][a["weather"] == w]] / 100).max()
    assert far_max(4) < 25.5 < far_max(0)                                 # fog caps the range at 25 m

    def spurious(w):                                                      # near hits where the true surface is far
        s = a["weather"] == w
        dense = a["dense"][s] / 100.0
        return (a["valid"][s] & (a["depth"][s] < 800) & ((dense == 0) | (dense > 15))).sum()
    assert spurious(5) > 10 * max(1, spurious(0))


def test_tiles_range_and_layout():
    d = make_dataset(1, 1)
    t = to_tiles(d["rgb"], d["th"], d["depth"], d["valid"])
    assert t.shape == (6, T_ALL, 3, 3, H, W)
    assert t.min() >= 0 and t.max() <= 1
    assert np.allclose(t[:, :, 1, 0], t[:, :, 1, 2])                    # thermal is a grey video
    assert np.array_equal(t[:, :, 2, 2] > 0, d["valid"])                 # lidar channel 2 = validity


def test_model_shapes_one_and_three_tiles():
    m = VideoWorldModel()
    for n in (1, 3):
        x = torch.rand(2, T_ALL, 3, H, TILE_W * n)
        y = m(x, 6)
        assert y.shape == (2, T_ALL - 1, 3, H, TILE_W * n)


def test_zero_init_decoder_is_persistence():
    m = VideoWorldModel()
    x = torch.rand(2, T_ALL, 3, H, TILE_W)
    y = m(x, 6)
    assert torch.allclose(y[:, 5:], x[:, 5:6].expand_as(y[:, 5:]))        # untrained model repeats the last frame


def test_rollout_is_causal():
    torch.manual_seed(0)
    m = VideoWorldModel()
    for p in m.dec.net[-1].parameters():
        torch.nn.init.normal_(p, std=0.1)
    x = torch.rand(1, T_ALL, 3, H, TILE_W)
    y0 = m(x, 6)
    x2 = x.clone()
    x2[:, 6:] = torch.rand_like(x2[:, 6:])                                # change the (unseen) future
    assert torch.allclose(y0[:, 5:], m(x2, 6)[:, 5:])


def test_attention_mixes_tiles():
    """Changing the RGB tile must change the LiDAR-tile forecast (cross-sensor path exists)."""
    torch.manual_seed(0)
    m = VideoWorldModel()
    for p in m.dec.net[-1].parameters():
        torch.nn.init.normal_(p, std=0.1)
    x = torch.rand(1, T_ALL, 3, H, 3 * TILE_W)
    y = m(x, 6)[..., 2 * TILE_W:]
    x2 = x.clone()
    x2[..., :TILE_W] = torch.rand_like(x2[..., :TILE_W])
    assert not torch.allclose(y, m(x2, 6)[..., 2 * TILE_W:])


def test_blank_and_lidar_loss_masking():
    x = torch.rand(2, T_ALL, 3, H, TILE_W)
    x[:, :, 2] = (x[:, :, 2] > 0.5).float()
    p = x[:, 1:].clone()
    p[:, :, :2] += (1 - x[:, 1:, 2:3]) * 5.0                              # errors only where there is no return
    l, _ = tile_loss(p, x[:, 1:], [2])
    assert float(l) < 1e-5
    m = VideoWorldModel()
    blank = torch.zeros(1, 1, 1, TILE_W)
    y = m(torch.rand(1, T_ALL, 3, H, TILE_W), 6, blank=blank)
    assert torch.allclose(y, torch.zeros_like(y))
