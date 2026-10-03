import sys, inspect, warnings
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore")
from maplightning import MapLightning, BEVBaseline, make_batch
from maplightning.data import project, nominal_camera, IMG_W, IMG_H, N_PTS, NOM
from maplightning.loss import set_loss, hungarian


def test_projection_straight_ahead_is_centered_horizontally():
    cam = nominal_camera(1)
    pts = torch.tensor([[[10., 0., 0.]]])
    u, v, zc = project(pts, cam["h"], cam["pitch"], cam["yaw"])
    assert abs(float(u) - IMG_W / 2) < 1e-3 and float(v) > IMG_H / 2 and float(zc) > 0


def test_left_world_point_is_left_in_image():
    cam = nominal_camera(1)
    u, _, _ = project(torch.tensor([[[10., 3., 0.]]]), cam["h"], cam["pitch"], cam["yaw"])
    assert float(u) < IMG_W / 2


def test_batch_shapes_and_range():
    b = make_batch(4, 1.0)
    assert b["img"].shape == (4, 3, IMG_H, IMG_W) and 0 <= float(b["img"].min()) and float(b["img"].max()) <= 1
    assert b["polys"].shape == (4, 4, N_PTS, 2) and b["present"][:, :3].all()


def test_model_output_shapes():
    img = make_batch(3)["img"]
    for M in (MapLightning(), BEVBaseline()):
        lg, pts = M(img)
        assert lg.shape == (3, 6, 3) and pts.shape == (3, 6, N_PTS, 2)


def test_maplightning_has_no_camera_input():
    assert list(inspect.signature(MapLightning.forward).parameters) == ["self", "img"]


def test_image_tokens_are_discarded_before_decoding():
    m = MapLightning(n_map=24).eval()
    assert m.map_tokens_out(make_batch(2)["img"]).shape[1] == 24          # not 24 + 72


def test_baseline_depends_on_camera_calibration():
    m = BEVBaseline().eval()
    b = make_batch(2, 0.0)
    cam2 = {k: v.clone() for k, v in b["cam"].items()}; cam2["pitch"] += 0.1
    assert not torch.allclose(m(b["img"], b["cam"])[1], m(b["img"], cam2)[1])


def test_hungarian_perfect_prediction_has_zero_point_loss():
    b = make_batch(4)
    pts = torch.zeros(4, 6, N_PTS, 2); logits = torch.zeros(4, 6, 3); logits[..., 2] = 5
    pts[:, :4] = b["polys"]
    for i in range(4):
        for g in range(4):
            logits[i, g, 2] = 0; logits[i, g, b["cls"][i, g]] = 8
    _, parts = set_loss(logits, pts, b)
    assert parts["pts"] < 1e-5


def test_gradients_flow_to_map_tokens_and_stem():
    m = MapLightning(); b = make_batch(4)
    lg, p = m(b["img"]); l, _ = set_loss(lg, p, b); l.backward()
    assert m.map_tokens.grad.abs().sum() > 0 and m.stem.net[0].weight.grad.abs().sum() > 0


def test_overfit_tiny_batch():
    torch.manual_seed(0)
    m = MapLightning(); b = make_batch(4, 0.0, torch.Generator().manual_seed(1))
    opt = torch.optim.AdamW(m.parameters(), 3e-3); first = None
    for i in range(150):
        lg, p = m(b["img"]); l, parts = set_loss(lg, p, b); opt.zero_grad(); l.backward(); opt.step()
        first = first or float(l)
    assert float(l) < 0.5 * first
