"""Shape and no-NaN sanity checks for every module in the ForeDrive reconstruction."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import pytest

from models.encoder import VisualEncoder
from models.world_model import MultiHorizonLatentPredictor
from models.fusion import GatedFutureFusion
from models.dit_planner import DiTPlanner
from models.foredrive import ForeDriveModel

CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


@pytest.fixture(scope="module")
def cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def test_config_types_are_numeric(cfg):
    # Guards against the known PyYAML bare-scientific-notation bug (see
    # config.yaml header comment): every value that should be a float must
    # actually parse as a Python float/int, not a string.
    assert isinstance(cfg["train"]["learning_rate"], float)
    assert isinstance(cfg["train"]["weight_decay"], float)
    assert isinstance(cfg["dit"]["beta_start"], float)
    assert isinstance(cfg["dit"]["beta_end"], float)


def test_encoder_shapes(cfg):
    enc = VisualEncoder(
        in_channels=cfg["data"]["raster_channels"], raster_size=cfg["data"]["raster_size"],
        tokens_per_side=cfg["encoder"]["tokens_per_side"], token_dim=cfg["encoder"]["token_dim"],
    )
    x = torch.randn(3, cfg["data"]["raster_channels"], cfg["data"]["raster_size"], cfg["data"]["raster_size"])
    tokens = enc(x)
    n = cfg["encoder"]["tokens_per_side"] ** 2
    assert tokens.shape == (3, n, cfg["encoder"]["token_dim"])
    assert not torch.isnan(tokens).any()


def test_world_model_shapes(cfg):
    token_dim = cfg["encoder"]["token_dim"]
    n = cfg["encoder"]["tokens_per_side"] ** 2
    h = len(cfg["data"]["horizons"])
    wm = MultiHorizonLatentPredictor(
        token_dim=token_dim, ego_state_dim=cfg["encoder"]["ego_state_dim"], num_horizons=h,
        hidden_dim=cfg["world_model"]["hidden_dim"], num_layers=cfg["world_model"]["num_layers"],
        num_heads=cfg["world_model"]["num_heads"], dropout=cfg["world_model"]["dropout"],
    )
    tokens = torch.randn(2, n, token_dim)
    ego = torch.randn(2, cfg["encoder"]["ego_state_dim"])
    pred, conf = wm(tokens, ego)
    assert pred.shape == (2, h, n, token_dim)
    assert conf.shape == (2, h, n)
    assert (conf >= 0).all() and (conf <= 1).all()
    assert not torch.isnan(pred).any()


def test_fusion_shapes(cfg):
    token_dim = cfg["encoder"]["token_dim"]
    n = cfg["encoder"]["tokens_per_side"] ** 2
    h = len(cfg["data"]["horizons"])
    fused_dim = cfg["fusion"]["fused_token_dim"]
    fusion = GatedFutureFusion(token_dim=token_dim, num_horizons=h,
                                status_dim=cfg["fusion"]["status_dim"], fused_token_dim=fused_dim)
    cur = torch.randn(2, n, token_dim)
    pred = torch.randn(2, h, n, token_dim)
    conf = torch.rand(2, h, n)
    fused, conf_out = fusion(cur, pred, conf)
    assert fused.shape == (2, n * (1 + h), fused_dim)
    assert not torch.isnan(fused).any()


def test_dit_planner_shapes(cfg):
    fused_dim = cfg["fusion"]["fused_token_dim"]
    t = cfg["data"]["num_waypoints"]
    s = 10
    dit = DiTPlanner(
        fused_token_dim=fused_dim, num_waypoints=t, hidden_dim=cfg["dit"]["hidden_dim"],
        num_layers=cfg["dit"]["num_layers"], num_heads=cfg["dit"]["num_heads"],
        dropout=cfg["dit"]["dropout"], tab_hidden_dim=cfg["tab"]["hidden_dim"],
    )
    noisy = torch.randn(2, t, 2)
    timestep = torch.randint(0, 50, (2,))
    cond = torch.randn(2, s, fused_dim)
    eps = dit(noisy, timestep, cond)
    assert eps.shape == (2, t, 2)
    assert not torch.isnan(eps).any()


def test_full_model_forward_train_no_nan(cfg):
    model = ForeDriveModel(cfg)
    b = 3
    h = len(cfg["data"]["horizons"])
    current = torch.randn(b, cfg["data"]["raster_channels"], cfg["data"]["raster_size"], cfg["data"]["raster_size"])
    future = torch.randn(b, h, cfg["data"]["raster_channels"], cfg["data"]["raster_size"], cfg["data"]["raster_size"])
    ego = torch.randn(b, cfg["encoder"]["ego_state_dim"])
    waypoints = torch.randn(b, cfg["data"]["num_waypoints"], 2)

    out = model.forward_train(current, future, ego, waypoints)
    assert not torch.isnan(out["forecast_loss"]).any()
    assert not torch.isnan(out["planning_loss"]).any()
    assert out["forecast_loss"].item() >= 0
    assert out["planning_loss"].item() >= 0


def test_full_model_sample_trajectory_no_nan(cfg):
    model = ForeDriveModel(cfg)
    model.eval()
    b = 2
    current = torch.randn(b, cfg["data"]["raster_channels"], cfg["data"]["raster_size"], cfg["data"]["raster_size"])
    ego = torch.randn(b, cfg["encoder"]["ego_state_dim"])
    traj, conf = model.sample_trajectory(current, ego)
    assert traj.shape == (b, cfg["data"]["num_waypoints"], 2)
    assert not torch.isnan(traj).any()
    assert not torch.isinf(traj).any()
    assert conf.shape == (b, len(cfg["data"]["horizons"]), cfg["encoder"]["tokens_per_side"] ** 2)
