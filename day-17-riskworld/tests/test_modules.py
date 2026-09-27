"""Shape / finiteness / value-range checks for each reconstructed module."""
import numpy as np
import torch

from src.actor_encoder import TemporalActorEncoder
from src.bev_encoder import BEVEncoder
from src.collision_score import CollisionScoreModule
from src.dataset import RiskWorldSyntheticDataset
from src.flow_occupancy import FlowGuidedOccupancyEvolution
from src.riskworld_model import RiskWorld
from src.risk_field import SpatialRiskField
from src.utils import load_config

CONFIG_PATH = "config.yaml"


def _cfg():
    return load_config(CONFIG_PATH)


def _all_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


def test_bev_encoder_shape_and_finite():
    cfg = _cfg()
    m = cfg["model"]
    G = cfg["data"]["grid_size"]
    enc = BEVEncoder(m["bev_in_channels"], m["bev_feat_channels"])
    x = torch.randn(2, m["bev_in_channels"], G, G)
    out = enc(x)
    assert out.shape == (2, m["bev_feat_channels"], G, G)
    assert _all_finite(out)


def test_actor_encoder_shape_and_finite():
    cfg = _cfg()
    m = cfg["model"]
    d = cfg["data"]
    G, A, T = d["grid_size"], d["max_agents"], d["history_len"]
    enc = TemporalActorEncoder(in_features=4, hidden_dim=m["actor_hidden_dim"], feat_dim=m["actor_feat_dim"], grid_size=G)
    history = torch.randn(2, A, T, 4)
    last_pos = torch.rand(2, A, 2) * (G - 1)
    mask = torch.ones(2, A)
    mask[:, -1] = 0  # one padding slot
    feat_map, agent_ctx = enc(history, last_pos, mask)
    assert feat_map.shape == (2, m["actor_feat_dim"], G, G)
    assert agent_ctx.shape == (2, A, m["actor_feat_dim"])
    assert _all_finite(feat_map)
    # padded agent's context and its scattered contribution must be exactly zero
    assert torch.all(agent_ctx[:, -1] == 0)


def test_risk_field_range_and_shape():
    cfg = _cfg()
    m = cfg["model"]
    G = cfg["data"]["grid_size"]
    risk = SpatialRiskField(m["bev_feat_channels"], m["risk_head_hidden"])
    feat = torch.randn(3, m["bev_feat_channels"], G, G)
    out = risk(feat)
    assert out.shape == (3, 1, G, G)
    assert _all_finite(out)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_flow_occupancy_valid_probability_range():
    cfg = _cfg()
    m = cfg["model"]
    G = cfg["data"]["grid_size"]
    horizon = cfg["data"]["horizon"]
    in_ch = m["bev_feat_channels"] + 1
    mod = FlowGuidedOccupancyEvolution(in_ch, G, m["flow_head_hidden"], m["residual_head_hidden"], max_horizon=horizon)
    feat = torch.randn(2, in_ch, G, G)
    prev_occ = torch.rand(2, 1, G, G)
    out = mod(feat, prev_occ, step_idx=0)
    for key in ("flow", "warped_occupancy", "residual", "occupancy"):
        assert _all_finite(out[key]), f"{key} has non-finite values"
    assert out["occupancy"].shape == (2, 1, G, G)
    # THE key invariant: warp+residual occupancy must stay a valid probability.
    assert float(out["occupancy"].min()) >= 0.0
    assert float(out["occupancy"].max()) <= 1.0
    assert float(out["warped_occupancy"].min()) >= 0.0
    assert float(out["warped_occupancy"].max()) <= 1.0


def test_riskworld_full_forward_shapes():
    cfg = _cfg()
    ds = RiskWorldSyntheticDataset(cfg, num_scenes=4, base_seed=1)
    batch_list = [ds[i] for i in range(4)]
    batch = {k: torch.stack([b[k] for b in batch_list]).float() for k in batch_list[0] if k != "has_hazard"}

    model = RiskWorld(cfg)
    out = model(
        bev_grid=batch["bev_grid"], agent_history=batch["agent_history"],
        agent_last_pos=batch["agent_last_pos"], agent_mask=batch["agent_mask"],
        prev_occupancy=batch["prev_occupancy"],
    )
    G = cfg["data"]["grid_size"]
    horizon = cfg["data"]["horizon"]
    assert out["risk_field"].shape == (4, 1, G, G)
    assert out["forecast_occupancy"].shape == (4, horizon, 1, G, G)
    assert out["persistence_occupancy"].shape == (4, horizon, 1, G, G)
    for key in ("risk_field", "forecast_occupancy", "persistence_occupancy", "flows", "residuals"):
        assert _all_finite(out[key])
    assert float(out["forecast_occupancy"].min()) >= 0.0
    assert float(out["forecast_occupancy"].max()) <= 1.0
    # persistence baseline must equal the input prev_occupancy repeated, unwarped
    for t in range(horizon):
        assert torch.allclose(out["persistence_occupancy"][:, t], batch["prev_occupancy"])


def test_collision_score_correction_always_nonnegative():
    cfg = _cfg()
    G = cfg["data"]["grid_size"]
    horizon = cfg["data"]["horizon"]
    B, K = 3, 5
    scorer = CollisionScoreModule(grid_size=G)

    torch.manual_seed(0)
    forecast_occ = torch.rand(B, horizon, 1, G, G)
    persistence_occ = torch.rand(B, horizon, 1, G, G)
    risk_field = torch.rand(B, 1, G, G)
    candidates = torch.rand(B, K, horizon, 2) * (G - 1)

    out = scorer(forecast_occ, persistence_occ, risk_field, candidates)
    assert out["correction"].shape == (B, K)
    assert _all_finite(out["correction"])
    assert float(out["correction"].min()) >= 0.0

    # Also stress-test with forecast intentionally much higher than persistence
    # everywhere, and the reverse, to make sure correction stays clamped at 0
    # in the case where persistence would "win".
    high_forecast = torch.ones(B, horizon, 1, G, G)
    low_persistence = torch.zeros(B, horizon, 1, G, G)
    out2 = scorer(high_forecast, low_persistence, risk_field, candidates)
    assert float(out2["correction"].min()) >= 0.0

    low_forecast = torch.zeros(B, horizon, 1, G, G)
    high_persistence = torch.ones(B, horizon, 1, G, G)
    out3 = scorer(low_forecast, high_persistence, risk_field, candidates)
    # forecast < persistence everywhere -> raw difference negative -> correction must clamp to 0
    assert float(out3["correction"].max()) == 0.0
