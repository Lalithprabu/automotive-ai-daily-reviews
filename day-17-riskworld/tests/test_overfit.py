"""Sanity test: the model can overfit a single small batch, i.e. loss
decreases substantially over a short training loop on fixed data. This is a
basic learnability check, not a claim about generalization."""
import torch
import torch.nn.functional as F

from src.dataset import RiskWorldSyntheticDataset
from src.riskworld_model import RiskWorld
from src.utils import load_config, set_seed


def test_overfit_single_batch_loss_decreases():
    cfg = load_config("config.yaml")
    set_seed(cfg["seed"])

    ds = RiskWorldSyntheticDataset(cfg, num_scenes=4, base_seed=123)
    batch_list = [ds[i] for i in range(4)]
    batch = {k: torch.stack([b[k] for b in batch_list]).float() for k in batch_list[0] if k != "has_hazard"}

    model = RiskWorld(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)

    losses = []
    for _ in range(40):
        out = model(
            bev_grid=batch["bev_grid"], agent_history=batch["agent_history"],
            agent_last_pos=batch["agent_last_pos"], agent_mask=batch["agent_mask"],
            prev_occupancy=batch["prev_occupancy"],
        )
        occ_loss = F.mse_loss(out["forecast_occupancy"], batch["future_occupancy_gt"])
        risk_loss = F.mse_loss(out["risk_field"], batch["risk_target"])
        loss = occ_loss + 0.5 * risk_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses))), "loss became non-finite (NaN/Inf) during overfitting"
    # Should drop substantially when overfitting 4 fixed scenes for 40 steps.
    assert losses[-1] < losses[0] * 0.5, f"loss did not decrease enough: {losses[0]:.5f} -> {losses[-1]:.5f}"
