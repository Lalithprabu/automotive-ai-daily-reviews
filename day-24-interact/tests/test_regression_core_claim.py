"""
Regression test for the core empirical claim this reconstruction makes:
"anchor-conditioned prediction beats a non-interactive (unconditional) baseline at
capturing the simulated reactive behavior, especially in interactive scenarios."

This is a REAL, self-contained quick training run (small dataset, few epochs, small
model) -- not a check against hand-picked numbers -- so it is a genuine falsifiable
regression test: if the AnchorConditionedPredictor stopped learning the intent
dependency, this test would fail. Kept small so pytest stays fast (~5-10s), separate
from train.py's larger/slower run.
"""
import torch
from torch.utils.data import DataLoader

from src.data.synthetic_interactive_scenario import InteractiveMergeDataset, collate_episodes
from src.models.anchor_conditioned_predictor import AnchorConditionedPredictor
from src.models.anchor_generator import AnchorGenerator


def _quick_train(model, loader, epochs, force_zero_cond):
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    for _ in range(epochs):
        for batch in loader:
            cond = torch.zeros_like(batch["cond"]) if force_zero_cond else batch["cond"]
            pred = model(batch["ego_hist"], batch["other_hist"], cond)
            loss = loss_fn(pred, batch["other_future"])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def test_conditioned_predictor_beats_unconditional_on_interactive_bucket():
    torch.manual_seed(123)
    H, T, HIDDEN = 8, 10, 16

    train_ds = InteractiveMergeDataset(n_samples=600, seed=42, history_len=H, future_len=T,
                                        cond_mode="continuous")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_episodes)

    anchor_gen = AnchorGenerator()
    anchors = anchor_gen.generate()
    anchor_conds = [a.cond for a in anchors]
    anchor_names = [a.name for a in anchors]

    val_ds = InteractiveMergeDataset(n_samples=400, seed=99, history_len=H, future_len=T,
                                      cond_mode="anchors", anchor_conds=anchor_conds,
                                      anchor_names=anchor_names)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False, collate_fn=collate_episodes)
    val_batch = next(iter(val_loader))

    conditioned = AnchorConditionedPredictor(hidden_dim=HIDDEN, future_len=T)
    unconditional = AnchorConditionedPredictor(hidden_dim=HIDDEN, future_len=T)

    _quick_train(conditioned, train_loader, epochs=8, force_zero_cond=False)
    _quick_train(unconditional, train_loader, epochs=8, force_zero_cond=True)

    conditioned.eval()
    unconditional.eval()
    with torch.no_grad():
        pred_c = conditioned(val_batch["ego_hist"], val_batch["other_hist"], val_batch["cond"])
        pred_u = unconditional(val_batch["ego_hist"], val_batch["other_hist"],
                                torch.zeros_like(val_batch["cond"]))

    gt = val_batch["other_future"]
    a = val_batch["assertiveness"]
    interactive_mask = a.abs() > 0.3
    assert interactive_mask.sum() > 0, "test scenario setup produced no interactive-bucket samples"

    ade_c = torch.norm(pred_c[interactive_mask] - gt[interactive_mask], dim=-1).mean().item()
    ade_u = torch.norm(pred_u[interactive_mask] - gt[interactive_mask], dim=-1).mean().item()

    print(f"\n[regression] interactive-bucket ADE: conditioned={ade_c:.4f} unconditional={ade_u:.4f}")

    # the core, falsifiable claim: anchor conditioning must materially reduce error
    # in scenarios where the other agent's reaction actually depends on ego intent.
    assert ade_c < ade_u, (
        f"anchor-conditioned predictor (ADE={ade_c:.4f}) did NOT beat the unconditional "
        f"baseline (ADE={ade_u:.4f}) on the interactive bucket -- core claim not supported."
    )
    # require a meaningfully large margin (not just noise) given the reaction magnitude
    # designed into the synthetic simulator (k_speed=0.6, k_lateral=0.5)
    assert ade_u - ade_c > 0.02
