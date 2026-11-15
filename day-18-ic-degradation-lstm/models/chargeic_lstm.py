"""
models/chargeic_lstm.py
==========================================================================
ChargeIC-LSTM -- this project's reconstruction of the architecture
described (at a high level, without implementation detail) in:

    "Intelligent Degradation Monitoring in Lithium-ion Batteries via
     Discharge Incremental Capacity Feature Estimation"
    Madmolilvand & Abdollahi, arXiv:2609.22843 (2026)

DISCLOSURE: The paper's retrievable abstract states only that "the LSTM
model provides the best balance of prediction accuracy and computational
efficiency" among "several architectures evaluated" -- it does NOT give
layer counts, hidden sizes, head structure, or a model name. Every
architectural detail below (2-layer LSTM with hidden_size=64, the exact
IC-regression head, the auxiliary SOH head, and the name "ChargeIC-LSTM"
itself) is THIS PROJECT'S OWN reconstruction default, chosen to be a
reasonable, lightweight, BMS-deployable design consistent with the paper's
stated goal ("real-time integration into Battery Management Systems"),
NOT a paper-verified specification.

The SOH auxiliary regression head is entirely this project's own addition
(for a richer telemetry dashboard in simulate.py) and has no basis in the
abstract at all.
==========================================================================
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChargeICLSTM(nn.Module):
    """
    Predicts a discharge incremental-capacity (IC) curve, dQ/dV vs V,
    directly from a charging-phase signal (voltage, current, temperature),
    plus an auxiliary state-of-health (SOH) scalar.

    Input
    -----
    x : FloatTensor, shape (B, T, 3)
        B = batch size
        T = number of timesteps in the charging window (reconstruction
            default T=120, e.g. a ~20 minute fast-charge segment sampled
            every 10 seconds)
        3 = [voltage, current, temperature] at each timestep

    Outputs
    -------
    ic_pred  : FloatTensor, shape (B, M)   -- predicted dQ/dV values on the
                                               fixed M-point voltage grid
                                               (reconstruction default M=50)
    soh_pred : FloatTensor, shape (B,)      -- predicted state-of-health
                                               fraction (this project's own
                                               auxiliary head, not paper-sourced)
    """

    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        ic_grid_size: int = 50,
        ic_head_hidden: int = 128,
        soh_head_hidden: int = 32,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ic_grid_size = ic_grid_size

        # --- Temporal encoder: 2-layer LSTM over the charging window -------
        # batch_first=True  => input/output tensors shaped (B, T, feature)
        # dropout applies BETWEEN stacked LSTM layers (not after the last
        # layer's output), which is the standard nn.LSTM semantics.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # --- IC-curve regression head ---------------------------------------
        # Maps the final hidden state (B, hidden_size) -> (B, ic_grid_size)
        self.ic_head = nn.Sequential(
            nn.Linear(hidden_size, ic_head_hidden),   # (B, 64)  -> (B, 128)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ic_head_hidden, ic_grid_size),   # (B, 128) -> (B, 50)
        )

        # --- Auxiliary SOH regression head (this project's own addition) ---
        # Maps the final hidden state (B, hidden_size) -> (B, 1) -> (B,)
        self.soh_head = nn.Sequential(
            nn.Linear(hidden_size, soh_head_hidden),  # (B, 64) -> (B, 32)
            nn.ReLU(),
            nn.Linear(soh_head_hidden, 1),              # (B, 32) -> (B, 1)
        )
        # SOH fractions live roughly in [0.6, 1.0] over a cell's useful
        # life (reconstruction default range). We predict a raw scalar and
        # squash it through a sigmoid rescaled into that band so the head
        # cannot output physically nonsensical values (e.g. SOH > 1.3).
        self._soh_lo = 0.55
        self._soh_hi = 1.02

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x : (B, T, 3)  charging-phase signal, already per-feature normalized
            (zero mean / unit std using TRAIN-SET statistics -- see
            data/synthetic_battery.py BatteryICDataset for how mean/std are
            computed on train and reused unchanged at val/test/inference).

        Returns
        -------
        ic_pred  : (B, M)
        soh_pred : (B,)
        """
        B, T, F = x.shape
        assert F == self.input_size, f"expected {self.input_size} input features, got {F}"

        # lstm_out : (B, T, hidden_size)  -- per-timestep hidden states of the TOP layer
        # h_n      : (num_layers, B, hidden_size) -- final hidden state of EACH layer
        # c_n      : (num_layers, B, hidden_size) -- final cell state of EACH layer
        lstm_out, (h_n, c_n) = self.lstm(x)  # x: (B, T, 3) -> lstm_out: (B, T, 64)

        # Take the LAST LAYER's final hidden state as the sequence summary.
        # h_n[-1] : (B, hidden_size) == (B, 64)
        h_final = h_n[-1]

        # --- IC-curve head ---------------------------------------------------
        ic_pred = self.ic_head(h_final)  # (B, 64) -> (B, 50)
        # dQ/dV is physically non-negative in this reconstruction's convention;
        # softplus keeps the output smooth and differentiable (unlike a hard
        # clamp) while enforcing non-negativity.
        ic_pred = torch.nn.functional.softplus(ic_pred)  # (B, 50)

        # --- SOH head ---------------------------------------------------------
        soh_raw = self.soh_head(h_final).squeeze(-1)  # (B, 32) -> (B, 1) -> (B,)
        soh_pred = self._soh_lo + (self._soh_hi - self._soh_lo) * torch.sigmoid(soh_raw)  # (B,)

        return ic_pred, soh_pred


def chargeic_loss(
    ic_pred: torch.Tensor,
    ic_true: torch.Tensor,
    soh_pred: torch.Tensor,
    soh_true: torch.Tensor,
    soh_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined training objective:
        L = MSE(ic_pred, ic_true) + soh_weight * MSE(soh_pred, soh_true)

    Returns (total_loss, ic_mse, soh_mse) so callers can log each term.
    """
    ic_mse = torch.nn.functional.mse_loss(ic_pred, ic_true)
    soh_mse = torch.nn.functional.mse_loss(soh_pred, soh_true)
    total = ic_mse + soh_weight * soh_mse
    return total, ic_mse, soh_mse


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Smoke test: one forward pass with random data, printing every
    # intermediate tensor shape through the model.
    # ------------------------------------------------------------------
    torch.manual_seed(0)
    B, T, F, M = 8, 120, 3, 50

    model = ChargeICLSTM(
        input_size=F,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        ic_grid_size=M,
        ic_head_hidden=128,
        soh_head_hidden=32,
    )
    print(model)

    x = torch.randn(B, T, F)
    ic_pred, soh_pred = model(x)
    print(f"\nInput x:         {tuple(x.shape)}")
    print(f"ic_pred:         {tuple(ic_pred.shape)}  (expected ({B}, {M}))")
    print(f"soh_pred:        {tuple(soh_pred.shape)}  (expected ({B},))")
    print(f"soh_pred range:  [{soh_pred.min().item():.3f}, {soh_pred.max().item():.3f}]")

    ic_true = torch.rand(B, M)
    soh_true = torch.rand(B) * 0.4 + 0.6
    total, ic_mse, soh_mse = chargeic_loss(ic_pred, ic_true, soh_pred, soh_true)
    print(f"\nSmoke-test loss: total={total.item():.4f}  ic_mse={ic_mse.item():.4f}  soh_mse={soh_mse.item():.4f}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
