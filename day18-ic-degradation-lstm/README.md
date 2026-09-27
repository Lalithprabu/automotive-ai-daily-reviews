# Day 18 -- ChargeIC-LSTM: Discharge IC-Feature Estimation from Charging Signals

**Series milestone:** this is the **first EV-battery-category paper** in this daily series
(Days 1-17 searched the battery/BMS category every day and found nothing suitable to cover --
this finally breaks that streak).

Reconstruction of the ideas in:

> **"Intelligent Degradation Monitoring in Lithium-ion Batteries via Discharge Incremental
> Capacity Feature Estimation"**
> Amir Madmolilvand, Farzaneh Abdollahi -- Amirkabir University of Technology
> arXiv:2609.22843 (submitted 2026-09-19), category eess.SY

---

## Reconstruction & sourcing note (READ THIS FIRST)

The paper's full text was **rate-limited / inaccessible** at build time. The **only** source
material available was the abstract, reproduced verbatim below. **Everything in this repo
beyond the abstract's claims is this project's own engineering reconstruction**, built to give
a runnable, honestly-evaluated demonstration of the *idea* the abstract describes -- it is
**not** a reproduction of the paper's actual model, dataset, or numbers.

> "Accurate and timely detection of degradation in lithium-ion batteries is crucial to ensure
> safety, reliability, and longevity in high-demand applications such as electric vehicles and
> energy storage systems. Traditional incremental capacity (IC) analysis methods require
> low-current cycling for discharge measurements, limiting their practical use in real-time
> battery management. This paper proposes a novel neural network-based framework that predicts
> discharge IC features directly from charging signals, eliminating the need for low-current
> discharge. Trained on a comprehensive dataset of 53 battery cells cycled under diverse
> fast-charging protocols, the model demonstrates robust generalization ability, effectively
> estimating degradation indicators on unseen battery data. Among several architectures
> evaluated, the LSTM model provides the best balance of prediction accuracy and computational
> efficiency. The proposed approach enables real-time integration into Battery Management
> Systems (BMS), enhancing degradation monitoring without disrupting normal battery operation."

### What is paper-sourced vs. this project's own default

| Item | Source |
|---|---|
| Task framing (predict discharge IC features from charging-phase signals only) | **Paper** (abstract) |
| "LSTM gives the best accuracy/efficiency balance among several architectures" | **Paper** (abstract) |
| Dataset size = 53 cells, diverse fast-charging protocols | **Paper** (abstract) |
| "Generalizes to unseen battery data" | **Paper** (abstract, claim reproduced as a design *goal* here via cell-level train/val/test split) |
| Real-time / BMS-deployable framing | **Paper** (abstract) |
| Model name **"ChargeIC-LSTM"** | **This project's own name.** The paper does not name its model. |
| Exact architecture: 2-layer LSTM, hidden_size=64, dropout=0.2, IC head `64->128->50`, T=120, M=50 | **This project's reconstruction default.** No layer counts/dims are given in the abstract. |
| Auxiliary **SOH regression head** | **This project's own addition**, entirely absent from the abstract, added for a richer telemetry dashboard in `simulate.py`. |
| Entire **synthetic dataset** (53 simulated cells, fade curves, resistance growth, Gaussian-bump IC curves, CC-taper charging signal) | **This project's own simulator.** No real cell-cycling data was used or is claimed. See `data/synthetic_battery.py` docstring for the full physics-inspired design rationale. |
| All hyperparameters in `config.yaml` (epochs, batch size, LR, etc.) | **This project's own defaults.** |
| All numeric results below (loss curves, MSE, MAE) | **Synthetic-data results from this reconstruction.** NOT the paper's own reported numbers (which we do not have). |

**Bottom line: nothing below is a claim about the real paper's performance.** It is a
demonstration that this project's ChargeIC-LSTM architecture can learn the *type* of mapping
the abstract describes (charging signal -> discharge IC feature) on physics-inspired synthetic
data, with a genuine held-out-cell evaluation and an explicit sanity check against a trivial
baseline.

---

## Task framing (this project's reconstruction)

- **Input:** a multivariate time series of the **charging phase only** -- voltage, current,
  temperature -- over a fixed-length fast-charge window: `T=120` timesteps (~20 minutes @ 10s
  sampling; reconstruction default).
- **Output:** the corresponding **discharge incremental-capacity curve** (dQ/dV vs. voltage),
  resampled onto a fixed `M=50`-point voltage grid (reconstruction default).
- **Post-processed degradation indicators** (simple, from the predicted curve):
  - **peak IC height** -- fades with degradation.
  - **peak voltage location** -- shifts as internal resistance grows.
- **Auxiliary head (this project's own addition):** a direct SOH% regression head, for a
  richer BMS-style telemetry readout in `simulate.py`. Not paper-sourced.

## Model: `ChargeIC-LSTM`

```
Input (B, T=120, 3)  [voltage, current, temperature]
        |
        v
  per-feature normalization (train-set mean/std, stored in checkpoint)
        |
        v
  nn.LSTM(input_size=3, hidden_size=64, num_layers=2,
          batch_first=True, dropout=0.2)
        |
        v
  h_n[-1]  ->  (B, 64)  final hidden state of the last layer
        |
        +----------------------------------+
        v                                  v
  IC head:                            SOH head (aux, own addition):
  Linear(64,128) -> ReLU ->           Linear(64,32) -> ReLU ->
  Dropout(0.2) -> Linear(128,50)      Linear(32,1) -> sigmoid-rescale
        |                                  |
        v                                  v
  ic_pred (B, 50) via softplus        soh_pred (B,) in [0.55, 1.02]

  Loss = MSE(ic_pred, ic_true) + 0.5 * MSE(soh_pred, soh_true)
```

Total parameters: **67,827** (lightweight, consistent with the abstract's "real-time BMS
integration" framing -- this project's own confirmation, not a paper-reported parameter count).

## Folder structure

```
day18-ic-degradation-lstm/
├── README.md
├── requirements.txt
├── config.yaml
├── models/
│   └── chargeic_lstm.py       # ChargeIC-LSTM model + loss, smoke test
├── data/
│   └── synthetic_battery.py    # 53-cell synthetic generator + Dataset class
├── src/
│   ├── train.py                 # training loop, checkpoint, sanity-check vs baseline
│   └── utils.py                 # config/seed/dataset/checkpoint helpers
├── simulate.py                   # real-time BMS-style animated GIF from the trained model
├── tests/
│   ├── test_model.py             # shape / NaN / loss-decreases tests
│   └── test_data.py              # dataset shape / split / linear-probe sanity tests
├── checkpoints/                   # created by train.py (best model + metrics.json)
└── battery_ic_simulation.gif     # created by simulate.py
```

## How to run

```bash
pip install -r requirements.txt

# 1. Run tests
pytest tests/ -v

# 2. Train (generates the synthetic 53-cell dataset, trains, saves checkpoint)
python src/train.py

# 3. Render the real-time simulation GIF from the trained checkpoint
python simulate.py
```

---

## Verified results (synthetic-data, this reconstruction only)

### 1. Tests

```
pytest tests/ -v
```
**14 passed, 0 failed** (7 in `test_data.py`, 7 in `test_model.py`). Covers: correct 53-cell
count, tensor shapes, no-NaN/no-Inf checks, cell-level train/val/test disjointness, SOH-fade
trend sanity, normalization using train-only statistics, a one-batch train-loss-decreases
check, and a linear-probe-beats-mean-baseline sanity check on the synthetic charging signal.

### 2. Training (`python src/train.py`)

- Split (by **cell**, not cycle): **37 train / 8 val / 8 test cells** -> 2,794 / 606 / 555
  training cycles respectively. Val and test cells are **entirely unseen** during training.
- 25 epochs, ~55.5s total on CPU.
- **Train loss:** 0.1293 (epoch 1) -> **0.00273** (epoch 25).
- **Best val loss:** 0.00179 (epoch 20), used for checkpoint selection.
- **Held-out TEST metrics (unseen cells):** IC-curve MSE = **0.00105**, SOH MAE = **0.0136**
  (i.e. ~1.4 SOH percentage points average error on cells never seen during training).
- **Sanity check vs. "predict the training-mean IC curve" baseline:**
  - val: baseline MSE 0.00753 vs. model MSE 0.00199 -> **model beats baseline (PASS)**
  - test: baseline MSE 0.00503 vs. model MSE 0.00105 -> **model beats baseline (PASS)**

These are modest, honest numbers on a synthetic task with real cell-level held-out
generalization -- not cherry-picked, not paper numbers.

### 3. Simulation (`python simulate.py`)

- Loaded the real trained checkpoint (best val epoch = 20).
- Ran on held-out **test cell_id=33** (3.54C fast-charge, 92 cycles available), animating the
  first 30 cycles.
- Output verified by reopening with PIL: `battery_ic_simulation.gif`, format **GIF**, size
  **1100x800 px**, **30 frames**, **~942 KB** on disk.
- Three panels per frame: (1) live V/I/T charging feed for the current cycle, (2) ground-truth
  vs. predicted discharge IC curve overlay, (3) a running true-vs-predicted SOH and peak-IC-height
  telemetry trend across cycles seen so far.

---

## Implementation notes (real bugs hit + fixes, per series convention)

1. **Bare scientific notation in YAML.** Following this series' established convention, every
   float in `config.yaml` is written as an explicit decimal (e.g. `0.001`, not `1e-3`) to avoid
   the known PyYAML `safe_load()` gotcha where some bare exponent forms parse as strings, not
   floats, in certain YAML 1.1 parser configurations. Caught proactively before it could bite;
   no runtime failure occurred because of it here, but the convention was kept.
2. **Real bug found and fixed via the sanity check:** the first version of the synthetic
   generator drew each cell's capacity-fade rate and internal-resistance-growth rate as fully
   **independent** random variables. The charging signal only encodes resistance (via the
   CC-phase-shortens-as-R-grows mechanism), while SOH only depends on the fade rate. Because
   these were independent per cell, `tests/test_data.py::test_charging_signal_beats_mean_baseline_via_linear_probe`
   **failed on the first run** (linear-probe MSE 0.00088 vs. baseline MSE 0.00087 -- essentially
   no signal). Fix: introduced a shared per-cell "quality" latent factor that partially drives
   *both* fade rate and resistance-growth rate (physically reasonable -- manufacturing variance
   plausibly affects multiple aging modes together), while keeping independent per-cell noise so
   it isn't a deterministic shortcut. After the fix, the same test passes and the trained model's
   SOH MAE dropped to a sensible ~0.014 on held-out cells. This is a genuine "was the model
   learning noise?" catch, not a cosmetic change -- documented here per this series' house style
   of disclosing real bugs rather than hiding them.
3. No NaN/masking or metrics-stuck-at-zero issues were encountered this round; the two items
   above are the real, load-bearing implementation issues from this day's build.

---

## Limitations (of this reconstruction, disclosed candidly)

- **No real data.** All 53 "cells" are synthetic and physics-inspired, not measured. Absolute
  numbers here say nothing about performance on a real fast-charging fleet.
- **IC-curve shape is a stylized single Gaussian bump**, not a real dQ/dV curve (real curves
  can have multiple peaks/shoulders reflecting multiple phase-transition electrochemistry).
- **"Real-time" simulation is cycle-by-cycle**, not a true sub-cycle streaming replay of BMS
  telemetry.
- **SOH auxiliary head is this project's own addition** and inherits whatever correlation
  structure the synthetic generator happens to encode between resistance-driven charging
  signal and fade-driven SOH -- a real deployment would need to verify this correlation holds
  on physical cells.
- Small model (68K params) trained briefly (25 epochs, CPU) -- no hyperparameter search,
  no architecture ablation against the "several architectures evaluated" the abstract mentions.

## References

- Madmolilvand, A., Abdollahi, F. "Intelligent Degradation Monitoring in Lithium-ion Batteries
  via Discharge Incremental Capacity Feature Estimation." arXiv:2609.22843, 2026.
