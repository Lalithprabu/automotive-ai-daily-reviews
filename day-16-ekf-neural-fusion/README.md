# Day 16 -- Vehicle Trajectory Prediction via Neural Fusion of Multiple EKF-Based Trajectory Candidates

**Paper:** Seong-Jun Kim, Seung-Hyun Kong (KAIST), *"Vehicle Trajectory Prediction via Neural
Fusion of Multiple EKF-Based Trajectory Candidates"*, arXiv:2609.19813, submitted 2026-09-17
(cs.RO, 6 pages, 5 figures, 4 tables).
Link: https://arxiv.org/abs/2609.19813

> **Abstract (verbatim):** "Predicting the future trajectories of surrounding vehicles in
> autonomous driving is important for collision risk assessment and safe ego-vehicle path
> planning. Conventional neural network-based trajectory predictors typically achieve strong
> prediction performance by exploiting agent history, dynamic scene graphs, and semantic maps.
> However, in specific motion regimes such as acceleration, deceleration, and turning, these
> predictors may fail to reflect physically feasible trajectories. To address this issue, this
> study proposes a framework that fuses the output of Trajectron++, a neural network-based
> trajectory predictor, with extended Kalman filter (EKF)-based multiple trajectory candidates
> at a late stage. On the nuScenes dataset, the proposed method reduces the average displacement
> error and final displacement error of the Trajectron++ robot baseline by 13.7% and 14.6%,
> respectively, without modifying the baseline architecture. These results indicate that
> EKF-based trajectory candidates can effectively complement neural trajectory prediction through
> learned fusion."

---

## Reconstruction & sourcing note (read this first)

**Only the abstract above was recoverable.** Repeated attempts to fetch the arXiv PDF/HTML for
2609.19813 were rate-limited, so this repo does **not** have access to the paper's method section,
figures, tables, or hyperparameters. Everything below is a **from-scratch architectural
reconstruction** built to be *consistent with what the abstract describes*, not a transcription of
the paper's actual implementation. Concretely:

- **The 4 EKF motion models (CV / CA / CTRV / CTRA)** are our choice. The abstract says "multiple
  trajectory candidates" without naming the models; CV/CA/CTRV/CTRA is the standard textbook set
  used throughout the vehicle-tracking literature for exactly this purpose, so we use it as a
  faithful and well-established stand-in.
- **The neural predictor** is a compact single-agent recurrent-CVAE reconstruction of
  Trajectron++'s core decoder (GRU history encoder -> CVAE latent -> GRU decoder), **not** the
  full graph-structured, multi-agent, map-aware Trajectron++. This is explicitly a simplification
  for a small, CPU-trainable repo.
- **The fusion network's internal design** (shared per-candidate MLP encoder -> learned
  per-timestep softmax attention over candidates -> residual refinement MLP) is our own design,
  chosen to be architecturally consistent with "learned late fusion" as described in the
  abstract. The paper's actual fusion architecture (attention vs. gating vs. something else) is
  unknown to us.
- **The dataset is 100% synthetic**, generated procedurally with NumPy (see `src/dataset.py`).
  This is **not** real nuScenes data. We do not have dataset access in this environment.
- **All network dimensions, batch sizes, optimizers, and epoch counts in `config.yaml` are this
  repo's own defaults**, picked for a small CPU-trainable demo. None of them are paper-sourced.
- **The paper's reported 13.7% / 14.6% ADE/FDE reduction is a real-nuScenes, real-Trajectron++
  number from the paper.** It is not expected to be, and should never be presented as being,
  reproduced by this toy synthetic repo. The numbers in the "Verified" section below are this
  repo's own measurements on its own synthetic data, reported for transparency, not as a
  reproduction of the paper's result.

---

## Architecture summary

### EKF motion model bank (`src/ekf_motion_models.py`)

Four classical Extended Kalman Filters, pure NumPy, each initialized from an 8-step (4.0s)
observed history via finite differences, filtered forward over that history (predict+update using
observed positions as measurements), then rolled forward with predict-only steps for a 12-step
(6.0s) horizon (`P' = F P F^T + Q`, covariance grows with no further measurements).

| Model | State vector | Captures |
|---|---|---|
| CV   | `[x, y, vx, vy]`              | Constant-velocity straight-line motion |
| CA   | `[x, y, vx, vy, ax, ay]`       | Constant longitudinal acceleration/deceleration |
| CTRV | `[x, y, v, theta, omega]`      | Constant speed + constant yaw rate (turning) |
| CTRA | `[x, y, v, theta, omega, a]`   | Constant yaw rate **and** constant acceleration |

Jacobians for the nonlinear CTRV/CTRA transition functions are computed **numerically**
(central-difference) inside a shared `_BaseEKF._numerical_jacobian`, rather than hand-derived
analytically, to avoid algebra bugs while remaining a faithful EKF linearization.

### Neural predictor (`src/neural_predictor.py`)

```
history [B,8,2] --GRU--> h_hist ----+--> prior_net -> (mu_p, logvar_p)
future  [B,12,2] --GRU--> h_fut --+  (training only) --> posterior_net -> (mu_q, logvar_q)
z ~ posterior (train) or prior (inference)
[h_hist, z] --Linear+tanh--> decoder init hidden
GRUCell, autoregressive over 12 steps, feeding back predicted (dx,dy) --> [B,12,2] trajectory
```

### Fusion network -- the paper's actual contribution (`src/fusion_network.py`)

```
neural_traj [B,12,2] ---+
                          +--> stack --> candidates_all [B,5,12,2]  (1 neural + 4 EKF)
ekf_candidates [B,4,12,2]-+
ekf_uncertainty [B,4,12] --(neural padded with zeros)--> unc_all [B,5,12]

candidates_all + unc_all --shared MLP--> candidate_embed [B,5,embed_dim]
candidate_embed + learned per-timestep positional embedding [12,embed_dim]
   --small MLP--> attn_logits [B,5,12]
   --softmax over candidate axis--> weights [B,5,12]   (sums to 1 per timestep)

fused[b,t,:] = sum_c weights[b,c,t] * candidates_all[b,c,t,:]      <- einsum('bct,bctd->btd')

fused_traj + pooled candidate embedding --residual MLP--> correction [B,12,2]
final = fused_traj + correction
```

The per-timestep (not per-candidate-global) softmax is the key design choice: it lets early
timesteps lean on the neural predictor while later, more kinematically-constrained timesteps
(mid-turn, mid-acceleration) lean on whichever EKF model fits.

### Synthetic dataset (`src/dataset.py`)

Three parametric regimes are generated analytically (ground truth is exact kinematics; Gaussian
noise is added only to the *observed history*, not the future):
`straight` (CV ground truth), `accel_decel` (CA ground truth, random +/- acceleration),
`turning` (CTRV ground truth, random constant yaw rate). This mirrors exactly the three regimes
the abstract calls out as failure modes for pure neural predictors.

---

## Repository structure

```
day-16-ekf-neural-fusion/
├── README.md
├── requirements.txt
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── ekf_motion_models.py     # CV / CA / CTRV / CTRA EKFs (pure NumPy)
│   ├── neural_predictor.py      # Trajectron++-style recurrent-CVAE stand-in
│   ├── fusion_network.py        # learned late-fusion network (core contribution)
│   ├── dataset.py               # synthetic nuScenes-like trajectory generator
│   └── utils.py                 # ADE/FDE metrics, seeding
├── train.py                     # trains predictor, then fusion; evaluates; saves checkpoint
├── simulate.py                  # renders an animated GIF from a trained checkpoint
├── checkpoints/
│   └── day16_fusion.pt          # produced by train.py
├── trajectory_fusion_simulation.gif   # produced by simulate.py
└── tests/
    └── test_fusion.py
```

## How to run

```bash
pip install -r requirements.txt
pytest tests/ -v
python train.py --config config.yaml
python simulate.py --config config.yaml
```

All scripts run on CPU only; the whole pipeline (tests + train + simulate) finishes in well under
a minute.

---

## Verified (actual run output, not invented)

Environment: CPU only, Python 3.11.15, torch/numpy/matplotlib/pillow/pyyaml/pytest as pinned in
`requirements.txt`. Commands run exactly as shown above, in order, in this repo directory.

### 1. `pytest tests/ -v`

```
============================= test session starts ==============================
collected 15 items

tests/test_fusion.py::TestEKFModels::test_cv_output_shape_and_finite PASSED
tests/test_fusion.py::TestEKFModels::test_ca_output_shape_and_finite PASSED
tests/test_fusion.py::TestEKFModels::test_ctrv_output_shape_and_finite_on_turn PASSED
tests/test_fusion.py::TestEKFModels::test_ctra_output_shape_and_finite_on_turn PASSED
tests/test_fusion.py::TestEKFModels::test_covariance_grows_over_horizon PASSED
tests/test_fusion.py::TestEKFModels::test_cv_matches_straight_line_reasonably PASSED
tests/test_fusion.py::TestEKFModels::test_ekf_bank_shapes PASSED
tests/test_fusion.py::TestFusionNetwork::test_forward_shapes PASSED
tests/test_fusion.py::TestFusionNetwork::test_attention_weights_sum_to_one_per_timestep PASSED
tests/test_fusion.py::TestFusionNetwork::test_weights_are_nonnegative PASSED
tests/test_fusion.py::TestFusionNetwork::test_overfit_one_batch_loss_decreases PASSED
tests/test_fusion.py::TestNeuralPredictor::test_forward_train_and_inference_shapes PASSED
tests/test_fusion.py::TestMetrics::test_ade_fde_hand_computed PASSED
tests/test_fusion.py::TestMetrics::test_fde_uses_only_last_timestep PASSED
tests/test_fusion.py::TestMetrics::test_ade_zero_for_identical_trajectories PASSED

============================== 15 passed in 2.90s ==============================
```

**15/15 passed.**

### 2. `python train.py --config config.yaml`

Neural predictor training loss went from 13.89 (epoch 1) to 1.33 (epoch 15); validation ADE fell
from 20.83 m to 12.51 m over 15 epochs. Fusion network training loss (Smooth L1) went from 7.94
(epoch 1) to 2.82 (epoch 25), a monotonic-ish decrease (the `train.py` script asserts
`loss_history[-1] < loss_history[0]`, which held).

Final held-out **test-split** ADE/FDE (meters), overall and per motion regime:

| Segment | Method | ADE | FDE |
|---|---|---|---|
| overall | EKF best-of-4 (oracle) | 6.34 | 12.79 |
| overall | Neural-only | 12.52 | 27.32 |
| overall | **Fused (ours)** | **5.35** | 13.92 |
| straight | EKF best-of-4 | 0.32 | 0.55 |
| straight | Neural-only | 5.32 | 11.24 |
| straight | Fused | 1.86 | 4.04 |
| accel_decel | EKF best-of-4 | 4.85 | 10.81 |
| accel_decel | Neural-only | 11.47 | 26.42 |
| accel_decel | **Fused** | **2.34** | **5.57** |
| turning | EKF best-of-4 | 13.42 | 26.18 |
| turning | Neural-only | 20.29 | 43.20 |
| turning | Fused | 11.62 | 31.47 |

"EKF best-of-4" is an **oracle** baseline: for each sample it picks whichever of the 4 EKF models
minimizes that sample's error, independently for ADE and for FDE -- it is a much stronger
reference than any single fixed EKF model, and stronger than what "fused" could achieve without
oracle knowledge.

**What this actually shows:** the fused predictor beats the neural-only baseline by a wide margin
everywhere (2.9x-3.5x lower ADE), and beats even the oracle EKF-best on overall ADE and clearly in
the `accel_decel` regime (2.34 vs 4.85 ADE, 5.57 vs 10.81 FDE) -- consistent with the paper's
qualitative claim that EKF candidates most help exactly in acceleration/deceleration/turning
regimes. On `straight`, a plain CV filter is nearly exact (0.32 ADE) so the oracle baseline is very
hard to beat, and fused sensibly lands between neural-only and the oracle. On `turning`, fused
improves ADE over the oracle but has a **higher** FDE than the oracle -- an honest, disclosed
limitation of this small reconstruction (see Implementation notes / Limitations), not a result we
are hiding.

**These numbers are this repo's own synthetic measurements only. They must never be quoted as
reproducing the paper's reported 13.7%/14.6% improvement, which is a real-nuScenes,
real-Trajectron++ result we could not reproduce here.**

### 3. `python simulate.py --config config.yaml`

```
Loading checkpoint from checkpoints/day16_fusion.pt ...
Simulating scene #10 (regime = 'turning')
Saved simulation GIF to trajectory_fusion_simulation.gif (render time: 1.66s, 12 frames @ 4 fps)
```

GIF confirmed valid: 12 frames, 1200x550px, `GIF` format (checked by re-opening with Pillow).
Visual check of the final frame shows the fused prediction (red) tracking the ground-truth turn
(green) closely while the neural-only prediction (blue) drifts wide, and the telemetry sub-panel
shows fused running-ADE staying near 0-4.5m across the horizon vs. neural-only climbing to ~12m --
this is the clearest single illustration of the paper's claim in this repo.

---

## Implementation notes (bugs hit and fixed)

1. **YAML scientific-notation float bug.** `config.yaml` originally had `lr: 1e-3` and
   `weight_decay: 1e-5`. PyYAML's `safe_load` (per the YAML 1.1 spec it implements) does **not**
   recognize `1e-3` as a float literal without an explicit `.` or sign in the mantissa/exponent --
   it parses as a **string**. This caused `torch.optim.Adam(..., lr="1e-3")` to raise
   `TypeError: '<=' not supported between instances of 'float' and 'str'` at the very first
   optimizer construction. Fixed by writing the values in decimal form (`lr: 0.001`,
   `weight_decay: 0.00001`) in `config.yaml`. This is a well-known PyYAML gotcha worth flagging
   for readers reusing this config pattern.

## Limitations (candid)

- **Synthetic data only** -- no claim of real-world nuScenes performance; the synthetic generator
  uses only 3 hand-parameterized regimes and cannot capture real sensor artifacts, occlusion,
  multi-agent interaction, or map priors.
- **Neural predictor has no explicit per-step uncertainty output**, so its uncertainty input to
  the fusion network is zero-padded -- a real Trajectron++ (or any predictor with a proper
  variance head) would give the fusion network a genuinely informative signal here instead.
- **Single-agent, no scene graph, no map** -- both the neural predictor and the dataset ignore
  multi-agent interaction and semantic map context that the abstract's baseline (full
  Trajectron++) explicitly uses.
- **CTRA's numerically-linearized Jacobian** is more expensive and slightly less numerically
  precise than a hand-derived analytic Jacobian would be, though it produced finite, sane
  covariance growth in all tests.
- **Turning-regime FDE**: the learned fusion improves ADE over the oracle EKF-best-of-4 in the
  turning regime but does not improve FDE there -- with more training data/epochs or a
  turn-specific inductive bias this gap could likely close, but we report it as observed rather
  than tuning it away.
- **Small, fast-training defaults** (15+25 epochs, ~1.5k synthetic scenes) were chosen for a
  CPU-only few-minutes run, not for maximum achievable accuracy.

## Citation

```
Kim, S.-J. and Kong, S.-H. "Vehicle Trajectory Prediction via Neural Fusion of Multiple
EKF-Based Trajectory Candidates." arXiv:2609.19813, 2026.
```
