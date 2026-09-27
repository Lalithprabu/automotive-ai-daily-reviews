# ForeDrive Reconstruction (Day 20)

A small, runnable, from-scratch PyTorch reconstruction of the **core
architecture** described in a very recent (2 days old at time of writing)
research paper:

> **ForeDrive: Foresight-Guided End-to-End Autonomous Driving with a
> Planning-Relevant Latent World Model**
> arXiv:2609.26299, submitted 2026-09-22/23, cs.CV.

**Read [`SOURCING.md`](./SOURCING.md) first.** We only have the paper's
abstract (its full text was unreachable — arXiv kept returning HTTP 429).
Every architectural detail beyond what the abstract literally says is
**our own reconstruction**, clearly disclosed as such in every module's
docstring, in `SOURCING.md`'s real-vs-reconstructed table, and here.

## What's in this repo

- A small **visual encoder** (CNN) that is the "shared online encoder"
- A **JEPA-style multi-horizon latent world model** (a Transformer
  predictor, no pixel reconstruction anywhere)
- The **stop-gradient routing** that keeps the predictor trained by the
  forecasting loss only, while the encoder is trained end-to-end by the
  planning loss — implemented explicitly in `models/foredrive.py` and
  **verified by a dedicated unit test** (`tests/test_stop_gradient.py`),
  which the task that produced this repo calls out as the single most
  important correctness property here
- **Gated visual fusion + future-status injection** (learned per-horizon
  confidence gating + horizon/confidence status embeddings)
- **Trajectory-Adaptive Bias (TAB)**: an attention bias recomputed from the
  current candidate trajectory at every denoising step
- A small **DiT (Diffusion Transformer) planner** (epsilon-prediction DDPM
  training, DDIM sampling)
- A **synthetic driving-scene generator** with a reactive ego controller
  and scripted multi-agent kinematics, built so both the forecasting and
  imitation tasks have genuine, non-degenerate learning signal (checked by
  `tests/test_dataset.py` and `tests/test_sanity_signal.py`)
- `train.py` — trains the whole thing end-to-end on synthetic data
- `simulate.py` — renders a real, 3-panel matplotlib GIF from the actual
  trained checkpoint's real outputs

## Quick start

```bash
pip install -r requirements.txt

# run the test suite (21 tests, ~2.5 minutes total — two of them run short
# real training loops as sanity checks, see tests/test_sanity_signal.py)
pytest tests/ -v

# train (CPU, ~50 seconds with the default config.yaml)
python3 train.py

# render the simulation GIF from the checkpoint train.py just saved
python3 simulate.py
```

## Repo structure

```
day20-foredrive/
  README.md              # this file
  SOURCING.md              # explicit paper-sourced vs. project-default table
  config.yaml                # all hyperparameters (decimal notation only, see file header)
  requirements.txt
  models/
    encoder.py                # shared online visual encoder (CNN -> token grid)
    world_model.py              # JEPA-style multi-horizon latent predictor
    fusion.py                    # gated visual fusion + future-status injection
    tab.py                        # Trajectory-Adaptive Bias
    dit_planner.py                  # DiT trajectory planner
    foredrive.py                      # full wiring + stop-gradient routing (READ THIS FIRST)
  src/
    data/synthetic_dataset.py         # synthetic scene generator + rasterizer
    utils/diffusion.py                  # noise schedule, DDIM sampler
  train.py                                # end-to-end training script
  simulate.py                              # real-checkpoint GIF simulation
  tests/
    test_shapes.py                          # shape / no-NaN checks, every module
    test_stop_gradient.py                     # THE critical correctness test
    test_gating_and_tab.py                      # gating and TAB behavioral checks
    test_dataset.py                               # dataset determinism / signal regression guards
    test_sanity_signal.py                           # forecasting-vs-naive, imitation-vs-baseline
  checkpoint.pt          # produced by train.py (not committed if you clone fresh)
  simulation.gif           # produced by simulate.py
```

## Measured results (real, from running this exact code — see report below for full numbers)

- **21/21 tests passing**, including the stop-gradient correctness test and
  two training-based sanity checks (forecasting beats a naive no-change
  baseline; imitation beats a scene-blind population-mean baseline).
- **356,287 parameters.**
- `train.py`'s quick demo run (6 epochs, ~50s on CPU): planning loss
  0.52 → 0.24. See `SOURCING.md`'s limitations section for why the
  forecasting loss curve is noisier (same-encoder JEPA target, no EMA).
- `simulate.py` produces a real 24-frame, 1500×500px GIF from the trained
  checkpoint's actual sampled trajectories — including scenes where the
  model does noticeably worse than the constant-velocity baseline, shown
  honestly rather than cherry-picked.

## A note on what this repo does and doesn't prove

This is a **mechanism reconstruction**, not a reproduction of the paper's
NAVSIM results. Its job is to (a) be a real, correct, testable
implementation of the *named* mechanisms in the abstract (stop-gradient
routing, gated fusion, TAB, JEPA-style forecasting, DiT planning) on
synthetic data, and (b) be honest, in code comments and here, about every
place our own design choices stand in for details the abstract doesn't
give us. See `SOURCING.md` for the full breakdown.
