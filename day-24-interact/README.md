# INTERACT reconstruction: Anchor-Conditioned Prediction + Trust-Region CEM

Day 24 of a daily automotive-AI technical review series. This repo is an
**independent, clearly-disclosed reconstruction** of the mechanism described in:

> **INTERACT: Interactive Planning for Autonomous Driving via Anchor-Conditioned
> Prediction and Trust-Region Refinement** -- Distelzweig, Look, Janjoš, Hagedorn,
> Palmieri, Boedecker (Bosch / Uni Freiburg / Coburg). arXiv:2609.31137 (2026-09-25).

**Read [`SOURCING.md`](SOURCING.md) first.** The paper's full text was not
accessible (arXiv 429 rate-limit at fetch time). Only the handful of claims in the
abstract are paper-sourced; every architecture detail, hyperparameter, dataset, and
number below is this project's own reconstruction, run and measured for real.

## The idea, in one paragraph

Planning around other drivers is a chicken-and-egg problem: what they'll do depends
on what you do. Querying an expensive interaction-aware predictor once per
*candidate* ego trajectory (hundreds of forward passes during optimization) is
accurate but slow. INTERACT's insight (per its abstract): other agents react to
your *intent*, not your exact path -- so predicting once per **anchor** (a small
set of diverse intents, e.g. "merge assertively" vs. "yield") and reusing that
forecast across every candidate trajectory sharing that intent is both cheap and
still genuinely reactive.

## Architecture (this repo's reconstruction)

```
Ego history [H,4]  ─┐
Other-agent hist ───┼──►  AnchorConditionedPredictor  ──►  other-agent future [T,2]
Anchor cond [2]  ────┘     (GRU encoder + FiLM(cond)          (queried ONCE per anchor)
                             + cross-attn decoder)
                                     │
                                     ▼ cached, reused across whole CEM population
                          TrustRegionCEM
                          sample candidates ─► cost(collision, comfort,
                          progress, trust-region) ─► CEM mean/cov update
                                     │
                                     ▼
                     best-of-population trajectory per anchor
                                     │
                                     ▼
                        argmin CEM cost across K=5 anchors
                              = committed plan
```

See `assets/architecture.png` (rendered by `make_assets.py`, this repo's own
diagram -- not the paper's figure).

### Core files
- `src/models/anchor_generator.py` -- derives K=5 anchors from simple merge
  geometry (gap-to-merge, lane width).
- `src/models/anchor_conditioned_predictor.py` -- **the centerpiece module.** GRU
  history encoder, FiLM conditioning on the 2D anchor/intent vector, cross-attention
  decoder unrolled over the future horizon. Verified by unit test to produce
  genuinely different outputs for different anchors given the same history.
- `src/models/trust_region_cem.py` -- CEM refiner with an explicit trust-region
  penalty term, and two query modes (`cached` vs `naive`) used to build the
  efficiency ablation below.
- `src/data/synthetic_interactive_scenario.py` -- a synthetic merge/negotiation
  simulator where the other agent's future genuinely, falsifiably depends on the
  ego's committed intent.

## Old vs. New, built as a real ablation (not just described)

| | Predictor queries | What it captures |
|---|---|---|
| **Old Way A** -- non-interactive predict-then-plan | **1** (unconditional model, cond forced to zero at train+eval) | fixed, non-reactive forecast reused for every candidate/anchor |
| **Old Way B** -- naive per-candidate re-query | **N x n_iter** per anchor (here: 32 x 4 = 128) | fully reactive per candidate, but expensive |
| **New Way (INTERACT)** | **1 per anchor** (here: K=5) | reactive at the anchor level, reused across the whole CEM population for that anchor |

All three share the *same* trained `AnchorConditionedPredictor` weights where it's
fair to do so (Old Way B literally re-queries that same network on each candidate's
own intent vector); Old Way A uses a separately-trained unconditional instance of
the identical architecture, since "ignore the intent" is fundamentally a different
training target, not just a different inference-time trick.

## Real, measured results (this repo's own synthetic scenario -- not the paper's numbers)

**Training** (`train.py`, CPU, 18 epochs, 3000 train / 600 val synthetic episodes):

| | conditioned model (New Way) | unconditional model (Old Way A) |
|---|---|---|
| final train loss (MSE) | 0.001712 | 0.035412 |
| final val loss (MSE) | 0.001732 | 0.033483 |
| wall time | **46.3s total** for both models, 18 epochs | |

**Held-out bucketed evaluation** (anchor-sampled validation set, n=1250, split by
`|assertiveness| > 0.3` = "interactive" vs. "neutral"):

| bucket | n | conditioned ADE | unconditional ADE | ADE improvement |
|---|---|---|---|---|
| interactive | 985 | 0.0514 | 0.3163 | **+83.7%** |
| neutral | 265 | 0.0511 | 0.0599 | +14.7% |
| all | 1250 | 0.0513 | 0.2619 | +80.4% |

This is the repo's core, falsifiable empirical claim, regression-tested in
`tests/test_regression_core_claim.py`: conditioning on the ego's intent helps a
little everywhere, but helps **~5.7x more** specifically where the scenario is
genuinely interactive -- mirroring (on a toy synthetic scenario, not nuPlan/
interPlan) the paper's qualitative claim that gains concentrate in interactive
scenarios.

**Simulation** (`simulate.py`, one fixed merge scenario, K=5 anchors):

| | value |
|---|---|
| New Way predictor calls | **5** (1 per anchor) |
| Old Way A predictor calls | **1** (fixed, non-reactive) |
| Old Way B naive calls, single anchor | **128** (4 CEM iters x 32 candidates) |
| Old Way B naive calls, all 5 anchors (extrapolated) | **640** (128x more than New Way) |
| Mean forecast ADE, New Way | **0.0472** |
| Mean forecast ADE, Old Way A | **0.2706** |
| Mean ADE reduction, New Way vs Old Way A | **82.6%** |
| Committed anchor this run | `accelerate_past` (lowest CEM cost) |

Per-anchor forecast error (New Way vs. Old Way A), most dramatic exactly where the
ego commits to something assertive (`aggressive_merge`, `decelerate_follow`) --
these are precisely the scenarios where a non-reactive forecast is most wrong:

| anchor | assertiveness | New Way ADE | Old Way A ADE |
|---|---|---|---|
| aggressive_merge | +0.90 | 0.0349 | 0.3953 |
| accelerate_past | +0.65 | 0.0537 | 0.2875 |
| hold_lane_delay | +0.00 | 0.0471 | 0.0590 |
| cautious_yield | -0.45 | 0.0404 | 0.1885 |
| decelerate_follow | -1.00 | 0.0597 | 0.4228 |

See `assets/results.png` for the rendered stat-tile summary and
`assets/simulation.gif` for the full animated walkthrough (anchors fanning out,
then the committed plan playing back against ground truth with a live forecast-
error telemetry panel).

## Folder structure

```
day-24-interact/
├── README.md
├── SOURCING.md
├── requirements.txt
├── config.yaml
├── conftest.py
├── src/
│   ├── models/
│   │   ├── anchor_generator.py
│   │   ├── anchor_conditioned_predictor.py
│   │   └── trust_region_cem.py
│   ├── data/
│   │   └── synthetic_interactive_scenario.py
│   └── utils/
│       └── kinematics.py
├── train.py
├── simulate.py
├── make_assets.py
├── tests/
│   ├── test_anchor_generator.py
│   ├── test_predictor.py
│   ├── test_cem.py
│   └── test_regression_core_claim.py
├── checkpoints/          (produced by train.py: *.pt, metrics.json)
└── assets/               (produced by simulate.py / make_assets.py)
    ├── architecture.png
    ├── results.png
    ├── simulation.gif
    └── simulation_summary.json
```

## How to run

```bash
pip install -r requirements.txt

# 1. Unit + regression tests (~10s)
pytest tests/ -v

# 2. Train both predictors on the synthetic scenario (~45s on CPU)
python3 train.py --config config.yaml

# 3. Run the full anchor -> predict -> CEM pipeline on one scenario, render the GIF
python3 simulate.py --config config.yaml

# 4. Render the architecture + results PNGs from the real run data
python3 make_assets.py
```

## Known limitations (candid, per this project's standing honesty norm)

- **2D state simplification.** Trajectories here are (lateral offset, speed) pairs,
  not full (x, y, heading) poses -- enough to demonstrate the anchor-conditioning
  mechanism cleanly, but not a full BEV planner.
- **One synthetic scenario family.** The reactive rule is hand-authored (linear-ish
  in assertiveness with noise); a real interactive predictor must handle far more
  scenario diversity, multi-agent interactions, and non-linear/discontinuous
  reactions (e.g. sudden braking) that this toy simulator does not attempt.
  `hold_lane_delay` (the "neutral" anchor) shows the *smallest* measured
  improvement (+14.7% vs. +83.7% for interactive anchors) -- exactly the
  regime where a non-reactive baseline is a weaker strawman, and worth keeping in
  mind: the reported gains are largest where the ground-truth reaction is strongest
  by construction.
- **Efficiency numbers are illustrative, not wall-clock-optimized.** `128x` fewer
  predictor calls is real and measured, but this repo's model is tiny (hidden_dim=
  32); the gap would look different (likely larger, since larger predictors amplify
  the per-query cost) on a production-scale network.
- **No real benchmark comparison.** Nothing here is evaluated on nuPlan or
  interPlan; the paper's own SOTA claim on those benchmarks is unverified by this
  repo and reported only as a qualitative, disclosed claim (see `SOURCING.md`).
- **Trust-region radius is a fixed hyperparameter**, not learned or adapted per
  scenario; a candidate right at the boundary can flip from "trusted" to
  "penalized" for a small perturbation.
