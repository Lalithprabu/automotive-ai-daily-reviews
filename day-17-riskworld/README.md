# Day 17: RiskWorld -- Risk-Aware World Modeling with Flow-Guided Occupancy Evolution

An original, from-scratch PyTorch reconstruction of the mechanisms described in:

> **Risk-Aware World Modeling with Flow-Guided Occupancy Evolution for Selective Trajectory Planning in Automated Driving**
> Rongxiang Zeng, Linsen Cai, Jiafu Zhang, Yijie Zhong, Yide Tao, Shuai Wang, Nan Zheng, Hai L. Vu, Alvaro Garcia Hernandez, Yongqi Dong
> arXiv:2609.18442, submitted 2026-09-16 (cs.AI, cross-listed cs.ET, cs.LG, cs.RO, eess.SY), 8 pages, 2 figures

**Paper abstract (verbatim, recovered from the arXiv abstract page):**

> "This work presents RiskWorld, a framework for autonomous vehicle planning that integrates risk assessment with occupancy forecasting. The system combines spatial risk fields and temporal actor context with visual bird's-eye-view (BEV) features to predict traffic evolution. Key features include flow-guided transport of occupancy information, collision-score comparisons against baseline trajectories, and selective trajectory replacement when predicted risks warrant intervention. Testing on the nuScenes dataset demonstrated lowest collision rate at a long evaluation horizon of 3s while achieving competitive L2 error metrics, running at 11.5 FPS on an RTX 4090 GPU with 90.81M parameters."

---

## Reconstruction & sourcing note (read this first)

This repository is **this project's own original implementation**, written to capture the *mechanism* the paper describes -- it is **not** copied from the paper's own code, because no source code or full-text PDF/HTML was retrievable (repeated HTTP 429s from this session's own web-fetch proxy, not arXiv itself, blocked full-text access -- a known recurring issue in this daily series).

What is actually paper-sourced (safe to cite as the paper's claims):

- The four named components: spatial risk fields, temporal actor context, visual BEV features, and flow-guided occupancy transport.
- **Flow-guided evolution transports occupancy and scene features, with signed residuals correcting occupancy after transport** (from a secondary alphaxiv.org summary, not the paper's own text).
- **One occupancy forecast is generated per planning step and reused across all trajectory candidates** (efficiency detail, same secondary source).
- **Each candidate receives a nonnegative collision-score correction**, computed against **a current-state persistence reference** (i.e. "assume the world stays exactly as last observed").
- **Selective replacement**: the nominal trajectory is replaced only when its predicted risk triggers intervention **and** an alternative exists that satisfies *component-wise* constraints on both predicted risk and trajectory error.
- Reported results: lowest collision rate among baselines at the 3-second horizon, second-best average L2 error, 11.5 FPS on an RTX 4090, 90.81M parameters (nuScenes, open-loop planning evaluation).

Everything else below -- every architecture detail, dimension, loss function, dataset, and default hyperparameter -- is **this repo's own reconstruction default**, flagged inline in each module's docstring as such. In particular:

| Choice | This repo's default | Paper detail available? |
|---|---|---|
| BEV backbone | small stride-preserving CNN with a residual skip | No -- paper only says "visual BEV features" |
| Temporal actor encoder | single-layer GRU per agent, scattered onto the BEV grid at the agent's last cell | No -- paper only says "temporal actor context" |
| Risk field head | 2-layer conv + sigmoid, per-cell scalar in [0,1] | No |
| Flow warp | `grid_sample` bilinear warp of the previous occupancy grid, flow displacement capped at ~4 cells/step via `tanh` | No -- only "flow-guided transport" is stated |
| Residual correction | second conv head, `tanh`-bounded signed residual added post-warp | Partially -- "signed residual" is paper-sourced; the head architecture/scale is not |
| Collision score | bilinear-sample occupancy + risk field along each candidate's waypoints, averaged over the horizon | No -- only that a score + nonnegative correction exist |
| Selective-replacement tie-break | among qualifying candidates, pick the lowest-risk one | Paper states the two gate conditions, not the tie-break |
| Losses (MSE on occupancy + risk target) | this repo's own training objective | Not stated in the abstract at all |
| Model size (~59K params here) | intentionally CPU-scale for a runnable demo | The real paper model is 90.81M params -- **not reproduced here**, only quoted as reported |

**Do not read the numbers this repo produces as validation of the paper's own 11.5 FPS / 90.81M-parameter / nuScenes results.** They are separate, synthetic-data numbers from a much smaller reconstruction, reported for transparency about what this repo actually demonstrates (see "Results" below).

---

## Architecture

```
BEV raster (4ch) ──► BEVEncoder (CNN) ──► bev_feat (C,H,W) ─┐
                                                              ├─► fusion (1x1 conv) ─► fused_feat
agent history (A,T,4) ─► TemporalActorEncoder (GRU) ─► scatter ─┘
                                                                     │
                                                                     ├─► SpatialRiskField ─► risk_field (1,H,W)
                                                                     │
                              [fused_feat ⊕ risk_field] ─► FlowGuidedOccupancyEvolution
                                     (per step t, autoregressive over the horizon)
                                        │             │
                                  flow field    signed residual
                                        │             │
                       prev_occupancy ─┴─ warp (grid_sample) ─┴─► occupancy_t (clamped to [0,1])

  ── planning time (not trained, no learnable params) ──
  candidate trajectories ─► CollisionScoreModule (samples forecast vs. persistence occupancy + risk field)
                          ─► forecast_score, persistence_score, correction = max(0, forecast - persistence)
                          ─► select_trajectory(): replace nominal iff risk(nominal) > tau_risk
                                AND exists a candidate with risk < tau_risk AND deviation < tau_dev
```

## Simulation

`simulate.py` loads a real trained checkpoint, searches synthetic hazard scenarios (a cross-traffic agent that only becomes visible/relevant in the last observed frames, then crosses into the ego's lane within the forecast horizon -- exactly the regime a persistence baseline misses), runs the full model + planning pipeline, and renders an animated GIF with:

- the BEV input scene,
- ground-truth future occupancy,
- the persistence-baseline occupancy (naive, unwarped),
- RiskWorld's flow-warped + residual-corrected forecast occupancy, with the nominal trajectory (and the replacement candidate, if the gate fires) overlaid,
- a live telemetry sub-panel plotting the running forecast/persistence risk score and the collision-score correction as the horizon unfolds.

## Results (this repo's own synthetic-data numbers -- NOT the paper's)

Trained 12 epochs, CPU, 256 synthetic training scenes / 48 validation scenes, batch size 8, `~59K` parameters:

```
loss[0] (epoch 1)  = 0.024406
loss[-1] (epoch 12) = 0.005108

Validation (occupancy-forecast MSE vs. ground truth):
  hazard scenes (n=28):     RiskWorld forecast MSE = 0.00263   persistence-baseline MSE = 0.00692
  non-hazard scenes (n=20): RiskWorld forecast MSE = 0.00143   persistence-baseline MSE = 0.00369
```

RiskWorld's forecast beats the naive persistence baseline by ~2.6x MSE on the hazard scenario family specifically -- consistent with (but on synthetic data, not proof of) the paper's stated motivation for flow-guided evolution over a "hold state static" forecast.

## Repository layout

```
day-17-riskworld/
  README.md
  requirements.txt
  config.yaml
  src/
    bev_encoder.py          # BEVEncoder
    actor_encoder.py         # TemporalActorEncoder
    risk_field.py              # SpatialRiskField
    flow_occupancy.py           # FlowGuidedOccupancyEvolution (warp + signed residual)
    collision_score.py           # CollisionScoreModule (forecast vs. persistence + correction)
    trajectory_selection.py        # select_trajectory() -- pure gating function
    planning.py                      # candidate generation + plan_step() convenience wiring
    riskworld_model.py                 # top-level RiskWorld nn.Module (encode -> risk -> occupancy)
    dataset.py                           # synthetic BEV scenario generator
    utils.py                               # config loading, seeding, grid coordinate helpers
  train.py                    # trains the model, prints loss curve, saves checkpoints/riskworld.pt
  simulate.py                  # renders outputs/riskworld_simulation.gif from a trained checkpoint
  tests/
    test_modules.py            # shape/finiteness/range checks per module
    test_trajectory_selection.py  # gate-condition unit tests
    test_overfit.py               # overfit-one-batch sanity test
```

## Running it

```bash
pip install -r requirements.txt
pytest tests/ -v
python train.py            # or: python train.py --epochs 4 --steps 50 for a quick smoke test
python simulate.py
```

## Known limitations of this reconstruction

- Model capacity (~59K params) is far below the paper's reported 90.81M -- this repo optimizes for a CPU-runnable few-minute demo, not for matching the paper's reported speed/accuracy numbers.
- The synthetic dataset is a simplified rasterized intersection, not nuScenes; results here say nothing about real-world performance.
- The flow field's displacement is capped (`tanh`-bounded to ~4 cells/step) as a training-stability safeguard -- this repo's own choice, not paper-sourced.
- Candidate trajectory generation (fixed lateral "swerve" offsets + one "slow down" candidate) is a placeholder planner, not a learned or sampling-based trajectory generator.
- Risk-field supervision uses a simple proxy target (max future occupancy) rather than any learned or paper-specified risk definition.
