# Day 56 - LinkDreamer-style intermittent-V2X cooperative perception (synthetic re-implementation)

**Paper:** *Agentic LinkDreamer: Adaptive Neuro-Fuzzy World-Model Decision Making for Intermittent V2X Cooperative Perception* - arXiv [2610.39258](https://arxiv.org/abs/2610.39258) (listed 2026-10-29).

> **Disclosure.** I read the abstract/architecture summary only (arXiv pages were rate-limited), not the full text. The architecture is my reconstruction; the world, dimensions, cost weights and every number below are mine, **not** the paper's. See `SOURCING.md`.

## Idea
When the V2X link drops, cooperative perception loses the collaborator's BEV features. Instead of only reconstructing them, an agent decides **what to do each frame**:
```
mode 0 ego-only | 1 motion-compensated reuse | 2 latent imagination (ConvGRU world model) | 3 low-rate refresh | 4 high-rate refresh
state x = [prediction uncertainty, perceptual complementarity, memory freshness, link quality, residual budget] in [0,1]^5
Takagi-Sugeno rules:  w_r = softmax_r(-1/2 sum_i ((x_i-c_ri)/s_ri)^2),  y_r = A_r x + b_r,  scores = sum_r w_r y_r
train: minimise  E_softmax(scores)[ detection_risk_m/ref + lc*(1+2(1-budget))*comm_m + ll*lat_m + lu*u*[m==imagine] ]
```
## Layout
```
day-56-linkdreamer/
  linkdreamer/synthetic.py  V2X world: trucks' shadows, parked/moving vehicles, pole clutter, Markov link, ego-motion warp
  linkdreamer/models.py     ConvGRUCell, LatentWorldModel, TSNeuroFuzzy, FusionDetector, build_modes, decision_state
  linkdreamer/metrics.py    peak decode, matching, AP / F1
  linkdreamer/agent.py      batch sampling, cost table, closed-loop episode runner (5 policies)
  train.py                  3 stages: world model -> detector -> controller, then closed-loop evaluation
  sweep_lambda.py           communication-weight sensitivity
  simulate.py               'AI in action' GIF/MP4: input vs ground truth vs prediction + telemetry
  tests/test_core.py        9 pytest checks
  assets/                   linkdreamer_demo.gif / .mp4 / .json      SOURCING.md
```
Run: `pip install -r requirements.txt && python train.py && python simulate.py && pytest -q`  (train ~23 min on 2 CPU cores; `--quick` ~1.5 min)

## Results (synthetic, 1 seed, 120 held-out episodes; 25.6% of frames are outage frames; AP = radius-3 m matching)
| policy | AP all | AP outage | AP link-up | comm / frame |
|---|---|---|---|---|
| ego-only | 0.656 | 0.666 | 0.652 | 0 |
| refresh when up, else reuse | 0.763 | 0.695 | 0.812 | 0.585 |
| refresh when up, else imagine | 0.763 | 0.698 | 0.812 | 0.585 |
| **TS controller (comm weight 0.10)** | 0.708 | 0.660 | 0.724 | **0.176** |
| upper ref (no outage, continuous high-rate) | 0.814 | 0.824 | 0.812 | 1.0 |

Comm-weight sweep (same test episodes): 0.0 -> AP 0.739 @ 0.44; 0.05 -> 0.739 @ 0.39; 0.10 -> 0.708 @ 0.18; 0.20 -> 0.708 @ 0.18.

**Read this honestly:** the paper's headline (92.9% of the cooperative gap recovered on outage frames) is NOT reproduced - here the controller is about -4% on outage frames and imagination only ties reuse. What does reproduce is the *trade-off shape*: a learned policy can cut communication by ~70-80% against always-refresh at a modest AP cost. One seed, no parameter-matched or statistical tests, a toy camera-free world, no reflection step.

## Core model
See `linkdreamer/models.py` (`LatentWorldModel`, `TSNeuroFuzzy`, `FusionDetector`, `build_modes`, `decision_state`).
