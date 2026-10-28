# Day 54 - SpatialBEV-style off-road dynamics learning (synthetic re-implementation)

**Paper:** *Structured Perception Representations for Off-road Dynamics Learning* - Nye, Triest, Sivaprakasam, Adu, Scherer, Wang (CMU / Penn State), arXiv [2610.22900](https://arxiv.org/abs/2610.22900), submitted 2026-10-16.

> **Disclosure.** I could only read the abstract-level summary (the full PDF/HTML was unreachable from this session). Architecture details below are *my reconstruction* from that summary. The synthetic world, ground-truth dynamics, every number in the results table and the demo are mine, **not** the paper's. Paper-reported (abstract): prediction error -17.1% vs physics-based and learned baselines; max cross-track error -25.3% in hardware tests at >=10 m/s and on geometrically hard courses.

## Idea
High-speed off-road navigation needs a good *forward dynamics model*: given state + action + terrain, where will the vehicle be next? Earlier learned models condition on local perception too simply (one cell, a global summary). SpatialBEV queries **BEV features around the predicted state at every rollout step**, so each next-state prediction sees nearby geometry and semantics.

```
s_t=(x,y,yaw,v), a_t=(steer, v_cmd)
patch  = grid_sample(BEV, ego-aligned 16x16 window at (x,y,yaw))   # re-queried at the PREDICTED state
z      = CNN(patch) -> flatten                                       # keeps WHERE things are
delta  = physics_delta(s_t,a_t) + MLP([z, state/action feats]) * scale
s_t+1  = integrate(s_t, delta)                                       # differentiable -> BPTT over the rollout
```

## Layout
```
day-54-spatialbev/
  spatialbev/terrain.py   synthetic BEV maps (height/friction/roughness) + ground-truth dynamics
  spatialbev/models.py    SpatialBEVDynamics (modes: blind | point | pooled | spatial) + physics baseline
  train.py                BPTT training, hard-case mining, evaluation -> checkpoints/results.json
  simulate.py             input vs ground truth vs prediction overlay -> assets/spatialbev_demo.{gif,mp4}
  tests/                  12 pytest checks
  assets/                 demo GIF / MP4
```
Run: `pip install -r requirements.txt && python train.py --iters 1200 --batch 64 && python simulate.py && pytest -q`

## Results (synthetic, 1 seed, 1,000 test episodes on unseen maps, 2.4 s rollouts)
| model | params | ADE (m) | FDE (m) | yaw err (deg) | hard-20% FDE (m) |
|---|---|---|---|---|---|
| physics (kinematic bicycle) | 0 | 0.99 | 2.23 | 3.5 | 4.92 |
| learned, terrain-blind | 21K | 0.80 | 1.78 | 2.4 | 2.87 |
| learned, point lookup | 26K | 0.74 | 1.64 | 1.5 | 2.51 |
| learned, CNN window + global pool | 60K | 0.12 | 0.27 | 0.7 | 0.32 |
| **SpatialBEV (CNN window, spatial flatten)** | 118K | **0.07** | **0.16** | **0.4** | **0.20** |

"Hard-20%" = test episodes mined by physics-baseline FDE (top 20%), echoing the paper's idea of mining hard scenarios.

**Read this honestly:**
- The ground truth *by construction* depends on wheel-patch traction (left/right friction -> yaw moment), so spatial structure must matter. This demonstrates the mechanism, not the paper's -17.1% figure.
- The spatial model has about 2x the parameters of the pooled one; I did not run a parameter-matched control, and only one seed.
- No closed-loop / hardware test (the paper's cross-track result) is reproduced.
- `simulate.py` shows the median episode of the hardest 20%, not a best case; in that episode the error is mostly along-track speed, and left/right friction is nearly equal, so the telemetry panel does not show a yaw pull there.
