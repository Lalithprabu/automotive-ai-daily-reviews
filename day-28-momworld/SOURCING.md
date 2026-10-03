# Sourcing & honesty disclosure — Day 28 (MomWorld)

**Paper:** MomWorld: Momentum-Aware Latent World Model for Long-Horizon Autonomous Driving — arXiv:2609.33737, submitted 2026-09-30.
Authors (first 9 of 12 shown on the mirror): Ziying Song, Shengkai Zhang, Lei Yang, Haozhuang Chi, Yuchen Liu, Jiangtao Su, Lin Liu, Ziyang Liu, Chen Lv. Affiliations: NTU, BJTU, North University of China, DUT, Tsinghua.

## Paper-sourced (alphaxiv overview only — arxiv.org/abs returned HTTP 429, no retry)
- Problem: existing methods struggle to propagate motion trends from observed history into the future; long rollouts from a single latent state lose motion information and destabilise near-term planning.
- Mechanism names: latent momentum extracted from history and propagated, predicting both configuration and momentum; Momentum Persistence; Scene-Conditioned Momentum Updates; Scene-Adaptive Reset Gate (suppresses outdated momentum on abrupt scene changes); **MoFlow** — momentum-conditioned flow-matching refiner, few integration steps, horizon-aware residual fusion.
- Benchmarks: NAVSIM, nuScenes, Bench2Drive.
- One number: **12.2% lower average collision rate vs. MomAD over a 6 s planning horizon.**

## This repo's own reconstruction (NOT paper-sourced)
Every dimension, the gate/update equations, the attention-over-differences momentum extractor, the Conv1d flow network, the loss weights, the synthetic world, and every number in `outputs/results.json`. The old-way baseline is a single-latent-state rollout of my own design — **not MomAD**. Not comparable to the paper's benchmarks.

## Findings (reported, not tuned away)
| Variant (3 seeds, synthetic test set) | ADE (m) | FDE (m) | Collision, event scenes |
|---|---|---|---|
| Stale-momentum extrapolation (non-learned) | 1.856 | 6.076 | 83.0% |
| Single-latent baseline (old way) | **0.423 ± 0.013** | **0.888** | 0% |
| MomWorld-style, full | 0.462 ± 0.036 | 0.962 | 0% |
| MomWorld-style, no MoFlow | 0.499 ± 0.027 | 0.999 | 0% |
| MomWorld-style, no reset gate | 0.464 ± 0.005 | 0.975 | 0% |

1. **The paper's headline claim is NOT reproduced at toy scale.** The single-latent baseline is slightly *better* than the momentum model (ADE 0.423 vs 0.462). A GRU's final hidden state already encodes the acceleration trend here, so a single-latent rollout does not "forget" it in this small world.
2. **The reset gate barely discriminates.** Mean keep-value is 0.495 on event scenes vs 0.465 on free scenes; removing the gate changes ADE by 0.002 m. I would not claim a learned scene-adaptive reset from this evidence.
3. **MoFlow helps a little**: 0.499 → 0.462 ADE for the momentum model (the largest effect in the ablation table, still small).
4. **Collision rate is saturated**: every learned model has ~0–3% collisions; only the non-learned stale-momentum extrapolation collides (83% in event scenes). It illustrates the failure mode the paper targets, but it is a strawman a learned model easily beats.
5. **Bug found and fixed:** MoFlow's horizon-aware fusion weights φ were never trained (they only appear in inference-time `refine`, not in the flow loss), so they sat at exactly 0.5. This silently handicapped any variant with MoFlow and produced a spurious "MomWorld 0.51 vs baseline 1.78 m" win in my first full run. After adding a fused-output loss, the baseline improved to 0.42 m and the gap vanished. Regression test: `test_gradients_reach_history_encoder_and_momentum_extractor` asserts `phi.grad` is non-zero. Learned fusion weights end at 0.57–0.65 (near flat across the horizon — no clear near-term-conservative / long-term-aggressive profile).
6. Gains/losses of ~0.04 m are within seed noise (3 seeds). Synthetic world, 113K parameters, single scene type.
