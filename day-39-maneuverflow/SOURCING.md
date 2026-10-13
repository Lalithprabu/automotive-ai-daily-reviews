# Sourcing & disclosure — Day 39 (ManeuverFlow)

**Paper:** ManeuverFlow: Maneuver Hypothesis Generation with Future Evidence for Trajectory Evaluation and Refinement in End-to-End Autonomous Driving — Yuhao Wei, Yong Chen, Chen Yang, Yingchen Wang, Delin Ouyang, Lingfeng Qi, Qi Lan, Zhanhua Zhang, Jie Li, Guofa Li (+2 more per the mirror page). Chongqing University; Geely Automotive Research Institute. arXiv:2610.13409, cs.RO, submitted 2026-10-09 (4 days old at run time).

**Access:** `arxiv.org/abs/2610.13409` returned HTTP 429 with a do-not-retry instruction, so it was not fetched again. The alphaxiv.org mirror gave the abstract-level facts only (no layer sizes, losses, training settings or ablations). No code release found (the abstract's link was a placeholder).

## Paper-sourced (only these)
- Task: end-to-end planning; multi-query regression can collapse to one mode; anchor/generative planners face a coverage-vs-cost scaling problem.
- Pipeline (abstract-level): lightweight TransFuser-style ResNet-34 backbone; maneuver-mode selection; Gaussian-mixture prior noise; maneuver-routed mixture-of-experts MeanFlow field generating intention-specific trajectories in one step; current + world-model-predicted future features aggregated along each candidate path into trajectory-specific evidence for PDM-factor prediction; Frenet-residual refiner; joint re-scoring of refined and retained candidates.
- Results: 90.6 PDMS (NAVSIM v1 navtest), 90.1 EPDMS (NAVSIM v2 navtest), highest listed PDMS* on WorldEngine 3DGS closed loop (75.19 logged traffic, 74.58 IDM traffic).

## This repo's own reconstruction (NOT paper-sourced)
Everything else: the 3-lane synthetic world and its generator (5 maneuvers, latent driver style, hint-flag cut-ins and hard-braking leads), the transformer scene encoder (no camera backbone), all dimensions, the K-mean-trajectory prior means (sigma 0.5), the MeanFlow training recipe details (25% r=t samples, loss weights), the stand-in world model, the evidence attention design, the refiner bounds (4 m / 1 m), the synthetic PDMS-style metric, and every number in `results.json`. Frenet coordinates are the (x, y) axes of a straight road. The paper's 90.6 / 90.1 / 75.19 / 74.58 are on different data and are NOT comparable to this repo's synthetic scores.

## Findings, reported as they came out (single seed, 600 held-out synthetic scenes)
| Variant | synthetic PDMS | no-collision | minADE@6 (m) | top-1 ADE (m) |
|---|---|---|---|---|
| Old: multi-query regression | 0.312 | 0.505 | **1.57** | **4.36** |
| New: router + MeanFlow + current-only scoring, no refinement | 0.346 | 0.797 | 3.17 | 6.13 |
| + Frenet refinement and joint re-score (current features only) | 0.411 | 0.807 | 2.51 | 5.92 |
| + future evidence (full) | **0.433** | **0.837** | 2.49 | 5.70 |

- Human-path reference PDMS on the same scenes: 0.642 (no-collision 0.940). The gap to this ceiling is large for every variant.
- The old regressor imitates the sampled human path better (lower ADE/minADE); ManeuverFlow wins on safety. Imitation error and safety are different objectives here.
- Future evidence adds +0.022 PDMS and +3 points no-collision over current-only: a small, single-seed difference, not a significance claim.
- Without refinement the raw one-step samples are jagged; 19% of the picked paths leave the corridor (drivable-area 0.807). Adding the refiner and re-scoring lifts this to 0.958.
- World-model future error 0.613 m vs. 0.679 m for constant velocity (-9.7%), mean over valid agents. With a loss weight of 4 it was WORSE than constant velocity (0.787 m); raising it to 40 plus a zero-init output layer fixed that. The first training run (weight 4) gave full PDMS 0.400 vs. current-only 0.418, i.e. future evidence did not help until the world model was good enough.
- The scorer is trained on a mix of generated, GT and perturbed paths, with 30% of batches dropping future evidence so one checkpoint serves both ablations (Day 21 lesson: train both paths of an enable/disable flag).

## Bugs / lessons
1. A weakly trained world model made "future evidence" useless or harmful; fix = zero-init the residual head (starts at constant velocity) and upweight its loss. Check the evidence source beats its trivial baseline before crediting the evidence mechanism.
2. MeanFlow's identity needs `torch.func.jvp` through the whole network including the maneuver-routed experts; hard routing with `torch.where` over expert outputs works with forward-mode AD.
3. Tiny-batch overfit test (MeanFlow loss halves in 120 steps) and an exact-recovery test for the one-step sampler are in `tests/`.
