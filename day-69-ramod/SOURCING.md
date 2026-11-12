# Sourcing — what comes from the paper vs. what is mine

**Paper:** *RaMOD: Moving-Object Detection from Raw mmWave Radar Cubes under Ego Motion* —
Yuan Fang, Long Zhuang, Lingzi Xing, Xijia Wei, Jingqi Jiang, Fangzhan Shi, Simon Julier, Kevin Chetty.
arXiv [2611.13271](https://arxiv.org/abs/2611.13271), cs.RO (cross-list cs.CV), submitted 2026-11-09. No official code linked.

**How it was read:** arXiv abstract page (via a summarising fetch), plus the alphaXiv overview of the full
text. The arXiv HTML/PDF full text returned rate-limit errors for the whole session, so table cells not shown
in those two sources (per-baseline RA/RD numbers, parameter count, mAP at other IoUs) were **not** read.

## Paper facts used
| Item | Value (paper) |
|---|---|
| Input | raw range-azimuth-Doppler cubes, single-chip TI AWR2243BOOST, 10 Hz, ~±23.19 m/s unambiguous velocity, example cube 192×256×255 |
| Ego branch | frame-wise translational ego velocity from each **unmasked** cube (supervised) |
| Static mask | ego velocity → predicted Doppler support of world-static reflectors **including wrapped copies**; global static-consistent mask, not a zero-Doppler cut |
| Temporal | T = 3 suppressed cubes, encoded with their differences from the target frame |
| Detection | multi-scale features → RAD-space boxes + class (person, car, large vehicle, two-wheeler) |
| Velocity | radial + forward velocity under raw-Doppler and unwrapped hypotheses + wrapping probability; unwrapped used if p ≥ 0.5 |
| Dataset | BRIDGE-Liv, sequence-disjoint split, 10,367 test frames; metric = mAP with 3-D IoU in RAD space |
| Headline | cube mAP@0.30 **80.8%** vs Mamba-RODNet 65.1% (+15.7 pts; baseline adapted from raw-ADC by the authors) |
| Velocity | MAE 1.128 m/s (radial 1.094, forward 1.161); wrapping accuracy 99.87% |
| Speed | 25.7 FPS full loop, 107.5 FPS model-only, 149.69 GFLOPs |
| Stage-1 ablations | full 80.85 mAP / ego MAE 0.129 / vel 1.465 / wrap 99.86% · no ego supervision 74.16 / 2.755 / 1.782 / 99.83% · no wrapping loss 75.39 / 0.126 / 1.415 / 58.42% |
| Stated limits | tangential motion invisible to Doppler; forward-velocity assumes ego-aligned motion; mask can suppress slow same-direction vehicles; baseline adaptation; ablation doesn't isolate masking; not lightweight; one dataset |

## My choices (toy)
* Synthetic forward radar world (`ramod/world.py`): 40×40×32 cubes, **v_max = 4 m/s** (much smaller than the
  paper's 23 m/s so wrapping happens at city speeds in a small toy), ego up to 8 m/s, 3 classes (no large vehicle).
* Ego estimator = learned Hough voting along v_r = −v_e·cos θ (yaw rate cancels for a radar at the rotation centre).
* Soft multiplicative mask with learned width σ and depth γ; mask spectrum is also fed to the backbone.
* Boxes are 2-D range-azimuth (CenterNet) — IoU is computed in RA bins, **not** the paper's 3-D RAD IoU.
* Baselines are mine (single-frame RAD-CNN, no-suppression variant); I did not re-implement Mamba-RODNet.

## Numbers quoted in the LinkedIn post
Paper numbers: from the table above. Toy numbers: `checkpoints/metrics_s0.json` (and `_s1` for the seed re-run).
