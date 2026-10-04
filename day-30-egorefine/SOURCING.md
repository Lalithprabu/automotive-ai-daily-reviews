# Sourcing & disclosure (Day 30 — EgoRefine)

**Paper:** EgoRefine: Ego-Referenced Predictive Alignment and Trajectory-Conditioned Reliability-Aware Fusion for Asynchronous Collaborative Perception — arXiv:2610.00319 (Hunan University: Kong, Zang, Kang, Yang, Fu, Zuo, Li; submitted 2026-09-29). Code announced at github.com/godk0509/EgoRefine (not inspected).

**Access:** arxiv.org (abs + html) returned HTTP 429 with a "do not retry" instruction; not retried. Only an alphaxiv.org overview summary was readable.

## Paper-sourced (abstract level only)
- Problem: cooperative features arrive with temporal delay -> misalignment.
- Module 1: ego-referenced predictive alignment — uses the receiver's current features to guide trajectory-field prediction and refine sampling offsets along an ego-referenced direction.
- Module 2: trajectory-conditioned reliability-aware fusion — trajectory discrepancy + refinement magnitude as alignment-quality indicators to reweight fusion.
- Benchmarks V2V4Real, DAIR-V2X-Seq; +1.6 AP@0.5 and +2.9 AP@0.25 over TraF-Align (average across datasets).

## This repo's own reconstruction (NOT from the paper)
Every dimension (C=24, K=3), the exact meaning of "direction"/"step" (unit direction x signed per-point scalar), the definition of discrepancy (feature-space |sample_ref - sample_base|) and magnitude, the gate, the global-context branch, the TraF-Align-style baseline (my simplification, not the CVPR code), the loss, the data, and every number below.
Synthetic data: 40x40 BEV, 2-6 objects with turn rate, ego sees a random disk, collaborator sees all but delayed 0-6 steps, receiver gets a noisy delay estimate (sd 0.7). Metric: centre-distance AP (1.0 / 2.0 cells), not IoU-based.
Note: all four modes instantiate the same module set (identical param count 68,427) — unused heads simply get no gradient in simpler modes.

## Results (synthetic, 700 steps CPU, one training seed)
| 3 eval seeds x 1000 scenes | AP@1.0 | AP@1.0 blind-zone objects |
|---|---|---|
| ego only | 0.789 | 0.004 |
| naive (no alignment) | 0.896 | 0.636 |
| old: TraF-Align-style | 0.899 | 0.651 |
| EgoRefine-style | 0.899 | 0.658 |

**Finding:** collaboration is worth +11 AP points; *alignment* is worth ~0.3 points overall and ~1-2 points on blind objects — inside seed noise. The paper's alignment claim is NOT clearly reproduced at toy scale. Likely reasons (untested): the conv encoder with dilations absorbs the shift implicitly; offsets learned are shorter than the true back-shift (max |offset| ~1.5 vs ~2.5 needed; zero-offset error 2.5 -> 1.8/1.85 cells). One training seed per mode.
