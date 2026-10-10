# Sourcing & disclosure — Day 36 (MESSENGER)

**Paper:** *MESSENGER: Memory-Enhanced Sequential Scene Flow Estimation via Autoregressive Next-Frame Forecasting* — arXiv:2610.10759 (cs.CV), submitted 2026-10-07 (v1). Authors: Jiuming Liu, Jianing Li, Mengmeng Liu, Hongyang He, Hesheng Wang, Per Ola Kristensson (Cambridge / SJTU / Twente / Warwick, per the mirror; per-author mapping not shown).

## Access
`arxiv.org/abs/2610.10759` returned HTTP 429 (do-not-retry). Only the alphaxiv.org mirror's abstract-level summary was read. **No full text, no layer-level architecture, no loss functions, no results tables. Code "will be released" — no URL available.**

## Paper-sourced (abstract level only)
- Pairwise two-frame scene-flow estimators lack long-term temporal correlation and extrapolate poorly; existing multi-frame methods are costly and degrade over long horizons.
- Sequential pipeline with a **memory buffer** of multiple history flow estimates and latent states.
- At each frame, stored flows/states are **correlated and retrieved** to **initialise** the current flow in a **next-frame forecasting** setup.
- An **uncertainty-aware reweighting** module filters unreliable retrievals and limits error accumulation.
- Datasets nuScenes and Argoverse 2; long-horizon future-extrapolation EPE3D reduced **71.6%** (nuScenes) and **67.7%** (Argoverse 2). (Baseline and exact protocol not recoverable.)

## This repo's own reconstruction (NOT paper-sourced)
Everything else: 2-D (not 3-D) points; the dense-attention pair estimator; the chained-flow retrieval; kNN soft matching; gate MLP inputs; the residual forecast head; the Laplace log-sigma head; d=48, K=3; the synthetic simulator (2–4 box-outline vehicles, sinusoidal accel/yaw-rate, resampled points, 192 pts/frame); the "old way" definition (same code, K=1, no gate); every number below. 68,952 params (old way) / 82,264 (K=3) — toys.

## Honest findings (synthetic, 1 seed, 300 test scenes, 600 steps each, CPU)
EPE in metres, moving points, rollout at frame t=5, horizons +0.5…+2.0 s:

| Horizon | CV carry (no learning) | Old way K=1 | MESSENGER-style K=3 | Reduction vs old way |
|---|---|---|---|---|
| +0.5 s | 2.36 | 2.24 | 2.11 | 6.0% |
| +1.0 s | 4.77 | 4.57 | 4.17 | 8.6% |
| +1.5 s | 7.22 | 6.99 | 6.34 | 9.3% |
| +2.0 s | 9.72 | 9.54 | 8.72 | 8.5% |

- The gain is real but modest (6–9%), far below the paper's 67–72% — the paper's baselines, data, and metric differ; do not compare.
- **Negative finding:** on *static* points the K=3 model is worse (EPE@+2.0 s 0.49 vs 0.22 m) — memory-conditioned residuals leak small spurious motion into the background.
- **Negative finding:** the reweighting gate did NOT learn to separate reliable from unreliable retrievals — mean gate on moving vs static points is nearly equal (≈0.49–0.64 across ages). The "uncertainty-aware reweighting" here is architecture-present but not demonstrably doing its job.
- Absolute errors are large (≈2 m at +0.5 s): the two-frame estimator is limited by the aperture problem on resampled box outlines (pair-flow L1 ≈ 0.9–1.0 m), which caps what any memory can fix.
- Not tested: real data, multiple seeds, 3-D, the paper's metrics, runtime, crowded scenes, sensor noise beyond σ=0.05 m.
