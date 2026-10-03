# Sourcing & honesty disclosure — Day 29 (MapLightning)

**Paper:** *MapLightning: Online Vectorized HD Map Construction with 1D Map Tokens* — arXiv:2610.01905, submitted 2026-10-01.
Authors: Shen Zheng, Anurag Ghosh, Mani Ramanagopal, Srinivasa Narasimhan (Carnegie Mellon University). Datasets: nuScenes, Argoverse 2.

## Paper-sourced (alphaxiv overview only — `arxiv.org/abs/2610.01905` returned HTTP 429, not retried)
- A compact set of 1D learnable map tokens replaces dense BEV grids.
- Map tokens and image tokens are concatenated and processed with full self-attention; image tokens are then discarded and the updated map tokens are decoded.
- The decoder uses full (not deformable) cross-attention.
- The network needs no camera projection parameters, giving robustness to camera-extrinsic perturbations.
- Reported: 16.7x fewer intermediate tokens vs. dense BEV methods; +10.1 mAP (nuScenes) and +16.2 mAP (Argoverse 2) over MapTRv2; 1.73x faster (40+ FPS); 53% less memory.

## This repo's own reconstruction (NOT paper-sourced)
Every layer size, the CNN stem, patch size, number of map tokens (24), number of encoder/decoder layers, the DETR-style set loss, the synthetic road/camera world, the dense-BEV baseline (an inverse-perspective `grid_sample` lift, a stand-in for LSS/BEVFormer-style lifting, NOT MapTRv2), and every number in `outputs/results.json`.
None of these numbers is comparable to nuScenes/Argoverse 2 mAP.

## Findings (synthetic, 256 held-out scenes per setting, 1 training seed)
| extrinsic jitter (deg) | err old / new (m) | F1@1m old / new |
|---|---|---|
| 0 | 0.177 / 0.193 | 1.000 / 1.000 |
| 1 | 0.297 / 0.311 | 0.994 / 0.994 |
| 2 | 0.501 / 0.510 | 0.907 / 0.899 |
| 4 | 0.818 / 0.811 | 0.672 / 0.669 |
| 6 | 1.004 / 0.941 | 0.552 / 0.594 |

- **The paper's extrinsic-robustness claim is NOT clearly reproduced.** The two models are within noise up to 4 degrees; the 1D-token model is modestly better only at 6 degrees (6% lower error, +0.04 F1). One training seed: that gap could be noise.
- Likely reason (untested hypothesis): both models were trained under the same jitter distribution, and a roll-less pitch/yaw/height shift is largely unobservable from lane lines alone, so neither network can fully infer it.
- The scripted `simulate.py` drive (one 6-degree pitch bump) shows old 0.48 m vs. new 0.11 m at peak. That is ONE scenario, chosen as a demo, and is not representative of the sweep above.
- Token/param/latency (CPU, batch 1): MapLightning-style 250k params, 4.8 ms, 24 map tokens reach the decoder (72 image tokens inside the mapper); BEV baseline 226k params, 4.4 ms, 384 BEV tokens. No speed win here; the paper's 16.7x token reduction is vs. much larger BEV grids.
- Bug caught: none blocking. The simulation telemetry plot initially crashed (trace length mismatch) and was fixed.
