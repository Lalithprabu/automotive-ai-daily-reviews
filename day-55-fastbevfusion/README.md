# Day 55 - FastBEVFusion-style LiDAR+camera fusion (synthetic re-implementation)

**Paper:** *FastBEVFusion: Simple Multi-Modal Fusion for 3D Object Detection at 50 FPS* - Sanitz, Gulsoylu, Frintrop, Wilms, arXiv [2610.37153](https://arxiv.org/abs/2610.37153), submitted 2026-10-27.

> **Disclosure.** I read the abstract plus tool-generated summaries of the full text, not the PDF itself. Architecture is my reconstruction; the world, dimensions, head and all numbers below are mine, **not** the paper's. See `SOURCING.md`.

## Idea
Skip per-pixel depth. Copy each camera feature along its ray at uniform depth -> an image BEV that is *semi-aligned* with the LiDAR BEV (right direction, unknown range). One cross-attention block (LiDAR tokens = queries, image tokens = keys/values) + MLP + residual lets each LiDAR cell find the matching image cells.
```
img [B,3,W] -> Conv1d -> [B,C,W] -> UniformDepthProjector(yaw) -> image BEV [B,C,G,G]
lidar [B,2,G,G] -> Conv2d -> LiDAR BEV [B,C,G,G]
tokens: stride-2 conv -> [B,(G/2)^2,C] + sin/cos pos-enc;  Q=LiDAR, K=V=image;  x = LiDAR + Up(MLP(Attn))  -> heatmap head
```
## Layout
```
day-55-fastbevfusion/
  fastbev/synthetic.py   LiDAR (range-sparse + tall pole clutter) + 1-D camera (decoy posters) world, heatmap targets
  fastbev/models.py      UniformDepthProjector, CrossAttnFusion, FastBEVFusion (lidar_only|camera_only|add|concat|xattn)
  fastbev/metrics.py     peak-pick decode, matching, AP / F1 / near-far recall
  train.py               3-stage recipe + ablation table -> checkpoints/results.json
  simulate.py            camera strip + BEV overlay: input vs ground truth vs predictions, attention shading, F1 telemetry
  tests/test_core.py     8 pytest checks
  assets/                fastbev_demo.gif / .mp4
```
Run: `pip install -r requirements.txt && python train.py && python simulate.py && pytest -q`  (train ~7 min on 2 CPU cores)

## Results (synthetic, 1 seed, 800 test scenes; AP = matching radius 3 m)
| model | params | AP clean | F1 clean | far recall | AP @ 2.5 deg extrinsic error |
|---|---|---|---|---|---|
| LiDAR only | 24.8K | 0.705 | 0.568 | 0.169 | 0.705 |
| camera only | 24.8K | 0.462 | 0.248 | 0.081 | 0.443 |
| LiDAR + addition | 24.8K | 0.848 | 0.735 | 0.500 | 0.692 |
| LiDAR + concat | 26.9K | 0.853 | 0.740 | 0.508 | 0.705 |
| **LiDAR + cross-attention (3-stage)** | 51.9K | 0.803 | 0.655 | 0.337 | **0.793** |
| cross-attention, no stage 3 | 51.9K | 0.799 | 0.645 | 0.319 | 0.790 |

Fusion block alone: ~1.0 ms (CPU, batch 1, 16x16 tokens).

**Read this honestly:** fusion clearly helps over LiDAR-only; the paper's *robustness* direction reproduces (attention loses 0.01 AP under calibration error, addition/concat ~0.15); but on clean data simple addition/concat won here, attention has 2x the parameters (no parameter-matched control), stage 3 added almost nothing, one seed, and the toy camera is perfectly azimuth-aligned. The paper's speed (17 ms, 59 FPS) and nuScenes numbers are NOT reproduced.

## Core model
See `fastbev/models.py` (`UniformDepthProjector`, `CrossAttnFusion`, `FastBEVFusion`).
