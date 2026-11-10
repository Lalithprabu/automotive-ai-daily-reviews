# Day 66 - FreeOcc-style 4D Occupancy Forecasting (PyTorch, synthetic toy)

Reference paper: **FreeOcc: Decoupling Ego-Motion for Efficient 4D Occupancy Forecasting via Continuous Flow Matching**
(Zhang, Zhao, Metari, Laganiere; arXiv 2611.08128, cs.CV, submitted 2026-11-05). Source read: arXiv abstract + an
alphaxiv machine-generated overview (arXiv HTML was rate-limited). Equations, loss weights and training setup were NOT seen.

> This is a **compact BEV re-implementation of the paper's described mechanism on a synthetic world** (78,409 params; paper: 28.3M).
> It is not the authors' code, not nuScenes, not the paper's numbers.

## Structure
```
day-66-freeocc/
  models/freeocc.py      ConvGRU cell, FlowNet (velocity U-Net), FreeOcc (warp + residual GRU + decoder + flow + blend)
  src/geometry.py        SE(2) pose algebra + warp_bev (affine_grid / grid_sample)
  src/synthetic.py       toy world: static + moving discs, ego arc, BEV rasteriser, history corruption
  train.py               training + evaluation (mIoU per horizon, baselines, blends, robustness)
  simulate.py            OpenCV simulation: input vs ground truth vs prediction + telemetry (mp4 / --show live window)
  tests/test_all.py      7 tests (pose inverse, warp-vs-renderer, shapes, zero-init=pure warp, no-future-leak, FM grad, sampler determinism)
  assets/demo.mp4  assets/demo_last.png    checkpoints/*.pt  *_metrics.json
```
## Run
```
pip install -r requirements.txt
python -m pytest tests -q
python train.py --steps 1000                         # ~10 min on 1 CPU thread
python train.py --steps 1000 --no-warp --out checkpoints/freeocc_nowarp.pt
python simulate.py --ckpt checkpoints/freeocc.pt --out assets/demo.mp4 [--show]
```
## Mechanism (paper -> this repo)
1. Encode history BEV occupancy; align each frame to the current ego frame with recorded odometry; **timestamp-aware ConvGRU** fuses them (no attention).
2. Take the **planner's intended future ego pose**; **SE(2)-warp** the latent state into each future ego frame (ego-motion removed analytically).
3. A ConvGRU + **zero-initialised residual projector** learns only the residual scene dynamics (starts as pure warp).
4. Decoder head -> occupancy logit + per-voxel **log-variance**; a **conditional flow-matching** branch (conv U-Net velocity field, Euler ODE) runs from noise conditioned on the same features.
5. **Uncertainty-aware blend**: w = sigmoid(a*(logvar-b)), p = (1-w)*p_dec + w*p_ode.

Repo-own choices (not paper): 2-D BEV 48x48 @ 1 m, d=24, 4 Euler steps, loss weights, heteroscedastic-BCE for logvar, smoothness + mass-conservation regularisers (no ray-traced free-space term), blend parametrisation, synthetic data.

## Honest toy results (synthetic, 1 seed, 300 held-out scenes, 6 horizons x 0.5 s, IoU of occupied cells @ p>0.5)
| Variant | mIoU | notes |
|---|---|---|
| Warp-only baseline (copy newest frame, ego-warp) | 0.214 | no learning |
| FreeOcc decoder only | 0.308 | IoU 0.50 @ +0.5 s -> 0.21 @ +3.0 s |
| FreeOcc flow (ODE) only | 0.226 | one stochastic sample |
| FreeOcc uncertainty blend | 0.308 | mean blend weight ~0.00 here: **no gain over decoder** |
| Max / mean blend | 0.280 / 0.280 | |
| No ego-motion warp (separate run) | 0.000 | best-threshold 0.04: cannot localise; collapses to empty map |
| Robust: 30% history-frame dropout + 0.08 s timestamp jitter | decoder 0.248 (80% of clean), blend 0.248, warp-only 0.210 | paper reports 92.8% retained |

Caveats: the toy is hard (3% voxel flips, small discs, 1,000 steps), so absolute IoU is low; the no-warp collapse is far larger than the paper's
32.1 -> 25.8 because this net has no other way to learn planner-conditioned geometry; the flow branch / blend did not help here
(the paper's ablations also do not isolate them). Not comparable to nuScenes. No NFCD implemented.
