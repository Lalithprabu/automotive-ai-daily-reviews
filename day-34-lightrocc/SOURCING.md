# Sourcing & disclosure — Day 34 (LighTROcc)

**Paper:** *LighTROcc: Lightweight 4D Occupancy Forecasting via Instance-Centric 3D Gaussians* — arXiv:2610.09444 (cs.CV, listed 2026-10-08; exact submission date not confirmed). Authors (alphaxiv mirror): Hwanhee Jung, SeungHyeon Kim, Inkyu Koo, Qixing Huang, Sang Ho Yoon, Sangpil Kim — Korea University, UT Austin, KAIST (author-to-affiliation mapping not shown).

## Access
`arxiv.org/abs/2610.09444` was not fetched; `arxiv.org/html/2610.09444` was rejected by the fetch proxy (HTTP 429, "do not fetch again"). Other 2610.* abstract pages hit the same wall. The only paper content used is the alphaxiv.org mirror's abstract-level summary. **No result numbers were recoverable. No code release found.**

## Paper-sourced (abstract level only)
- Task: camera-based 3D occupancy forecasting; dense voxel/BEV representations get costly with resolution/horizon and track object identity poorly.
- A compact set of learned queries, one per movable object, predicts present + future occupancy in a single forward pass.
- Forward lifting: image-space cross-attention + query-specific depth + camera geometry give each query a 3D centre.
- Each instance = mixture of anisotropic 3D Gaussians; instances are propagated to future steps with predicted displacements.
- Datasets: nuScenes and a supplemented nuScenes-Occupancy. Qualitative claims: better instance-level forecasting accuracy than dense and instance-wise baselines, strong voxel-level quality, favourable accuracy/compute trade-off.

## This repo's own reconstruction (NOT paper-sourced)
Everything else: the synthetic simulator and rendered feature maps; 6 queries, 3 Gaussians/instance, d=64, layer counts; the Gaussian-mixture noisy-OR occupancy rendering; the *rigid* propagation of instance + Gaussians (translation + yaw); the loss (Hungarian matching, BCE+Dice, existence, centre, yaw, size, depth); the dense-BEV baseline; all metrics (voxel IoU, centre-matching F1/ADE at 2 m); all numbers below. 219,030 parameters — a toy.

## Honest findings (synthetic, 1 seed, 600 held-out scenes)
| Model (steps) | mean IoU +0.5..2 s | IoU +2 s | inst. F1 +2 s | centre err +2 s (m) |
|---|---|---|---|---|
| Instance Gaussians (2,500) | 0.404 | 0.305 | 0.612 | 1.06 |
| Dense BEV (2,500) | 0.235 | 0.139 | 0.263 | 1.02 |
| Dense BEV (7,500, ~equal wall-clock) | **0.433** | **0.327** | 0.611 | **0.90** |
| Instance Gaussians, displacement off | 0.227 | 0.115 | – | – |

- At equal **steps**, instance Gaussians win clearly; at roughly equal **wall-clock** (LighTROcc-style step ≈0.55 s vs dense ≈0.15 s on 2 CPU cores) the dense baseline matches or beats it on IoU and centre error. The toy does **not** show a final-accuracy advantage.
- Inference (CPU, batch 1): instance model 19.4 ms, 0.069 GFLOPs; dense 4.7 ms, 0.161 GFLOPs. The reconstruction is fewer-FLOPs but *slower* (the Gaussian→4,096-voxel evaluation is memory-bound). The paper's "lightweight" claim is not reproduced here.
- Displacement propagation matters: with displacements off, +2 s IoU falls 0.305 → 0.115.
- Not tested: real images, nuScenes, ego-motion, multiple seeds, deformable motion, multi-modal futures, crowded scenes (>4 objects), the paper's metrics (e.g. VPQ).
- Demo scenes in `simulate.py` are chosen by a non-outcome rule (>=3 vehicles, >=2 moving), not by model performance; the simulation's old-way panel uses the stronger 7,500-step dense baseline.
