# Sourcing & disclosure — Day 33 (Sparse2comm)

**Paper:** *Sparse2comm: Towards Robust Cooperative 3D Object Detection* — arXiv:2610.08573, submitted 2026-10-06 (1 day old at run time). Authors (per alphaxiv mirror): Lei Yang, Boqi Li, Chunmian Lin, Li Wang, Ziying Song, Shaoqing Xu, Heye Huang, Haibao Yu, Chen Lv, +1. Affiliations not recovered.

## Access
`arxiv.org/abs/2610.08573` returned HTTP 429 ("do not fetch again") and was not read; arXiv listing pages were used for discovery. The only paper content used is the alphaxiv.org mirror's abstract-level summary. No code release found.

## Paper-sourced (abstract level only)
- Setting: cooperative 3D detection under limited bandwidth + unreliable cooperation (packet loss, transmission delay, spatial misalignment).
- Sparse Feature Encoding: collaborators transmit randomly mask-sampled foreground features; ego reconstructs dense semantic features (ultra-low bandwidth + packet-loss recovery).
- Latency-Aware Alignment: predicts motion flow to handle delayed messages.
- Self-Calibrating Fusion: estimates residual spatial offsets self-supervised, then adaptive cross-agent fusion. Order: semantic completeness -> temporal consistency -> spatial alignment.
- Reported (vs. Where2comm, mixed real-world degradations, AP@0.5 / AP@0.7): DAIR-V2X +20.15 / +11.79; OpenV2V +12.66 / +11.07; V2V4Real +15.36 / +12.61.

## This repo's own reconstruction (NOT paper-sourced)
Everything else: BEV grid 32x32, C=8 channels, hidden 24, 3-conv dilated decoder, flow net (inputs: two consecutive reconstructed messages + delay), max flow 4 cells, the soft-argmax shift-correlation-volume calibrator (range ±3 cells, step 0.5, Gaussian prior), gated fusion, heatmap head, all losses/weights (incl. the supervised flow auxiliary loss — the paper's flow supervision is unknown), the synthetic simulator (vehicles = gaussian blobs with class signatures), the centre-distance AP metric, and every number below. 55,572 parameters — a toy.

## Honest findings (synthetic, 1 seed, 1,024 held-out scenes, mixed degradations: delay 0-3, keep ratio 0.15-0.6, packet loss 0-40%, offset ±1.5 cells)
| Model | AP@r1 | AP@r2 |
|---|---|---|
| Ego only | 0.697 | 0.737 |
| Old way (zero-filled sparse msg, plain gated fusion) | 0.721 | 0.814 |
| Sparse2comm-style | **0.788** | **0.865** |

- AP vs delay (r2): d=0 old 0.854 / new 0.889; d=3 old 0.768 / new 0.818. The gain is present at every delay and bandwidth tested, but the *shape* of the paper's gains is not independently verified.
- Inference-time ablation of the trained full model (n=512, r2): full 0.857; no flow 0.831; no calibration 0.838; none of the three 0.776.
- Calibration is label-free but imperfect: mean offset error 0.52 cells vs 1.15 when uncalibrated (offset magnitude is U(±1.5) per axis).
- Mean transmitted bandwidth ≈ 2.0% of BEV cells per frame in training distribution.
- `simulate.py` clip (one scene, 28 frames): cumulative recall of ego-hidden vehicles 0.60 (new) vs 0.54 (old) — anecdotal, single scene.
- Not tested: real sensors, 3D boxes, learned (rather than norm-threshold) foreground scores, non-rigid misalignment, whole-message loss, multi-collaborator fusion, multiple seeds.
