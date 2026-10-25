# SOURCING — what is paper-sourced vs. this repo's own reconstruction

Paper: **Foundation4D: Towards a unified perception and reconstruction model for autonomous vehicles**
(NVIDIA; Agostinho, Apodaca, Attia, Braso, Dong, Fedorov, Fenzi, Huang, Huang, Huang, Ke, Leal-Taixé, Lee, Liu, Liu,
Meinhardt, Ošep, Rai, Ren, Saltori, Wu, Xu, Xue, Yu, Zhang, Zhou) — arXiv:2610.30340, cs.CV, submitted 2026-10-21.

Access: only the arXiv **abstract page** was readable. `arxiv.org/html/2610.30340` returned a fetch rate-limit twice, and
web search surfaced no mirror, summary, or code release. **No full text, tables, or figures were seen.**

## Paper-sourced (from the abstract only)
- One feed-forward model, multi-camera driving video in; four tasks: geometry estimation, 3D object detection,
  semantic occupancy prediction, Gaussian scene reconstruction.
- Shared backbone with **factorized spatiotemporal attention** across cameras and time.
- Inference-time pose-graph optimisation of camera trajectories and cross-window object-track linking.
- Data-annotation pipeline + **PAI-Foundation4D** benchmark (dense depth + semantic occupancy, 1,253 driving clips).
- Claims (with benchmark-specific fine-tuning): lowest depth AbsRel on Waymo and nuScenes, highest nuScenes detection mAP,
  highest OpenOccupancy mIoU among compared methods.
- Task-combination study: co-training detection with geometry+occupancy raises top-class mAP by **1.2 points**;
  co-training geometry with detection cuts camera-to-rig translation error by **13.4%** (relative to geometry-only).
- Predicted geometry/trajectories serve as priors for lidarless NuRec (simulation reconstruction).

## NOT paper-sourced (this repo's own, disclosed choices)
Everything else: dim 64 / 4 layers / 4 heads / 4x4 patches; the spatial -> cross-view -> temporal ordering inside a
block; learned camera/frame embeddings; the depth/Gaussian/occupancy/detection head designs; the geometry-guided BEV
lifting; all losses and weights; the fully synthetic ray-cast dataset; and every number in `results.json`.

## Deliberately NOT implemented
Pose-graph optimisation, cross-window track linking, a real Gaussian-splat renderer (the Gaussian head is supervised by
splatting each Gaussian onto its own source pixel — a deliberately trivial photometric proxy), NuRec, real datasets,
multi-class semantics. Camera poses are fixed/known in the synthetic rig, so no pose head exists to test the 13.4% claim.

## Honest findings on this reconstruction
1. **Co-training benefit NOT reproduced.** Detection-only model F1 0.883 vs. joint 4-task model F1 0.884 (val n=128 scenes):
   no measurable gain. The paper's +1.2 mAP is on real data with a far larger model; this tiny synthetic task has little
   shared structure left to exploit. Reported, not tuned away. (Both models instantiate all heads, so param counts match.)
2. **Dense attention was as good or slightly better here.** Dense joint attention F1 0.900 vs. factorized 0.884, and
   ~2x faster to train — but it is **not parameter-matched** (272K vs. 672K; a factorized layer has 3 attention blocks).
   With only 384 tokens factorization saves 37% of attention pairs (92,928 vs. 147,456 per layer); the saving grows
   quickly with scale (6 cams x 8 frames x 200 patches: 6.8x). The efficiency claim is analytic here, not an accuracy win.
3. **Detection head needed a redesign.** First try (learned BEV queries + cross-attention) sat at chance (F1 ~0.3) on
   this data; geometry-guided sampling (project each BEV cell into every camera, bilinear-sample) fixed it. Same lesson
   as Day 22: geometry priors beat free queries at small data scale.
4. **Threshold calibration:** the F1-optimal threshold for the joint model landed at the grid floor (0.05), i.e. its
   heat-map is under-confident; F1 is therefore a calibrated, best-case number (Day 14 convention).
5. Simulation depth AbsRel (0.13 on the shown scene) is higher than validation (0.077) because the chosen scene's
   objects drift closer than the training distribution's 5 m minimum range.
