# Sourcing & reconstruction disclosure — Day 23 (PriorMapBEVNet)

**Paper:** "Leveraging Vision-Based Point Cloud Map Priors for Camera-Based 3D
Object Detection and Online Vectorized HD Mapping," Markus Käppeler, Rohit
Mohan, Abhinav Valada (University of Freiburg). arXiv:2609.26325, submitted
2026-09-22. Comment: IROS 2026 Workshop on Long-Term Perception for
Human-Centric Autonomy. Subjects: cs.CV, cs.RO.

## What was actually retrieved

`arxiv.org/abs/2609.26325` was fetched successfully and directly: full
title, complete 3-author list, submission date, venue/workshop, subject
categories, and the full verbatim abstract.

`arxiv.org/html/2609.26325` (full text) was rejected with an HTTP 429 and
an explicit "do not retry" instruction — the same recurring wall hit on
most days since Day 9 of this series. A mirror (alphaxiv.org) was checked
and returned only the same abstract-level description already recovered
from the abstract page itself; no additional architectural detail, no
dimensions, no loss functions, no ablation tables, and no dataset-split
detail were recovered from any source. No public code release was found.

## Paper-sourced (verbatim, from the abstract)

- The general pipeline shape: a static point-cloud prior map built from
  previous camera traversals using **Pi3X**, augmented with **DINOv3**
  per-point features; retrieved via localization and encoded with a
  **sparse voxel backbone**; fused with live multi-camera features **in
  bird's-eye view**; decoded by **task-specific transformer heads** for 3D
  object detection and online vectorized HD mapping.
- The two headline numbers, both on the **Argoverse 2** benchmark:
  3D object detection **0.287 → 0.299 CDS**; vectorized mapping
  **0.669 → 0.750 mAP** (with vs. without the prior-map branch).
- The paper's own framing of the contribution: camera-only, LiDAR-free
  construction *and* inference, addressing depth ambiguity via long-term
  scene memory.

## This repo's own reconstruction default (NOT paper-sourced)

Everything below is disclosed as this project's own design choice, made
because the underlying mechanism had to be built from the one-paragraph
abstract alone:

- **BEV grid**: 32×32 cells, 1.0 m/cell, ego near the bottom row (`src/geometry.py::BEVGrid`).
- **Camera rig**: 3 synthetic pinhole cameras (front/left/right, 100° FOV,
  60 px-equivalent focal length, 1.4 m mount height) — the paper does not
  specify a camera count or intrinsics.
- **Feature dimensionality**: both the prior-point descriptor and the live
  per-pixel camera feature are 16-d stand-ins for real DINOv3 (768-d+) and
  a real image-backbone feature map, respectively — chosen purely to keep
  this reconstruction small enough to train on CPU in minutes.
- **Sparse Voxel Prior Encoder**: implemented as scatter-mean point pooling
  into occupied BEV voxels + a per-voxel MLP + one dense smoothing conv to
  fill sparse gaps. A real sparse-conv backbone (spconv / MinkowskiEngine)
  would keep the representation sparse through several conv layers; this
  reconstruction only keeps the *spirit* (only occupied cells do the
  pooling work) at far smaller scale.
- **Camera BEV Lifter**: geometry-guided `grid_sample` lifting via the
  known pinhole rig (echoing this series' Day 22 S2Planner reconstruction's
  camera-projection approach) — the paper does not specify how its live
  camera branch lifts features to BEV.
- **Prior-Map BEV Fusion**: a learned per-cell sigmoid gate deciding how
  much to trust the prior vs. the live branch, followed by a residual conv
  block. The paper says only "fuse ... in bird's-eye view" — the gate and
  residual design are this repo's own choice.
- **Detection Head / Map Head**: a CenterNet-style heatmap+box head and a
  per-cell binary lane/curb classification head, standing in for the
  paper's "task-specific **transformer** heads" and its **vectorized**
  polyline output. This is a real, disclosed simplification — this repo
  does not implement a vectorized-polyline transformer decoder (e.g.
  MapTR-style query-based prediction); `simulate.py` extracts approximate
  polyline visualizations from the per-cell heatmap purely for the GIF.
- **Synthetic dataset**: a hand-built scene generator (`src/dataset.py`)
  with curved lane/curb polylines and 2–4 dynamic vehicles that are
  present in the live camera views but *never* in the static prior map
  (a static map cannot contain moving objects) — designed specifically to
  let this repo test the paper's own qualitative claim (prior helps
  mapping much more than detection) on data where the ground truth for
  both effects is known exactly. Live-camera visibility of static lane/curb
  points is deliberately occluded (65% dropped per scene) to simulate the
  camera-only depth/parallax ambiguity the paper's abstract names as the
  problem the prior map solves.

## Result: this repo's honest finding vs. the paper's

This repo's own measurements (synthetic data, `metrics.json`, not the
paper's Argoverse 2 numbers) show a **much larger effect on mapping than
on detection** — Map IoU 0.348 → 0.763 (prior on vs. off) vs. Detection F1
0.635 → 0.552 — which **matches the paper's own qualitative pattern**
(a small CDS delta, a much larger mAP delta) in direction of the mapping
effect, but **not in the direction of the detection effect**: the paper
reports a small *positive* detection change with the prior enabled
(0.287 → 0.299 CDS), while this reconstruction's detection F1 actually
*drops* slightly with the prior enabled (higher precision, lower recall).
This is disclosed honestly rather than tuned away — see the Day 23
daily-review doc's Implementation Notes for the likely cause (both
detection and mapping heads read the *same* shared fused-BEV features in
this reconstruction, so a change that helps the map head's gate can shift
the shared feature distribution in a way that makes the detector more
conservative; the real paper's task-specific transformer heads may
decouple this more effectively than this repo's shared-trunk simplification).
