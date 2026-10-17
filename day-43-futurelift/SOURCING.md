# Sourcing & disclosure — Day 43, FutureLift (arXiv:2610.22215)

**Paper:** *FutureLift: Lifting Trajectory Proposals into Bidirectional Future Graphs for Motion Forecasting* — Daniel Grimm, J. Marius Zöllner (FZI Research Center for Information Technology), cs.CV, arXiv 2610.22215v1 (Oct 2026).
Retrieved via `arxiv.org/html/2610.22215` (abs pages for this and another candidate returned HTTP 429). The appendix (hidden width, depth, optimizer settings) and Sec. 4.6 were not available.

## Paper-sourced
- Backbone-agnostic refiner; inputs are only proposal coordinates, scores and scene context (no backbone queries/features).
- Each proposal lifted to a heterogeneous future graph: R contiguous intervals per agent-mode (R=3 on AV2/6 s, R=4 on WOMD/8 s), node anchored at the interval endpoint, features from a GRU over the segment in a local frame.
- Typed edges: historical, map, **bidirectional temporal**, modal (+ social for joint prediction only). Relative-geometry edge features.
- L layers of relation-specific MHA with edge features, residual updates, pre-LN FFN; graph built once, all segments updated simultaneously.
- Heads: per-interval coordinate residuals, Laplace scale (ELU+1+0.1), confidence residual. Loss: Laplace NLL on winner-takes-all mode + endpoint Laplace-mixture NLL for scores, unit weights, gradients stop at proposals/winner selection.
- Paper results (NOT reproduced here): AV2 marginal QCNet minADE6 0.729→0.593, minFDE6 1.304→1.124; WOMD test minADE 0.5049 / minFDE 0.9942 (first among compared standard-input methods).
- Paper-stated limitations: refines only supplied hypotheses; per-backbone training; graph size ∝ A·K·R.

## This repo's own defaults (not paper-sourced)
d=64, 4 heads, 2 layers, edge-MLP form, layer ordering (history→map→temporal→modal→FFN), residual scale ×5 m, zero-initialised head, 1-layer Transformer scene encoder over (4 histories + 46 map points), winner picked on detached *refined* ADE, per-step (not per-interval-cumulative) residuals, the **synthetic junction world** (`futurelift/data.py`), and the **map-blind MLP backbone** (`futurelift/backbone.py`). A=1 (marginal) only; social edges are not implemented.

## What the numbers mean
All metrics are on a held-out synthetic set (800 scenes). The stand-in backbone is deliberately weak and map-blind, so the large gain over it (minADE −58.7 %) is **not comparable** to the paper's ~19 % gain over QCNet. The *temporal-edge* ablation is the paper's actual novelty and here it is within noise (see README). Single seed.
