# Sourcing notes (Day 68 - LOWP)

Paper: *Look Only Where You Plan for Real-Time End-to-End Urban Driving* - Haohua Que, Tianle Zhu, Qianyi Wu,
Zhipeng Bao, Handong Yao. arXiv:2611.10264 (cs.RO), v1 submitted 2026-11-06. https://arxiv.org/abs/2611.10264

What was read: the arXiv abstract page and the arXiv HTML full text (method, tables 1-8, limitations), via a
summarising web fetch (the last ~5% of the HTML was not read). No official code was linked.

Taken from the paper: two readers x 16 learnable queries, 3 layers each; metric 3D anchors on a grid ahead,
per-scene shift, bounded tanh refinement; 4 heights over 3 m x 8 ring points = 32 candidates; projection
K R^T (p - t); bilinear sampling + multi-head attention with a learned null candidate; anchor position embedding;
re-seeding of invisible anchors; ray reader labelling every feature cell with 3D points at 8 depths (1-60 m);
per-expert 2-layer transformer planning head with an ego-status token; L1 per expert; inference mix
alpha = 0.5 straight / 0.8 turns.

My choices (not in the paper): hidden size 64, tiny stride-8 CNN trunk, log-spaced ray depths, self-attention
between anchor queries, command one-hot inside the ego token, displacement-integration output, IPM dense
baseline (a stand-in, not ResWorld), synthetic ray-cast world.

Not implemented: LiDAR depth adapter (nuScenes), GTRS vocabulary scorers (NAVSIM/Bench2Drive), control
branch, reverse / brake-release rules, history frame, host-sync latency engineering.

Paper numbers quoted in the post (all from the paper, not reproduced here):
- nuScenes open-loop avg L2: 0.296 m (seed 1) / 0.298 (seed 3) vs ResWorld 0.302 (text) / 0.300 (table);
  LOWP-S 0.309. Collision 0.056% / 0.048%; paired intervals include zero.
- 42.9M params vs ResWorld 80.8M (~53%); 17.55 vs 31.29 ms min latency (RTX PRO 6000, FP32, b1) = ~1.7-1.8x.
- NAVSIM navtest PDMS 85.89 (scorer + own candidates; +0.54) vs TransFuser 84.0, DiffusionDrive 88.1, iPad 91.7.
- Bench2Drive DS 65.51 (218 routes, control config) vs iPad 65.02; Hydra-NeXt 73.86; HiP-AD 86.77
  (non-matched wrappers / route sets).
- Failure modes: 26 of 42 blocked Bench2Drive routes are brake-head standstill deadlocks; brake release
  +1.59 +/- 2.06 DS (inconclusive) with vehicle collisions 0.44 -> 0.62 per route; turns: 7 collisions after mixing,
  global reader alone 24; ResWorld keeps a small turn-L2 edge (0.760 vs 0.784 MAX L2).
