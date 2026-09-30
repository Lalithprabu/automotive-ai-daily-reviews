# Sourcing & honesty disclosure — Day 26 (WALT)
**Paper:** WALT: Learning World-Model-Aligned Latent Trajectories for Autonomous Driving — arXiv:2609.30436, submitted 2026-09-24 (HKUST, Horizon Robotics, CUHK, others; Ping Tan, Wei Yin among authors).
## Paper-sourced (alphaxiv overview only; arxiv.org abs/html returned HTTP 429)
- Learns a compact generative trajectory latent space by transferring information from a frozen pretrained driving world model; dual-branch trajectory autoencoder; investigates JEPA and REPA for trajectory learning.
- NAVSIM v1 PDMS 89.4→89.8; NAVSIM v2 EPDMS 87.3→87.9; 30.5% lower trajectory-planner compute.
## This repo's own reconstruction (NOT paper-sourced)
Every dimension, the geometry/semantic branch design, cross-attention "transfer", the cosine REPA loss, the planner, the synthetic road/lead-vehicle world and every number in results.json. Synthetic numbers are not comparable to NAVSIM.
## Findings (reported, not tuned away)
- In this toy, WALT-style aligned latent planner (ADE 3.48 m) is WORSE than a raw-waypoint planner (3.25 m); a geometry-only latent AE (no world-model transfer) is best (2.52 m). The paper's claim is NOT reproduced here.
- Bug found: world model pretrained with plain MSE ignored the sparse lead-vehicle channel; its tokens carried no vehicle info (all planners stuck at ~4.3 m vs 5.05 m mean baseline). Fixed with occupancy-weighted loss.
- All planners are undertrained (a plain CNN gets 1.9 m ADE); single seed, no error bars; differences of ~0.2 m may be noise.
