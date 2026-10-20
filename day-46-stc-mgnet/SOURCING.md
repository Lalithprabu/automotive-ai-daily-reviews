# Sourcing & disclosure — Day 46 (STC-MGNet)

**Paper:** STC-MGNet: Spatial-Temporal-Channel Attention-Enhanced Multimodal Graph Network for Pedestrian Intention Prediction — Yuanzhe Li, Meng Wang (TU Berlin listed). arXiv:2610.23356, cs.CV, submitted 2026-10-16 (4 days old at run time).

**Access:** arxiv.org/abs fetch returned HTTP 429 with a do-not-retry instruction; the alphaxiv.org abstract page worked (abstract-level only: no layer sizes, losses, baselines, ablation tables). No code release found.

## Paper-sourced (abstract level only)
- Pose-sequence graph encoder with STC attention: three parallel branches emphasising joints, frames, feature channels, combined with learnable fusion weights.
- Multi-stage cross-modal channel modulation: modality-specific channel gating at successive graph-encoding stages, using visual and ego-vehicle motion information.
- Reported accuracy: 70% (JAADbeh), 87% (JAADall), 88% (PIE); ablations say both visual and ego motion help.

## This repo's own reconstruction (NOT paper-sourced)
Everything else: attention formulas (conv1d/MLP + sigmoid), fusion = 3*softmax(theta), gate form 2*sigmoid(Linear(embedding)), 14-joint skeleton, 3 stages of 32/64/64 channels, GRU ego encoder, MLP visual encoder, the synthetic generator, and every number in `results.json`. No JAAD/PIE data or images are used; "visual" is a 6-d synthetic scene vector.

## Findings, reported as they came out (synthetic, single seed)
- Test accuracy: pose-only ST-GCN 65.9% -> +STC attention 68.3% -> full STC-MGNet 89.2%.
- **Confound:** the old-way baseline never sees visual/ego data, so the +23-pt jump mostly measures "having context at all", not the gate design. A late-fusion baseline with the same context was NOT run.
- STC attention alone: +2.4 pts, within plausible seed noise (1 seed, 1200 test windows, ~±1.4 pt binomial s.e.).
- Learned STC fusion weights barely moved from (1,1,1) (~0.92/1.01/1.07); learned frame attention is nearly saturated. The attention panels in the GIF are illustrative, not evidence of interpretability.
- `simulate.py` picks, out of 60 random scenes per class, the one the final model classifies most confidently — representative of "works", not of average behaviour.
- Dataset construction determines the result: cues are built to be independently informative.
