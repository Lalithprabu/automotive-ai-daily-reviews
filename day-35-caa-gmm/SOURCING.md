# Sourcing & disclosure — Day 35 (CAA-GMM)

**Paper:** *Context-aware Attention-based Gaussian Mixture Models for Vehicular Trajectory Prediction* — arXiv:2610.09174 (cs.LG), submitted 2026-10-06 (v1). Authors: Arash Raftari, Babak Ebrahimi Soorchaei, Yaser P. Fallah — University of Central Florida / AMD.

## Access
Only the alphaxiv.org mirror's abstract-level summary was retrieved. `arxiv.org` abs/html/list pages were not usable this run (listing pages truncated at 50 of 615 entries). **No numeric results, layer-level architecture, loss, or training details were recoverable. No code release found.**

## Paper-sourced (abstract level only)
- Multimodal, uncertainty-aware trajectory forecasts as a Gaussian mixture; one interpretable Gaussian per mode.
- Conditioned on scene context + agent dynamics; rasterized map/environment cues + motion history.
- A lightweight attention mechanism weighs agent interactions and the relative importance of contextual cues.
- Qualitative: competitive/better than raster-based SOTA on nuScenes and Argoverse 2 at low compute; ablations support the attention module; resilient under imperfect communication and perception.

## This repo's own reconstruction (NOT paper-sourced)
Everything else: the synthetic junction-approach simulator (IDM lead vehicle, route-hint channel, 75% hint reliability), the 4-channel raster, d=64, 4 heads, the two cross-attentions, the null token, the 3-way softmax gate, the shared per-mode head with mode embeddings, relaxed-WTA loss, K=6, the raster+MLP baseline, the outage-dropout augmentation, and every number below. 148,964 params (baseline 160,945) — toys.

## Honest findings (synthetic, 1 seed, 1,500 test scenes)
| Setting | CAA-GMM minADE@6 | Baseline minADE@6 | CAA top-1 ADE | Base top-1 ADE |
|---|---|---|---|---|
| Trained clean, clean test | 0.055 | 0.136 | 1.42 | 2.24 |
| Trained clean, 60% msg drop | 0.390 | 0.179 | 1.78 | 2.29 |
| Trained clean, 100% drop | 0.578 (mixNLL +102) | 0.213 | 1.97 | 2.32 |
| Outage-aug trained, clean test | 0.093 | 0.170 | 1.12 | 1.17 |
| Outage-aug trained, 60% drop | 0.142 | 0.171 | 1.25 | 1.17 |
| Outage-aug trained, 100% drop | 0.175 | 0.171 | 1.33 | 1.17 |

- minADE@6 is easy on this task (3 branches, K=6); top-1 ADE and mixture NLL are the more informative columns.
- Outage-aug baseline became insensitive to dropout (it learned to ignore neighbours) at a cost in clean accuracy (0.136→0.170).
- Attention mass on the true lead: 0.357 (clean-trained) / 0.446 (aug-trained) vs 0.319 uniform reference — only partly interpretable.
- Not tested: real data, multiple seeds, the paper's metrics, position-noise robustness beyond a 1 m bias (clean-trained CAA 0.085 vs 0.170), crowded scenes (>4 neighbours), compute/latency claims.
- Demo episode in `simulate.py` is chosen by a non-outcome rule (lead vehicle, ≥1 distractor, turning branch).
