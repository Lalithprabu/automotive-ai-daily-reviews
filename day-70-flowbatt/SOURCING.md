# Sourcing - what comes from the paper vs what is mine

**Paper:** *FlowBatt: Flow Matching for Probabilistic Battery Degradation Prediction*. Hamidreza Eivazi, Eslam Sharaawy,
Stefan Wittek, André Hebenbrock, Raphael Ginster, Steffen Blömeke (TU Braunschweig, TU Clausthal).
arXiv [2611.03150](https://arxiv.org/abs/2611.03150), v1 submitted 2026-11-02, listed as a NeurIPS 2026 poster.
Official code: https://github.com/HamidrezaEiv/FlowBatt

## Where I read things
* arXiv abstract page + arXiv HTML: **rate-limited (HTTP 429) every time I tried this session**, so the full text is unread.
* alphaXiv abstract + overview pages: abstract, method summary, datasets, baselines, MIX-20 headline number, limitations.
* Official repo README (raw.githubusercontent.com): 256-point trajectories, 100x100 capacity matrices (20x100 MIX-20),
  10-point SOH prefix (2 for MIX-20), train/test cell counts, AdamW lr 1e-3, 100 warm-up steps, 10,000 epochs,
  final-epoch checkpoint (no test selection), LeakyReLU slope 0.3 in the xai-cnn encoder, evaluation definitions.
* NOT read: Table 1 per-dataset RMSEs (image), RUL numbers, coverage numbers, DiT depth/width, sampler step count.

## Paper facts used in the post
| Claim | Source |
|---|---|
| Conditional flow matching + DiT backbone, conditioned on early-cycle capacity data | abstract |
| CNN encoder -> embedding + generative-time embedding -> adaptive normalisation | alphaXiv overview |
| Grad-CAM found initialisation-sensitive dead-ReLU encoder failures -> LeakyReLU | abstract + overview |
| Best trajectory SOH RMSE on all 5 benchmarks, errors < 2.5 % | abstract |
| MIX-20: 0.88 ± 0.03 % vs DiffBatt 1.15 ± 0.08 % (generative ± = 10 draws from ONE checkpoint) | overview |
| Competitive but not uniformly best on RUL (vs BatLiNet etc.) | abstract |
| 90 % intervals from 100 samples; coverage BELOW nominal, esp. MATR-2 / HUST | overview (limitations) |
| No held-out chemistries / protocols; capacity-only conditioning (no temperature, current, cooling) | overview (limitations) |
| DiffBatt shares backbone + encoder; flow objective's share of the gain not isolated; LeakyReLU not isolated | overview |

## My choices (toy)
Synthetic fleet (3 chemistries, hidden late defect, random knee) - `flowbatt/world.py`; d_model 128, depth 4, 4 heads,
patch 16; linear path, t ~ U(0,1); Euler 32 steps; DiffBatt-lite = cosine-schedule DDPM, 100 steps, same DiT + encoder;
deterministic Transformer = same trunk + MSE; MLP head; random forest on Severson-style features; 2,500 steps, batch 64.
