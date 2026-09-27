# Sourcing disclosure

This repository is a **from-scratch reconstruction** of the core ideas in a
very recent paper, built from its abstract only. We could not retrieve the
paper's full text (arXiv full-text fetch returned repeated HTTP 429 rate
limits during this project's research step — a recurring issue for this
series). **Everything beyond the paper's title, authors, and verbatim
abstract below is this project's own design**, chosen to be architecturally
consistent with what the abstract describes, and documented as such
throughout the code.

## The paper (everything we actually know)

**ForeDrive: Foresight-Guided End-to-End Autonomous Driving with a
Planning-Relevant Latent World Model**
arXiv:2609.26299 — submitted 2026-09-22 (v1) / 2026-09-23 (v2), cs.CV
Authors: Sinuo Wang, Zichong Gu, Yuhan Huang, Wenxin Wen, Xun Yang, Yiqing
Zhang, Xingyu Zhang, Ningyu Che, Jie Ling, Qiankun Yu, Wei Liu, Jing Xu,
Xinggang Wang. 9 pages + 4 figures, 8-page supplementary.

> Existing latent world models are typically optimized for future
> predictability, yet the resulting representations are not necessarily
> useful for planning in autonomous driving. Predictions are commonly used
> for pretraining or auxiliary supervision rather than as direct
> conditioning signals for trajectory generation. We propose ForeDrive,
> which learns a planning-relevant latent representation and couples it
> asymmetrically to a Diffusion Transformer (DiT) planner. The planner
> consumes multi-horizon latent future representations learned with a
> JEPA-style world model; planning gradients update the shared online
> encoder, while stop-gradient routing trains the latent predictor with
> forecasting losses only. Because predicted futures have varying
> reliability across horizons and BEV trajectories are misaligned with
> image tokens, we use gated visual fusion, future-status injection, and
> Trajectory-Adaptive Bias (TAB) to inject future latents as guidance
> without overriding the current observation. Trained with pure imitation
> learning and using only the current front-view image as visual input at
> inference, ForeDrive attains 89.9 PDMS on NAVSIM v1 and 90.0 one-stage
> EPDMS on NAVSIM v2, without reinforcement learning or an external
> trajectory scorer.

That's it. No architectural diagrams, dimensions, loss formulas, TAB math,
dataset splits, ablations, or baseline numbers beyond the two headline
PDMS/EPDMS figures are known to us.

## What's real vs. reconstructed

| Item | Status | Notes |
|---|---|---|
| Paper title, arXiv ID, authors, dates | **Real** (from the abstract page) | |
| Verbatim abstract | **Real** | Quoted above and in README.md |
| 89.9 PDMS (NAVSIM v1), 90.0 EPDMS (NAVSIM v2) | **Real** (paper's own claimed numbers) | Not reproduced here — no real NAVSIM data or training was used |
| "JEPA-style world model", "asymmetric coupling to a DiT planner", "stop-gradient routing", "gated visual fusion", "future-status injection", "Trajectory-Adaptive Bias (TAB)" as named mechanisms | **Real** (named in the abstract) | The abstract confirms these mechanisms EXIST; it does not specify HOW they are implemented |
| Encoder architecture (CNN depth/widths, token grid size) | **Reconstructed** | Our own small strided-CNN, sized for a 64×64 synthetic raster and CPU training |
| World-model / predictor architecture (shared Transformer vs. per-horizon heads, hidden sizes) | **Reconstructed** | We chose a shared Transformer with per-horizon embeddings; documented in `models/world_model.py` |
| Exact stop-gradient wiring (where `.detach()` is called) | **Reconstructed, but principled** | Derived directly from the abstract's description of what must and must not receive gradient from which loss; see `models/foredrive.py` |
| Gating / confidence mechanism | **Reconstructed** | A learned per-token, per-horizon sigmoid confidence head; the abstract only says reliability "varies across horizons" |
| Future-status injection format | **Reconstructed** | Horizon-index embedding + confidence, concatenated and projected |
| TAB formula | **Reconstructed** | Our own trajectory→attention-bias projection; the abstract names the mechanism and its motivation (BEV/image-token misalignment) but not its formula |
| DiT planner architecture, diffusion objective (epsilon-prediction), noise schedule, DDIM sampler | **Reconstructed** | Standard, well-known diffusion-model choices, sized down for this demo |
| NAVSIM / nuScenes training data | **Not used** | We have no access to it; see synthetic dataset below |
| Synthetic driving-scene dataset (`src/data/synthetic_dataset.py`) | **Entirely ours** | A small kinematic simulator + ego-centric BEV occupancy rasterizer, standing in for real camera + trajectory data, built to give both the forecasting and imitation tasks genuine (non-degenerate) learning signal — see that file's docstring and `tests/test_dataset.py` / `tests/test_sanity_signal.py` for how we checked this |
| All training curves, loss values, parameter counts, GIF outputs reported in this repo / the accompanying LinkedIn post | **Real, measured** on our synthetic data with our reconstructed model | Not comparable to the paper's real NAVSIM PDMS/EPDMS numbers |

## Known limitations of this reconstruction (be candid — this is content, not just a caveat)

- **Same-encoder JEPA target, not a momentum/EMA target encoder.** Real
  JEPA/BYOL-style methods typically use a slowly-updated (EMA) target
  encoder to keep the forecasting target stable while the online encoder
  changes. We use the *same* online encoder for both, for simplicity. This
  has a real, observable consequence: because the online encoder is also
  being updated by the planning loss, the forecasting target keeps moving
  under the predictor during training. In our `train.py` runs, the average
  forecast loss did not monotonically decrease across epochs for this
  reason (see the training log in the project's daily-series notes). It
  does not affect the stop-gradient correctness property itself (verified
  in `tests/test_stop_gradient.py`), only training dynamics/stability.
- **Full generative DDIM sampling does not reliably beat a strong
  constant-velocity baseline** in our compute-limited (CPU, tiny model, a
  few dozen epochs) setup, even though it clearly learns real, scene-
  conditioned structure (see `tests/test_sanity_signal.py`'s docstring for
  the exact numbers and reasoning). A production system with real data and
  real training budget would be expected to do much better; this is a
  demo-scale limitation, not a claim about the real ForeDrive paper.
- **The "front-view" raster is an ego-centric BEV occupancy proxy**, not a
  perspective camera image. See `src/data/synthetic_dataset.py`'s module
  docstring.
- **TAB's attention-head handling is simplified**: our TAB bias is shared
  across attention heads rather than being per-head.
