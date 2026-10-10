# Sourcing & disclosure — Day 31

**Paper:** *Localized Conformal Safety Monitoring with Vision-Language Models for Autonomous Driving*
Luís Marques, Rong Fang, Disha Kamale, Dmitry Berenson (University of Michigan) — arXiv:2610.02765, submitted 2026-10-02.

## Access
`arxiv.org/abs/2610.02765` and `arxiv.org/html/2610.02765` returned HTTP 429 ("do not retry") from this session's fetch proxy and were not read. Everything below marked *paper-sourced* came from the alphaxiv.org mirror (abstract-level summary only). No full text, no code release, no results tables were available.

## Paper-sourced (abstract level only)
- Problem: judging whether a planned ego trajectory will collide with actors whose behavior reacts to the ego; classical approaches limited by forecasting-model quality.
- Method name & idea: Split Label-Localized Conformal Prediction (SLLCP) — turns unreliable VLM predictions into probabilistically calibrated safety prediction sets, with scene-dependent localization that re-weights relevant calibration experience; label-conditional finite-sample distribution-free coverage under exchangeability; backbone-agnostic.
- Results: 15,000 novel CARLA trajectories; unsafe trajectories detected 89.6% (Qwen backbone) and 88.4% (Cosmos) vs. 4.6% and 39.1% for the uncalibrated baselines.

## This repo's own reconstruction (NOT paper-sourced)
Nonconformity score (1 − p_y), Gaussian kernel and bandwidth, the scene embedding, weighted-quantile construction, the stand-in MLP predictor (the paper uses frozen Qwen/Cosmos VLMs), the entire synthetic ego-reactive scenario simulator, the covariate-shift protocol (calibrate mostly-clear / test mostly-occluded), α = 0.1, and every number printed by `train.py`.

## Honest findings from this reconstruction
- Calibration layer vs. uncalibrated argmax: unsafe recall 0.1% → 89.5% (qualitatively the paper's pattern; absolute numbers are not comparable — different data and predictor).
- Localization vs. plain global label-conditional CP: only a small gain here (occluded-regime recall 0.885 → 0.893) at slightly higher false alarms. The big win is the conformal calibration itself; localization's measurable benefit in this synthetic setup is modest.
- Cost: the weak stand-in predictor forces ~65% false alarms on safe trajectories to hit 90% recall. A better predictor shrinks this; the guarantee holds regardless.
- `simulate.py`'s stream is curated (every 4th scene unsafe, ~27%) vs. the 12.5% test base rate, for visibility; headline metrics come from `train.py`, not the GIF.
