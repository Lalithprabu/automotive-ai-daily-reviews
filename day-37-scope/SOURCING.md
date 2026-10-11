# Sourcing & disclosure — Day 37 (SCOPE)

**Paper:** *Control-Ready Uncertainty for Trajectory Diffusion* — Zhiwei Xue, Jia Yue Kam, Jinhang Qiu, Yifeng Cheng, Ege Gursoy, Jiaming Wang, Vincent Bonnet, Harold Soh. arXiv:2610.12431 (cs.RO), submitted 2026-10-08, accepted CoRL 2026 (Spotlight).

**How it was read:** the arXiv abstract page was fetched directly. A third-party summary (alphaxiv overview) supplied mechanism detail and numbers. The full text (`arxiv.org/html/2610.12431`) was rate-limited (HTTP 429, "do not retry") and **was never read** — so the alphaxiv-sourced items below are second-hand and unverified against the paper itself. No public code release was found.

## Paper-sourced (abstract and/or alphaxiv summary)
- SCOPE = Score-Curvature for Online Precision Estimation; lightweight module on a frozen, mode-conditioned diffusion trajectory model.
- Learns a structured precision matrix around each nominal trajectory by distilling score-curvature (Hessian-vector products on random Hutchinson-style probes) → calibrated Gaussian "tubes", no repeated sampling.
- Precision = block-banded + low-rank; per-scale calibration factors fit on held-out data to minimise trajectory-level NLL.
- Tubes serve as occupancy predictions for moving agents and as time-varying MPPI exploration covariance.
- Evaluated on pedestrian forecasting, crowd navigation, Maze2D, real Franka Panda. Maze2D @1,024 rollouts: success 99.25% vs 96.35%, collisions 0.65% vs 2.60% (vs Diff-MPPI); Franka: 70% vs 53% over 30 trials.
- Stated limitations: no formal coverage guarantee; depends on a reliable backbone; Gaussian tube is local, not multimodal.

## This repo's own reconstruction (NOT paper-sourced)
Everything else: the EDM-style MLP backbone (217k params), the synthetic pedestrian generator (known heteroscedastic noise), the exact parameterisation `P = s0(LLᵀ+UUᵀ+εI)` with element-banded L, the loss (Hutchinson probe matching **plus** a Gaussian-KL term that needs the dense 24×24 target — only feasible because D=24), the dense Cholesky solve (a structured O(T) solver is not implemented), the single-scalar calibration, the Monte-Carlo and fixed baselines, the lane-encroachment "brake" demo, and every number in `results*.json`. Diffusion runs on the residual from a constant-velocity extrapolation. σ0=0.05 and all dims are our defaults.

**All reported numbers are on synthetic data and must not be compared with the paper's.**

## Findings / caveats worth knowing
- Joint 24-D NLL was abandoned: smooth temporal kernels give near-singular covariances, so even the *oracle* looked terrible (cov90 0.17). Metrics are per-timestep 2-D.
- The "oracle" (true noise cov) is calibrated with a scalar α=1.34 because it ignores the nominal's own bias.
- SCOPE's α came out ≈1.0 — the distilled curvature was already near-calibrated here.
- Marginal 95% coverage over all scenes is similar for SCOPE and a fixed covariance (95.3% vs 94.4%); the difference is *where*: fixed over-covers low-density (97.5%) and under-covers high-density (89.2%) scenes.
- Brake-decision demo (754 mode-0 test scenes, threshold 0.05): SCOPE agrees with the oracle 64.1% vs fixed 28.8%; SCOPE 0 false alarms / 271 misses; fixed 214 false alarms / 323 misses. The oracle uses the true mean, so agreement is capped by the backbone's nominal bias.
- Monte-Carlo (K=32) is close to SCOPE in NLL (−1.339 vs −1.379) but ~45× slower on CPU at batch 1.
- The GIF scene is selected from the held-out set to show the high-density regime (not a random draw).
