# Sourcing & disclosure (Day 41)
Paper: *Uncertainty-Aware Optimization for Physics-Aware Highway Trajectory Prediction* — A. R. Chugh, S. Dorn — arXiv:2610.11580 (submitted 2026-10-08). No code release found ("upon acceptance").

## Paper-sourced (via arxiv.org/html full text, paraphrased)
X-TRACK structure (xLSTM encoder, GAT, LSTM decoder, kinematic bicycle layer, inputs a_x & yaw rate); Gaussian-NLL loss L_traj + 0.1 L_dyn; deep ensemble K=5 heterogeneous; MC dropout M=20 with locked dropout (0.15/0.2/0.2/0.2); sampling-based propagation N=16; law of total covariance; split conformal with Mahalanobis score, per-horizon quantile, 95% target; highD 3 s history / 5 s horizon; headline results (ADE 0.56→0.44, FDE 1.76→1.49, coverage 62.01→95.39 DE / 67.47→95.72 MCD, MPIW 1.57→2.11 DE); stated limitations.

## This project's reconstruction (NOT the paper)
Everything else: synthetic data generator (5 Hz, not highD 25 Hz), LSTM instead of xLSTM, hidden sizes, GAT details, lead-vehicle interaction, adjacency, ensemble variant list, learning-rate schedule, epochs (22), split sizes, the 3x-noise stress test. All numbers in README / post tagged "synthetic" are from this repo only and are not comparable to the paper.

## Honest findings from the replica
* Heads trained with NLL on this (near-Gaussian) synthetic noise are already fairly calibrated: raw 95% coverage 89–94%, far less overconfident than the paper's 62–67%. Conformal still lifts all three to ~94.3–94.4% (target 95%, ±~1% sampling error).
* Ensemble ADE gain 11.6% vs the paper's 21%; MC dropout 7%.
* Stress test with 3x future noise did not break coverage (ADE barely moved), so the exchangeability limitation was not demonstrated here.
