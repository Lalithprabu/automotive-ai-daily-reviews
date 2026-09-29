# Sourcing & honesty disclosure (Day 25 — ECO)

**Paper:** "Guiding End-to-End Driving Models with Endpoint-Constrained Trajectory Optimization" — arXiv:2609.31383, submitted 2026-09-25.
Authors: Brayden Zhang, Mahsa Golchoubian, Igor Gilitschenski, Boris Ivanovic, Kashyap Chitta (+2). Univ. of Toronto / Vector Institute / NVIDIA Research / ELLIS Institute Tübingen / KE:SAI.

## Paper-sourced (abstract-level only)
- ECO is a training-free post-processing layer: anchors the trajectory to executed history, preserves the policy's predicted endpoint, reshapes intermediate waypoints; needs no map, no privileged simulator state.
- Reported: VaVAM 18.1 → 31.0 HD-Score on HUGSIM (+71%, 1st place); VaVAM +123% and DiffusionDrive +22% on AlpaSim; tested on six generative and regression policies.

## NOT recoverable (arxiv.org/abs/2609.30818 and full text were unreachable; alphaxiv gave abstract only)
The objective terms, constraint formulation, solver, and hyperparameters are **not known to this repo**.

## This repo's own reconstruction (not paper-sourced)
The quadratic objective (acceleration + jerk + fidelity), the closed-form solver, the toy waypoint policy, the synthetic route/bicycle-model world, the pure-pursuit tracker, and every number in `results.json`.
The synthetic numbers are NOT comparable to the paper's HUGSIM/AlpaSim scores.

## Findings worth knowing
- ECO cut RMS jerk 68% and steer-rate 39% in the toy world, but RMS lateral error got slightly WORSE (0.35 → 0.45 m): smoothing trades some path fidelity for comfort. Reported, not tuned away.
- The off-road / progress deltas (2.5% → 0%, +0.9 pts) come from 40 episodes and are within noise.
