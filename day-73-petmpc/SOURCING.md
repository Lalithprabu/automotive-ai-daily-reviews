# Sourcing - what comes from the paper and what is mine

**Paper:** Alessandro Febbraro, Federico Rollo, Marcello Farina, Luca Bascetta, Navvab Kashiri.
*Prediction-Error-Tightened MPC for Socially Compliant Crowd Navigation.* arXiv:2611.15781 (cs.RO),
submitted 11 Nov 2026. 9 pages, 3 figures, 4 tables; submitted to ECC 2027.
Politecnico di Milano (DEIB) and Leonardo S.p.A. Robotics Lab. No public code found.

## What I could read
* The arXiv abstract page (all numbers quoted below come from the abstract).
* The arXiv HTML **up to Section II-D only** (the fetch was truncated, and later attempts were rate-limited).
  So the Koopman predictor, the MPC on a look-ahead point and the collision-cone screening follow the paper;
  the residual-moment estimator, the exact Markov margin equation, the grouping criterion, the
  personal-space model and the simulation setup are **my reconstruction**.

## Taken from the paper
| Element | Paper (sections I-II) | This repo |
|---|---|---|
| Observable | 7-D: planar position, planar velocity, heading, angular rate, speed | same, `predictor.py` |
| Operator | structured (sparse) 7x7, identified online by regularised LS over a sliding window of one-step transitions, Frobenius pull to the previous operator, held fixed over the horizon | same |
| New tracks | constant-velocity / constant-turn-rate prior | same (`cvtr_prior`) |
| Robot model | outer linear MPC on a look-ahead point, single integrator ξ+ = ξ + T_s u, I/O linearisation to (v, ω), heading from the previous iteration | same, `planner.py` |
| Cost | quadratic tracking + terminal + control weights | same |
| Constraints | v / ω bounds, velocity-variation constraints, linear half-plane collision constraints | same |
| Aggregation | groups of obstacles -> group collision cone from tangent rays -> threat if relative velocity is a non-negative combination of cone edges or robot inside -> ONE separating line per retained group | same idea (cone test per member disc) |
| Uncertainty | prediction errors characterised online through **second moments**, entering both screening and separating constraints via **Markov-based margins** | `ResidualMoments` + `markov_margin` |
| Semantics | pedestrians vs generic obstacles; personal space preserved for pedestrians | `aware` flag, soft personal-space slack |

## My choices (not from the paper)
* Sparsity pattern: position rows FIXED to kinematics, identified (vx, vy, s) block + θ/ω rows (14 free entries).
* **Spectral projection** of the (vx, vy, s) block to ρ ≤ 1. Without it the online operator diverged
  (4.8 km FDE at 2 s in the crowd study). The paper may handle this differently.
* Window 12, λ = 2.0, horizon N = 20 (2 s), T_s = 0.1 s, v_max 1.2 m/s, ω_max 1.5 rad/s, a_max 1.5 m/s².
* Residual moments: EMA (β = 0.08) of the squared i-step-ahead position error per track and step; young
  tracks are blended with a population prior. Margin m_i = sqrt(Σ_i / δ), δ = 0.25 (Markov's inequality on
  ||e||²), capped at 1.0 m.
* Slack prices that decay along the horizon (0.85^i). Without this, far-horizon rows with large margins were
  infeasible and PET-MPC was *worse* than nominal (6.7% vs 3.3% contact on development seeds 0-59).
* Grouping: single linkage when the gap between discs is smaller than the robot's width.
* Personal space: 0.9 m radius disc around each pedestrian, enforced as a soft slack band (w = 40) on top
  of the hard physical constraint (w = 4000).
* World: a straight 6 m corridor, 20 m run, non-reactive agents with Ornstein-Uhlenbeck turn rates,
  abrupt turns and stop-and-go. Development seeds 0-119; all reported numbers are on seeds >= 1000.

## Paper numbers (abstract) vs mine
| | Paper | Mine (toy) |
|---|---|---|
| Moving generic obstacle, runs with contact | 10.0% -> 1.33% (300 paired seeds) | 6.0% -> 0.0% (300 held-out paired seeds) |
| Travel-time cost | +0.68 s | +0.31 ± 0.10 s |
| Personal-space overlap, pedestrian-aware vs generic | 2.11% vs 22.84% of time (offline ablation) | 2.4% vs 11.8% (closed loop, 40 seeds), but contacts 15% vs 7.5% |
