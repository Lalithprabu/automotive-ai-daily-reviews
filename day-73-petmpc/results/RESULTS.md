# Results (synthetic toy world, held-out seeds >= 1000)


## obstacle study (300 paired seeds)

| Controller | Runs with contact | Contact time | Personal-space overlap time | Mean time to goal (s) | Success | 5th pct min clearance (m) |
|---|---|---|---|---|---|---|
| Const-velocity, no margin | 5.67% | 0.20% | 0.00% | 21.28 | 100% | -0.01 |
| Koopman, no margin (nominal) | 6.00% | 0.22% | 0.00% | 21.27 | 100% | -0.08 |
| Koopman + fixed 0.30 m margin | 3.00% | 0.11% | 0.00% | 21.45 | 100% | 0.13 |
| **PET-MPC** (Koopman + Markov residual margins) | 0.00% | 0.00% | 0.00% | 21.58 | 100% | 0.26 |

paired_pet_vs_nominal: contact only in A = 18, only in B = 0, both = 0; time B - A = +0.31 ± 0.10 s (95% CI, runs where both reached)

paired_pet_vs_fixed: contact only in A = 9, only in B = 0, both = 0; time B - A = +0.13 ± 0.10 s (95% CI, runs where both reached)

## crowd study (40 paired seeds)

| Controller | Runs with contact | Contact time | Personal-space overlap time | Mean time to goal (s) | Success | 5th pct min clearance (m) |
|---|---|---|---|---|---|---|
| Const-velocity, no margin | 27.50% | 0.68% | 5.78% | 28.05 | 100% | -0.52 |
| Koopman, no margin (nominal) | 25.00% | 0.75% | 6.94% | 29.11 | 100% | -0.44 |
| Koopman + fixed 0.30 m margin | 22.50% | 0.84% | 4.62% | 31.33 | 98% | -0.47 |
| **PET-MPC** (Koopman + Markov residual margins) | 15.00% | 0.42% | 2.42% | 32.76 | 95% | -0.45 |
| PET-MPC, pedestrians treated as generic obstacles | 7.50% | 0.13% | 11.78% | 28.13 | 98% | -0.02 |

paired_pet_vs_nominal: contact only in A = 8, only in B = 4, both = 2; time B - A = +2.79 ± 1.78 s (95% CI, runs where both reached)

paired_aware_vs_generic: contact only in A = 3, only in B = 6, both = 0; time B - A = +4.42 ± 1.77 s (95% CI, runs where both reached)

## Open-loop prediction (20 seeds x 300 steps, 2 s horizon)

| Scenario | Predictor | ADE (m) | FDE @2 s (m) | Error @1 s | Markov margin @1 s | Empirical coverage (target >= 75%) |
|---|---|---|---|---|---|---|
| obstacle | cv | 0.503 | 1.246 | 0.397 | 0.79 | 82.3% |
| obstacle | koopman | 0.553 | 1.343 | 0.446 | 0.81 | 79.5% |
| crowd | cv | 0.356 | 0.901 | 0.275 | 0.65 | 87.7% |
| crowd | koopman | 0.396 | 0.983 | 0.313 | 0.68 | 85.8% |
