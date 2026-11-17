# Results (synthetic toy world; held-out seeds 100-119)

## Open-loop multi-step prediction (horizon 0.75 s), RMS lateral error e_y [cm], mean over seeds

| Test friction | matched data | mismatched data | pooled data | mismatched / matched | pooled / matched | pooled share of |g| on mu_high columns |
|---|---|---|---|---|---|---|
| mu = 1.0 (dry) | 0.93 | 3.64 | 1.90 | 3.94x | 2.05x | 0.52 |
| mu = 0.5 (low) | 1.79 | 4.82 | 2.40 | 2.69x | 1.34x | 0.48 |

Paper (abstract): pooled within 1.5x of matched; mismatched 2.2x-3.4x.

## Closed-loop tracking, 2 laps, RMS |e_y| [cm] (mean, 95% bootstrap CI) / mean max |e_y| [cm] / off-track runs

| Track | dpc_high | dpc_low | dpc_pool | dpc_oracle | model_mpc_true_mu |
|---|---|---|---|---|---|
| uniform_high | 0.68 [0.65, 0.72] / 2.4 / 0 | 1.89 [1.80, 2.00] / 7.4 / 0 | 1.09 [1.00, 1.19] / 3.1 / 0 | 0.68 [0.65, 0.72] / 2.4 / 0 | 0.38 [0.38, 0.39] / 1.8 / 0 |
| uniform_low | 4.38 [4.31, 4.45] / 17.8 / 0 | 2.06 [1.93, 2.19] / 9.6 / 0 | 3.15 [2.99, 3.34] / 14.4 / 0 | 2.06 [1.93, 2.19] / 9.6 / 0 | 2.74 [2.73, 2.75] / 14.1 / 0 |
| mixed | 1.52 [1.48, 1.56] / 5.1 / 0 | 1.77 [1.69, 1.86] / 7.3 / 0 | 1.15 [1.08, 1.23] / 3.2 / 0 | 0.73 [0.69, 0.76] / 2.5 / 0 | 0.43 [0.43, 0.44] / 1.8 / 0 |

Mixed track, paired: pooled beats high-only data in 20/20 seeds and beats the better of the two single data sets (chosen in hindsight per seed) in 20/20.

Paper (abstract): sim tracking DPC pooled 13.7 cm vs NMPC 13.3 cm; CPM Lab low-mu 1.6 vs 1.5 cm, high-mu 0.9 vs 1.1 cm.
