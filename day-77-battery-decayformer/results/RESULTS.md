# Full results (synthetic toy, 288 cells, 48 aging conditions, 6 condition-exclusive folds)

Each cell is forecast by the model of the fold that held its condition out. One run per fold, one fixed config (the paper's protocol). Forecast from the first 100 cycles unless stated.

| Variant | params | traj MAPE % | late-life MAPE % (SOH<0.9) | lifetime MAPE % | non-monotone curves |
|---|---|---|---|---|---|
| decayformer | 120,465 | 1.06 | 3.06 | 16.8 | 0% |
| df_freehead | 126,141 | 0.88 | 2.63 | 33.0 | 49% |
| df_nosoc | 119,185 | 1.34 | 3.59 | 20.5 | 0% |
| df_noattn | 54,289 | 0.98 | 2.87 | 16.4 | 0% |
| transformer | 679,869 | 0.88 | 2.62 | 41.8 | 47% |
| gru | 182,973 | 0.86 | 2.55 | 36.1 | 42% |
| sqrt_fit | 4 (fit) | 0.99 | 3.40 | 68.8 | 0% |

## DecayFormer minus other (negative = DecayFormer better), 95% CI by condition bootstrap

| vs | traj MAPE (pts) | lifetime MAPE (pts) |
|---|---|---|
| df_freehead | +0.18 (+0.12 .. +0.25) | -16.1 (-21.0 .. -11.7) |
| df_nosoc | -0.28 (-0.41 .. -0.16) | -3.6 (-5.9 .. -1.5) |
| df_noattn | +0.09 (+0.04 .. +0.14) | +0.4 (-0.6 .. +1.5) |
| transformer | +0.19 (+0.11 .. +0.26) | -24.9 (-32.2 .. -18.4) |
| gru | +0.20 (+0.13 .. +0.27) | -19.2 (-24.8 .. -14.2) |
| sqrt_fit | +0.07 (-0.03 .. +0.17) | -52.0 (-69.6 .. -35.9) |

## Knee-dominated cells (knee > 25% of the fade at EOL, n = 164) vs the rest

| Variant | knee traj | knee life | other traj | other life |
|---|---|---|---|---|
| decayformer | 1.38 | 21.5 | 0.65 | 10.6 |
| df_freehead | 1.19 | 47.8 | 0.47 | 13.4 |
| df_nosoc | 1.66 | 25.4 | 0.91 | 13.9 |
| df_noattn | 1.30 | 21.9 | 0.55 | 9.0 |
| transformer | 1.18 | 62.5 | 0.47 | 14.4 |
| gru | 1.18 | 53.7 | 0.44 | 12.7 |
| sqrt_fit | 1.45 | 109.4 | 0.38 | 15.1 |

## Observed prefix length (traj MAPE % / lifetime MAPE %)

| Variant | 25 cycles | 50 cycles | 100 cycles |
|---|---|---|---|
| decayformer | 1.24 / 21.1 | 1.12 / 18.3 | 1.06 / 16.8 |
| df_freehead | 1.10 / 40.3 | 0.95 / 33.9 | 0.88 / 33.0 |
| df_nosoc | 1.49 / 22.3 | 1.37 / 21.0 | 1.34 / 20.5 |
| df_noattn | 1.26 / 21.4 | 1.05 / 18.3 | 0.98 / 16.4 |
| transformer | 1.11 / 46.6 | 0.97 / 42.0 | 0.88 / 41.8 |
| gru | 1.06 / 44.6 | 0.93 / 39.8 | 0.86 / 36.1 |
| sqrt_fit | 2.19 / 72.0 | 1.33 / 69.5 | 0.99 / 68.8 |

## Per-fold traj MAPE % (one fixed config, no per-fold tuning)

| Variant | f0 | f1 | f2 | f3 | f4 | f5 |
|---|---|---|---|---|---|---|
| decayformer | 1.02 | 0.91 | 1.03 | 1.12 | 1.32 | 0.97 |
| df_freehead | 0.82 | 0.82 | 0.86 | 1.02 | 0.88 | 0.89 |
| df_nosoc | 1.42 | 1.08 | 1.42 | 1.62 | 1.27 | 1.23 |
| df_noattn | 0.96 | 0.86 | 0.93 | 1.07 | 1.06 | 0.97 |
| transformer | 0.83 | 0.88 | 0.81 | 1.01 | 0.88 | 0.86 |
| gru | 0.78 | 0.83 | 0.82 | 0.99 | 0.90 | 0.85 |
| sqrt_fit | 0.95 | 1.06 | 0.86 | 1.10 | 0.91 | 1.06 |

CPU latency per cell (batch 1, 100 cycles): decayformer 2.7 ms, transformer 2.8 ms, gru 4.9 ms
