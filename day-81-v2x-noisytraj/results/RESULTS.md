# Full results (3000 held-out scenes x 8 agents, 2 seeds per learned pipeline)

minADE6 in metres (best of 6 modes, averaged over all agents). Kinematic baselines have one mode.

| Pipeline | level 0 | level 0.5 | level 1 | level 2 | level 3 | level 4 | degradation at 4 |
|---|---|---|---|---|---|---|---|
| constant velocity (raw) | 0.888 | 1.290 | 1.787 | 2.878 | 4.001 | 5.178 | 5.83x |
| constant velocity + Kalman | 1.164 | 1.394 | 1.717 | 2.441 | 3.207 | 3.970 | 3.41x |
| clean-trained | 0.253 | 0.674 | 1.040 | 1.652 | 2.217 | 2.828 | 11.19x |
| clean-trained + Kalman filter | 0.344 | 0.595 | 0.894 | 1.447 | 1.963 | 2.455 | 9.71x |
| clean-trained + learned denoiser | 0.425 | 0.470 | 0.570 | 0.871 | 1.227 | 1.596 | 6.31x |
| noise-augmented retrain | 0.329 | 0.370 | 0.436 | 0.618 | 0.849 | 1.068 | 3.25x |

## minFDE6 (m)

| Pipeline | level 0 | level 0.5 | level 1 | level 2 | level 3 | level 4 |
|---|---|---|---|---|---|---|
| constant velocity (raw) | 2.441 | 3.039 | 3.838 | 5.648 | 7.528 | 9.474 |
| constant velocity + Kalman | 2.905 | 3.201 | 3.648 | 4.686 | 5.798 | 6.910 |
| clean-trained | 0.614 | 1.208 | 1.737 | 2.596 | 3.319 | 4.077 |
| clean-trained + Kalman filter | 0.749 | 1.057 | 1.437 | 2.132 | 2.773 | 3.352 |
| clean-trained + learned denoiser | 0.899 | 0.948 | 1.060 | 1.422 | 1.868 | 2.350 |
| noise-augmented retrain | 0.731 | 0.781 | 0.860 | 1.089 | 1.354 | 1.613 |

## minADE6 by agent source at the top level

| Pipeline | sensor-tracked | V2X-received | top-1 ADE |
|---|---|---|---|
| constant velocity (raw) | 3.612 | 7.116 | 5.178 |
| constant velocity + Kalman | 2.726 | 5.513 | 3.970 |
| clean-trained | 2.032 | 3.811 | 4.122 |
| clean-trained + Kalman filter | 1.703 | 3.389 | 3.468 |
| clean-trained + learned denoiser | 1.102 | 2.214 | 2.674 |
| noise-augmented retrain | 0.703 | 1.530 | 2.273 |

## Per-seed minADE6

| Pipeline | level 0 | level 0.5 | level 1 | level 2 | level 3 | level 4 |
|---|---|---|---|---|---|---|
| clean-trained seed 0 | 0.217 | 0.759 | 1.157 | 1.770 | 2.296 | 2.848 |
| clean-trained seed 1 | 0.288 | 0.588 | 0.923 | 1.534 | 2.138 | 2.808 |
| noise-augmented retrain seed 0 | 0.324 | 0.364 | 0.429 | 0.612 | 0.845 | 1.063 |
| noise-augmented retrain seed 1 | 0.334 | 0.376 | 0.442 | 0.623 | 0.854 | 1.073 |

## t0 position error of the history fed to the predictor (m, mean over agents)

| Input | level 0 | level 0.5 | level 1 | level 2 | level 3 | level 4 |
|---|---|---|---|---|---|---|
| raw | 0.000 | 0.237 | 0.466 | 0.918 | 1.396 | 1.885 |
| kalman | 0.050 | 0.217 | 0.407 | 0.800 | 1.202 | 1.605 |
| learned | 0.105 | 0.172 | 0.279 | 0.543 | 0.842 | 1.138 |

## Paired differences in minADE6, 95% CI by scene bootstrap

| Comparison | diff (m) | CI |
|---|---|---|
| aug_minus_clean@4 | -1.760 | -1.785 .. -1.736 |
| aug_minus_clean_dn@4 | -0.528 | -0.539 .. -0.516 |
| aug_minus_clean_kf@4 | -1.387 | -1.407 .. -1.367 |
| clean_dn_minus_clean_kf@4 | -0.859 | -0.877 .. -0.841 |
| aug_minus_clean@0 | +0.076 | +0.073 .. +0.080 |
| aug_minus_clean_dn@1 | -0.134 | -0.139 .. -0.130 |
| aug_minus_clean_dn@2 | -0.253 | -0.259 .. -0.246 |

Simulation clips: {"random": 172, "win": 1345, "loss": 1919, "n_win": 2725, "n_loss": 96, "n_top_level_worse": 0}
