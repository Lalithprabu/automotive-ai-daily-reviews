# Full results (1500 held-out scenes: 607 small, 440 big, 453 clear)

Score = NC x DAC x (0.75 EP + 0.25 C) of the trajectory each planner picks (top-8 glance hypotheses -> refine -> re-scored). Collision = share of scenes where the picked trajectory hits an object within 4 s.

| Planner | score | small-hazard | stalled car | clear lane | collision | collision (small) | off-road | patches seen by scorer | rank corr (45) | top-1 among 8 | regret |
|---|---|---|---|---|---|---|---|---|---|---|---|
| keep lane | 0.490 | 0.384 | 0.386 | 0.732 | 0.317 | 0.453 | 0.000 | - | - | - | - |
| oracle anchor | 0.907 | 0.902 | 0.897 | 0.925 | - | - | - | - | - | - | - |
| oracle refined | 0.916 | 0.914 | 0.908 | 0.926 | - | - | - | - | - | - | - |
| glance only (shared tokens) | 0.564 | 0.568 | 0.504 | 0.617 | 0.275 | 0.282 | 0.000 | 1.00 | 0.542 | 0.135 | 0.269 |
| glance only, seed 1 | 0.568 | 0.572 | 0.542 | 0.588 | 0.253 | 0.259 | 0.000 | 1.00 | 0.537 | 0.209 | 0.263 |
| gaze for refinement only | 0.851 | 0.832 | 0.819 | 0.908 | 0.029 | 0.041 | 0.000 | 1.00 | 0.776 | 0.527 | 0.050 |
| gaze for scoring only | 0.861 | 0.851 | 0.824 | 0.911 | 0.025 | 0.025 | 0.000 | 0.17 | 0.782 | 0.559 | 0.042 |
| GGDriver-style: gaze refine + score | 0.869 | 0.853 | 0.847 | 0.912 | 0.022 | 0.033 | 0.000 | 0.17 | 0.796 | 0.655 | 0.034 |
| random patches, matched count | 0.649 | 0.630 | 0.592 | 0.729 | 0.189 | 0.217 | 0.000 | 0.18 | 0.610 | 0.168 | 0.202 |
| attention over all 240 patches | 0.841 | 0.818 | 0.813 | 0.900 | 0.025 | 0.033 | 0.000 | 1.00 | 0.765 | 0.441 | 0.054 |

## Paired differences, 95% CI by scene bootstrap

| Comparison | diff | CI |
|---|---|---|
| glance_s1_minus_glance/score | +0.0038 | -0.0116 .. +0.0187 |
| glance_s1_minus_glance/score_small | +0.0038 | -0.0238 .. +0.0299 |
| glance_s1_minus_glance/collision | -0.0220 | -0.0407 .. -0.0033 |
| gaze_refine_minus_glance/score | +0.2868 | +0.2687 .. +0.3051 |
| gaze_refine_minus_glance/score_small | +0.2640 | +0.2339 .. +0.2938 |
| gaze_refine_minus_glance/collision | -0.2460 | -0.2680 .. -0.2227 |
| gaze_score_minus_glance/score | +0.2972 | +0.2792 .. +0.3156 |
| gaze_score_minus_glance/score_small | +0.2833 | +0.2543 .. +0.3128 |
| gaze_score_minus_glance/collision | -0.2500 | -0.2727 .. -0.2267 |
| gaze_minus_glance/score | +0.3048 | +0.2868 .. +0.3238 |
| gaze_minus_glance/score_small | +0.2850 | +0.2553 .. +0.3163 |
| gaze_minus_glance/collision | -0.2533 | -0.2767 .. -0.2307 |
| random_minus_glance/score | +0.0848 | +0.0664 .. +0.1016 |
| random_minus_glance/score_small | +0.0623 | +0.0324 .. +0.0923 |
| random_minus_glance/collision | -0.0867 | -0.1073 .. -0.0640 |
| all_minus_glance/score | +0.2774 | +0.2593 .. +0.2957 |
| all_minus_glance/score_small | +0.2506 | +0.2227 .. +0.2807 |
| all_minus_glance/collision | -0.2500 | -0.2727 .. -0.2267 |
| gaze_minus_random/score | +0.2201 | +0.2036 .. +0.2363 |
| gaze_minus_random/score_small | +0.2227 | +0.1943 .. +0.2511 |
| gaze_minus_all/score | +0.0274 | +0.0191 .. +0.0357 |
| gaze_minus_all/score_small | +0.0344 | +0.0177 .. +0.0509 |
| gaze_score_minus_gaze_refine/score | +0.0104 | +0.0015 .. +0.0189 |

Glance-only pick (top-1 anchor by glance logit, no refinement): glance 0.569, glance_s1 0.574, gaze_refine 0.842, gaze_score 0.834, gaze 0.852, random 0.628, all 0.799

Simulation clips: {"random_small": 1351, "gaze_win": 1109, "gaze_fail": 194, "n_gaze_win": 161, "n_gaze_fail_vs_glance_ok": 14}
