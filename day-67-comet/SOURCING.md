# Sourcing & disclosure (Day 67)

**Paper-sourced** (arXiv abs page + alphaxiv overview; full-text HTML hit HTTP 429 / was never read):
title, authors (Shuwei Sun, Jian Huang, Hui Cao), cs.CV, submitted 6 Nov 2026, "25 pages, 4 figures", code on Zenodo (10.5281/zenodo.23999418);
shared-translation-error idea, per-sender Kalman filter, no learned parameters, matching of mutually seen objects, correction applied to all roadside boxes, a separate frozen head that ranks merged detections;
V2X-Seq, BEV AP@IoU0.7, delays 0-400 ms, 21 validation sequences: CoMET 66.63 vs 63.18 (online poses) vs 62.96 (best intermediate fusion); +2.2..+3.5 AP over online poses across 5 detectors / 3 architectures, +0.3..+0.6 over current-message-only; 28% of validation messages use a zero-offset placeholder, 45% of the rest hit the search boundary; 38% of messages have no matched pair at zero delay.

**Not paper-sourced (this repo's own choices):** process/measurement noise values, isotropic scalar covariance, median measurement, gate formula, mutual-NN details, the whole synthetic world, the F1@1 m metric, all toy numbers. The paper's exact filter equations were not read. The frozen ranking head is NOT implemented.

**Tuning disclosure:** `p0`, `gate`, `gate_sigma_k` were adjusted on seeds 1000-1009 after a first config produced offset RMSE worse than no correction (a wide start gate let a wrong pair set the offset). Final numbers use unseen seeds 5000-5019.

**Toy results** (20 seeds x delays 0/100/200/400 ms, F1@1.0 m, ego+roadside fused): uncorrected 0.342, current-message-only 0.683, CoMET 0.884, oracle offset 0.927; offset RMSE (200 ms) 2.03 / 1.27 / 0.44 m; 24% of messages unmatched. Single toy world, hand-set noise: relative ordering is the claim, not the magnitudes.
