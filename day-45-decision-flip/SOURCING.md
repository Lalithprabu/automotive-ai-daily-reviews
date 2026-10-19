# Sourcing & disclosure (Day 45)
**Paper:** Decision-FLIP: Budget-Free Structured Pruning for Gaussian Occupancy Models — J. Li, A. Paul, S. Das, C. Xu. arXiv:2610.20917 (cs.CV), submitted 2026-10-14.
**Access:** arxiv.org/abs abstract fetched directly; arxiv.org/html returned HTTP 429 ("do not retry") so full text was NOT read. alphaxiv gave abstract-level detail only.
**Paper-sourced (abstract only):** Mean-FLIP idea (count voxel decisions flipped when a row is removed, unlabeled calibration inputs), guarded Otsu survivor selection that can abstain, physical compilation of a smaller checkpoint, streaming lifecycle aggregation, headline numbers (12-79% rows removed, 7-45% bank latency cut, <=8.0% whole-model, <=16.6% peak memory, <=0.18 mIoU change, label-free matches/beats supervised Taylor), limitation (needs stable row identity; not parameter-free).
**This repo's own reconstruction (NOT from the paper):** the entire model (384-row Gaussian bank, MLP refinement, additive splat readout), synthetic 16x16x4 scenes, Otsu "valley" guard, baselines, all numbers below. Streaming lifecycle aggregation is NOT implemented.
**No paper metric is claimed to be reproduced.**
