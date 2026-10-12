# Sourcing & disclosure — Day 38 (WZDrive)

**Paper:** Trajectory Prediction for Autonomous Driving in Road Construction Zones Using Human-in-the-Loop Driving Data — Shuo Zhang, Aman Sunesh, Semiha Ergan, Kaan Ozbay (NYU). arXiv:2610.12747, cs.RO, submitted 2026-10-08 (4 days old at run time).

**Access:** `arxiv.org/abs` and `arxiv.org/html` returned HTTP 429 with a do-not-retry instruction. The alphaxiv.org mirror gave the abstract-level facts only. No code release found.

## Paper-sourced (only these)
- Task: ego-trajectory prediction in urban roadway work zones; WZDrive = "multimodal ego-trajectory predictor".
- WZDrive-Data: 1,183 segments, 33 drivers; human-in-the-loop driving simulator, prerecorded worker animations + VR-controlled workers; driver control and gaze signals, vehicle trajectories, scene information.
- Metrics/results: minADE@6 = 1.09 m, minFDE@6 = 2.12 m; 7.18% (minADE) and 13.18% (minFDE) lower than the best SOTA baseline per metric.

## This repo's own reconstruction (NOT paper-sourced)
Everything else: the GRU/transformer architecture, GazeConditionedSceneAttention, K=6 mode queries, WTA loss, all dimensions, the synthetic generator (4 behaviours: KEEP / MERGE_EARLY / MERGE_LATE / YIELD), the gaze-leads-steering assumption, and every number in `results.json`. Whether the paper's model actually consumes gaze/control as inputs is not confirmed by the retrievable text.

## Findings reported as they came out
- Top-1 ADE 0.855 → 0.430 m (−49.6%), driven by MERGE_LATE (2.02 → 0.38 m): the baseline picks the early-merge mode because it cannot see the gaze cue.
- minADE@6 did NOT improve (0.311 vs 0.335, −7.8%) and minFDE@6 is flat (+1.0%). K=6 covers the generator's four intents, so coverage saturates. The paper's 7.18% minADE gain is not reproduced here.
- Top-3 minADE also slightly worse (−10.1%).
- Single seed, single run; WZDrive minADE@6 varied roughly 0.33–0.50 across earlier runs with different budgets/settings. Treat small differences as noise.

## Bugs / lessons
1. Overfit-tiny-batch test failed with loss normalised by /20 (sub-metre errors under-weighted); /5 fixed it.
2. Raw gaze yaw (~0.2 rad) was too small a signal for the GRU; rescaled ×5 inside the model.
3. First headline (minADE@6) hid the real effect; added ranked top-k metric. The core-claim test asserts top-1 ADE, not minADE@6.
