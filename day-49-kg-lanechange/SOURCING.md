# Sourcing & disclosure — Day 49

**Paper:** *Decision Making and Ride Comfort Evaluation Based on Lane-Change Prediction for Autonomous Vehicles* — M. Manzour, A. Kumar, R. Izquierdo, M. A. Sotelo (University of Alcalá). arXiv:2610.30528, v1, submitted 2026-10-21. No code release found.

## Paper-sourced (abstract-level, via alphaxiv mirror; arxiv.org abs/html were rate-limited this run)
- Predictor = knowledge-graph embeddings + Bayesian inference; anticipates a lane change ~3–4 s ahead.
- Three strategies in risky CARLA merging scenarios: reactive, prediction-based longitudinal, prediction-based lateral; ISO 2631-1-based horizontal-vibration comfort estimate.
- Reactive driving collides with aggressive deceleration; both predictive strategies avoid collisions in the evaluated records; peak pre-impact deceleration −31.5% (longitudinal) / −91.0% (lateral) vs. reactive. Results come from a limited set of simulated records.

## Precursor-paper-sourced (NOT confirmed to be identical to the 2610 paper's model)
From the same group's *Vehicle Lane Change Prediction based on KG Embeddings and Bayesian Inference* (arXiv:2312.06336): TransE (chosen over ComplEx), embedding size 100, Adam lr 5e-4, self-adversarial loss, highD data, input cues (lateral velocity/acceleration, TTC to preceding / left-right preceding / left-right following), TTC risk bins [0,4] s HIGH / (4,10) s MEDIUM / else LOW, Bayes rule over LLC/LK/RLC hypotheses; macro-F1 97.86 / 97.95 / 93.60 / 66.52 % at 1/2/3/4 s.

## This repo's own reconstruction (every other detail)
Hypothesis-centric triple layout; naive-Bayes factorisation with softmax-over-values likelihood and post-hoc temperature; 5-bin vy/ay discretisation edges; the entire synthetic highway generator (bell-shaped lateral velocity, TTC motive model, noise levels); the 1-D longitudinal cut-in simulator, braking law, delay, debounce, 5% lane-change prior; the MLP baseline. **The paper's lateral-evasion strategy and its ISO 2631-1 comfort metric are NOT reproduced.** No CARLA, no highD.

## Honest findings
- Synthetic macro-F1 (balanced, 1500/horizon): KG-Bayes 0.952 / 0.949 / 0.848 / 0.508 at 1/2/3/4 s; MLP 0.924 / 0.919 / 0.916 / 0.712. KG-Bayes wins at 1–2 s, **loses at 3–4 s** (hard bins discard the weak early signal). Precursor's 4 s F1 is 66.5%; this repo's is lower.
- Dropping both rear-TTC sensors costs ≤0.013 F1 at 1–3 s (rear cues are weakly informative in this generator).
- Closed loop (200 runs/gap, own sim): at a 7 m crossing gap reactive collides 100% (peak decel pinned at the 9.0 cap, so "9.0" is the cap, not a measurement), predictive 0% at mean peak decel 2.46. At 14 m and 10 m neither collides, but reactive still brakes at the cap while predictive brakes at 1.8–2.1 m/s². Scenarios were chosen by me.
- Without debouncing, a lane-keeping neighbour triggered false braking in 26% of 11-s runs (5% prior); a 6-frame debounce brought this to 0%. The test reuses the training generator's LK features, so it is optimistic.
- Early iteration without a lane-change prior braked at t≈0.5 s, ~5 s before crossing, because the sim cutter's TTC "motive" cues leak intent; fixed with the 5% prior + debounce.
