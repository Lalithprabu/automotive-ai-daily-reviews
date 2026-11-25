# Sourcing: what comes from the paper and what is mine

**Paper:** "SpecFlow: Flow Matching in the Acceleration Spectrum for Multi-Agent Trajectory Forecasting"
(Ziyang Xu, Peng Han, Hai Zhou; cs.RO), [arXiv 2611.26390](https://arxiv.org/abs/2611.26390), listed 20 Nov 2026.
I read the **abstract only** (through the arXivSignals mirror, because arxiv.org was rate-limited). The authors say code
"will be made publicly available"; it was not used here.

## From the abstract
* Generative trajectory forecasters usually generate positions frame by frame, which leaves the position / velocity /
  acceleration relationship implicit.
* SpecFlow's generative state is the **DCT coefficients of future accelerations**. Conditional flow matching learns a
  transport from Gaussian noise to these acceleration spectra.
* Velocity and position come from **fixed analytical integration** starting at the last observed state.
* A **joint phase-space loss** supervises all three kinematic levels.
* An **endpoint predictor** proposes candidate destinations with confidences from the observed context only. The
  **top candidate conditions** the spectral denoiser.
* **Alternating attention over hypotheses and agents** handles multimodality and interaction.
* Results (best-of-20): ETH/UCY average minADE/minFDE 0.15/0.22 (best on all five scenes), NBA 4 s 0.69/0.86 (lowest ADE, tied-best FDE), SDD 7.38/11.57 ("competitive").

## My construction (not in the abstract)
* The synthetic 4-way intersection (vehicles + pedestrians, lights, IDM-lite following, 0.03 m noise), dt = 0.4 s,
  8 observed / 12 future frames, 24k / 1k / 4k scenes. None of the paper's datasets are used.
* Full 12-coefficient orthonormal DCT-II per axis, per-coefficient normalisation, semi-implicit Euler integration
  (v_k = v_0 + dt Σa, p_k = dt Σv), v_0 = last observed finite difference.
* The encoder (GRU + 2 social-attention layers with relative-pose keys), D = 128, K = 6 endpoint candidates,
  winner-takes-all endpoint loss, training condition = the candidate closest to the truth (detached).
* Rectified-flow path, one shared t per scene, 6 hypotheses per agent in training, 10 Euler steps and 20 hypotheses
  at test time.
* Phase-space loss = L1 on (a, v, p) of the one-step estimate x̂₁ = x_t + (1 − t) û, weights 0.05 / 0.1 / 0.1.
* The ablations (pos_state, acc_time, no_phase, no_endpoint), the confidence-stratified endpoint extension,
  the constant-velocity baseline, the kinematic-realism metrics and the few-step study.

The numbers in this folder are **not comparable** to the paper's.
