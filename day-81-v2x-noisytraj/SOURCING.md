# Sourcing

**Paper:** "Robust Trajectory Prediction Under Sensing and V2X Noise: Evaluation and Mitigation Strategies"
(Shahab Salehi, Luca Lusvarghi, Miguel Sepulcre, Javier Gozalvez; cs.CV / cs.RO), arXiv
[2611.28128](https://arxiv.org/abs/2611.28128), listed 24 Nov 2026.

**What I read:** the abstract only, through the arXivSignals mirror (arxiv.org returned rate-limit errors). Author
names and the abstract should be checked against arxiv.org before publishing.

**What the abstract says (paraphrased):**
* Transformer-based, interaction-aware trajectory predictors are normally trained on curated datasets whose noise was
  filtered offline.
* Deployed systems see noisy position, heading and speed estimates, worse for states received over V2X.
* The authors evaluate five state-of-the-art Transformer interaction-aware models under realistic noisy object states.
  Accuracy degrades by up to a factor of 6 as noise grows.
* Retraining on a noise-augmented dataset beats adding input denoising modules and keeps degradation below a factor
  of 1.5. The noise-augmented dataset is released.
* The abstract names no dataset, no specific models and no noise model.

**What is my construction (everything else):** the synthetic highway simulator, the noise model (AR(1) sensor jitter,
GNSS-like V2X bias, packet loss with held values), the noise-level scale 0-4, the single predictor architecture
(temporal Transformer -> social Transformer -> 6-mode decoder), the two denoisers (a constant-velocity Kalman filter
and a learned Transformer denoiser), training schedules, metrics and the degradation definition. Numbers from this
repo are not comparable to the paper's.
