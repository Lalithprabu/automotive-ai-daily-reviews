# Sourcing - what comes from the paper and what is mine

**Paper:** *Multimodal Masked Diffusion Autoencoder for Modeling Impaired Driving*, John Gideon, Anshul Gupta,
Megan Applegate-Kenton, Kayli Battel, Todd J. Rowell, Patricio Reyes Gomez, Laporsha Dees, Emily Sumner,
Avinash Balachandran, Guy Rosman. arXiv [2611.23211](https://arxiv.org/abs/2611.23211) (cs.CV / cs.LG / cs.RO),
listed 18 Nov 2026. The abstract says code and data will be released on acceptance. I found no code.

I worked from the **abstract only** (read through arXivSignals; arxiv.org returned rate-limit errors every time I
tried the abstract page).

## From the abstract
* Goal: infer the driver's cognitive state / impairment from gaze, vehicle behaviour and scene, because labels are
  expensive to collect at scale.
* ID-MAE: a multimodal masked-diffusion autoencoder. It **predicts the impairment label as a masked token** and
  **reconstructs masked gaze and trajectory** as a self-supervised objective.
* DIVERT: a new simulator dataset, 103 h, 109 participants, ten impairment inductions (cognitive, visual,
  emotional), with gaze, vehicle state and scene video.
* Paired AUC on DIVERT (location-matched pairs): **0.706 with the reconstruction loss vs 0.590 without**.
* It also claims strong trajectory and gaze prediction, ablations over modalities, masking, positional encoding,
  attention and decoding, and that visually demanding distractions can be separated from cognitive ones. The
  abstract gives no numbers for these.

## My construction (not in the abstract)
* **Data:** a synthetic driving simulator (`idm/data.py`) with 4 conditions instead of 10 inductions, per-driver
  style, and per-driver susceptibility from 0.1 to 1.0. Its gaze model has fixations on the road and areas of
  interest. Its lateral control uses delay and preview, and it has a speed controller. Location matching drives
  the same road from the same start once per condition. 90 training drivers and 30 test drivers, 30 locations
  each, 10 s windows at 10 Hz. Scene video is replaced by a road-curvature stream.
* **Tokenisation:** 0.5 s patches per stream, plus a modality embedding, a sinusoidal time embedding and one label
  token.
* **Masked-diffusion schedule:** noise level t ~ U(0.1, 1). Gaze, vehicle and label tokens are each masked
  independently with probability t. The label loss is CE on masked label tokens, and the reconstruction loss is
  MSE on masked patches. lambda = 1. I do not use the MDLM 1/t weighting.
* **Sampler:** forecasts are produced by iterative unmasking of future patches, nearest first, in K = 4 steps. I
  also report one-shot (K = 1).
* **Partial labels (10%)** with half-labelled batches, the GRU baseline, the kinematic forecast baselines, the
  linear probe for visual vs cognitive, the susceptibility breakdown, and label-conditioned forecasting.
* **Sizes:** d = 96, 3 layers, 4 heads, about 0.26 M parameters. 1,000 AdamW steps with batch 128 and 3 seeds.

Absolute numbers from this toy are **not comparable** to DIVERT. The toy is useful for checking the direction of
each claim, not the size of the effect.
