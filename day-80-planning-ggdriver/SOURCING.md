# Sourcing

**Paper:** "GGDriver: Glance at What Is Shared, Gaze at What Differs for Perception-Free Planning"
(Jiawei Sun, Maosen Li, Jiamin Shi, Xiaoxu Zheng, Jiun Tian Hoe, Yechao Bai, Michael Bi Mi), arXiv 2611.27330,
listed 23 Nov 2026 on the arXivSignals mirror (arxiv.org was rate-limited during this run).

**What I used: the abstract only, paraphrased.** It states:
* Many end-to-end planners score every candidate trajectory against the same compressed scene representation.
* GGDriver: a global glance proposes trajectory hypotheses from shared scene features; a corridor around each
  trajectory is projected into the camera views and the mask selects patch features at the backbone's native
  resolution; after refinement, the scorer re-projects each final proposal and ranks it on evidence along its own path.
  Only the predicted trajectory and camera calibration are needed.
* It improves three planners with different scene representations (learned registers, a perception-supervised BEV
  grid, a coarse image-token grid).
* Ablations: path-aligned patches beat a matched number of random patches and attention over all patches; gazing for
  ranking helps more than gazing only for refinement.
* With a ViT-S backbone and no perception labels: 8.110 RFS on WOD-E2E test, 94.2 PDMS on NAVSIM-v1, 48.6 EPDMS on
  NAVSIM-v2, 72.18 Driving Score on Bench2Drive.

**My construction (not from the paper):** the synthetic one-camera simulator, static objects, the 45-anchor (d, v)
vocabulary, the PDMS-like rule score, the corridor geometry (1.3 m half-width, 2 m tall, from 6 m ahead, one-patch
horizontal dilation), the conv backbone, 8 registers, every head, every loss, every size, the top-8 selection, and all
ablation variants. Numbers in this repo are NOT comparable to the paper's.

I did not read or use the authors' code.
