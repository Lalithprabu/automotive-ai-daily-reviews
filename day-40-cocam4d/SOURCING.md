# Sourcing & disclosure (Day 40)

**Paper-sourced** (arxiv.org/html/2610.11577v1, abs page; abstract only summarised by the fetch tool, no verbatim text):
title, 4 authors + affiliations (ISI Kolkata, UPES Dehradun, Valeo Brain/anSWer India), submitted 2026-10-08, cs.RO/cs.CV;
VGGT-based feed-forward network producing 3D Gaussians with uncertainty; six heads (camera, depth, uncertainty, Gaussian, motion, sky);
35-byte DOP (mu 12 B, v 6 B, theta 2 B, d 6 B, diag R 9 B), ~1,400 B per 10 Hz C-V2X message (~40 objects);
fusion: spatial alignment -> velocity latency compensation -> Hungarian on Mahalanobis cost, chi2(3,0.95)=7.815 -> additive information filter;
headline: +11.48% (OPV2V+) and +10.62% (DAIR-V2X-C) over recent vision-only methods; OPV2V+ N=2 AP@0.5 BEV 57.3 (0-50 m) at 58 KB/s;
robustness at 500 ms: 53.7 AP@0.5 vs CoCa3D 38.4 / Where2comm 32.8; ablations (N=3): w/o uncertainty head 55.2 vs 60.4 full AP@0.5.
No code release found. No limitations section; the paper's numbers are quoted only as reported.

**This repo's own reconstruction (NOT from the paper):** operating on tracked-cluster centroids instead of VGGT pixels/Gaussians; the
synthetic scene generator and noise model (depth sigma = 0.0005 D^2 + 0.05); all model dimensions; window-3 reading of "frame-adjacent
causal mask"; Gaussian-NLL velocity-variance head (paper uses an analytic sigma_v formula); L1 positional-sigma head (follows Eq. 7 idea);
the layout of the last 9 DOP bytes; the process-noise constant q_acc; baselines (stale late fusion etc.); every number in README/results.json.
Not reproduced: depth/camera/Gaussian/sky heads, rendering losses, lifespan, Gaussian-union shape completion, static-Gaussian sharing.

**Known limitations of this reconstruction:** collaborator poses are noisy-but-known; constant-velocity objects only; sigma coverage at
1.96 sigma is ~88%, i.e. mildly over-confident; learned uncertainty adds only a small gain over a fixed covariance.
