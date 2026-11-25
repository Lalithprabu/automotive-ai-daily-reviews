# Sourcing

**Paper:** "RadarOSV: Unified 3D Scene and Motion Perception from mmWave Radars", Yuan Fang, Long Zhuang, Xijia Wei,
Jingqi Jiang, Fangzhan Shi, Simon Julier, Kevin Chetty (cs.CV, cs.RO). arXiv 2611.30078, listed 24 Nov 2026.

**What I read:** the abstract only, via the arXivSignals listing (arxiv.org was rate-limited during this run). No code
release was flagged.

**Taken from the abstract:**
* Radar-only joint prediction of 3D occupancy, dynamic-object semantics and ground-relative velocity in unified voxels.
* Two radars: a cascaded imaging radar (fine spatial detail, limited unambiguous velocity range) and a single-chip
  radar (wider velocity range, coarser resolution).
* A Polar-to-Cartesian Geometry (PCG) backbone with fixed and deformable sampling.
* Single-chip detections give class evidence; raw Doppler spectra give motion features and radial-velocity
  candidates; the velocity readout selects among candidates to resolve aliasing.
* Ego-motion compensation from radar-estimated velocity, no external odometry or extra sensors at inference.
* Reported numbers: K-Radar single-frame PCG 40.7% mIoU within 12.8 m (RadarOcc: 34.3%). BRIDGE-Liv, against a
  cascaded-radar baseline given external ego velocity: dynamic macro accuracy 63.1% -> 75.5%, radial-velocity MAE
  1.03 -> 0.51 m/s, geometric occupancy mIoU maintained.

**My construction (not from the paper):** the street simulator, both radar models (bins, ranges, noise, micro-Doppler,
40% single-chip dropout, no occlusion), the 2D BEV grid instead of 3D voxels, all layer sizes, the phase + candidate
parameterisation of the velocity readout, the losses, the training schedule, and every number in results/.
None of my numbers are comparable to the paper's.
