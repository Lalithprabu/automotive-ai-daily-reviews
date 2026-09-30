# Day 26 — WALT: World-Model-Aligned Latent Trajectories (arXiv:2609.30436)
Reconstruction of the core idea: a trajectory autoencoder whose latent space is enriched/aligned with features from a frozen driving world model, and a planner that predicts in that latent space instead of raw waypoints.
```
day-26-walt/
├── config.yaml  train.py  simulate.py  SOURCING.md  LINKEDIN_POST.md  results.json
├── walt/  data.py (synthetic scenes)  model.py (FrozenWorldModel, TrajAutoencoder, LatentPlanner, WALTPlanner, DirectPlanner)
├── tests/test_walt.py   (6 tests)
└── assets/walt_simulation.gif
```
Run: `pip install torch pytest matplotlib pillow pyyaml && pytest tests && python train.py && python simulate.py`
Key shapes: raster (B,2,32,32) → WM tokens (B,64,32) → planner latent z (B,4,32) → decoder → waypoints (B,8,2).
Result (synthetic, 500 val scenes, 1 seed): OLD raw 3.25 m ADE | WALT aligned latent 3.48 | geo-only latent 2.52. **Paper's claim not reproduced in this toy** — see SOURCING.md.
