# Day 30 — EgoRefine-style asynchronous collaborative perception (PyTorch)

Reconstruction of the two module ideas in arXiv:2610.00319 (ego-referenced predictive alignment + trajectory-conditioned reliability-aware fusion). **Read `SOURCING.md` first**: the paper's full text was not accessible; architecture details and all numbers here are this repo's own, on synthetic data.

```
day-30-egorefine/
├── egorefine/{model.py,data.py,metrics.py}   # AsyncFusionNet (4 ablation modes), synthetic V2V data, CenterNet focal loss, centre-distance AP
├── tests/test_all.py                          # 12 tests (shapes, identity-init, refine-along-one-direction, gradients, AP, overfit)
├── train.py  eval.py  simulate.py  make_assets.py  config.yaml
└── assets/{architecture.png,results.png}  egorefine_simulation.gif
```
```
pip install -r requirements.txt
pytest tests -q
python train.py            # trains ego_only / naive / traf / egorefine (~25 min CPU)
python simulate.py         # renders egorefine_simulation.gif from the checkpoints
```
Core idea: for each ego cell p, sample the *delayed* collaborator map at K points p+o_k. Old way (TraF-style): o_k come from the collaborator's features alone. New way: a refiner that also sees the **ego** features predicts a unit direction d and signed steps s_k, o_ref = o_base + s_k·d; a gate built from discrepancy(sample_ref vs sample_base) and mean|s_k| down-weights unreliable alignments before fusion.

Key tensor shapes: ego (B,5,40,40), col (B,5,40,40, last ch = noisy delay), offsets (B,K,2,H,W), samples (B,K,C,H,W), logits (B,1,H,W).

Results (synthetic): see `assets/results.png` and `SOURCING.md` — collaboration +11 AP pts; alignment gain within noise.
