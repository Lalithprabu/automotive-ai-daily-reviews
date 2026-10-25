# Day 51 — Foundation4D-Mini: one model for depth, 3D detection, occupancy and Gaussians

PyTorch reconstruction of the *shape* of NVIDIA's **Foundation4D** (arXiv:2610.30340, 2026-10-21): a single feed-forward
network over multi-camera driving video whose shared backbone uses **factorized spatio-temporal attention** and feeds four heads.
Trained and evaluated on a **fully synthetic ray-cast rig** — these are NOT the paper's numbers. See `SOURCING.md` first.

```
day-51-foundation4d/
├── config.yaml            # all hyper-parameters (decimal floats only — Day 16 YAML lesson)
├── train.py               # trains joint model, detection-only "old way", dense-attention ablation -> results.json
├── simulate.py            # 4-camera + BEV + occupancy + telemetry GIF from the trained checkpoint
├── src/
│   ├── foundation4d.py    # FactorizedSTBlock, JointBlock, Foundation4DMini, losses, attention_pairs()
│   ├── synthetic.py       # exact ray-cast 4-camera scenes with analytic depth / BEV / occupancy labels
│   └── metrics.py         # AbsRel, calibrated detection F1, occupancy IoU
├── tests/test_foundation4d.py   # 8 tests (geometry, cross-view/temporal flow, grads, overfit)
├── assets/foundation4d_simulation.gif
├── SOURCING.md  LINKEDIN_POST.md  SIMULATION_PROMPT.md  results.json
```

## Run
```bash
pip install torch matplotlib pillow pyyaml pytest
pytest tests -v                      # 8 passed
python train.py --config config.yaml # ~17 min CPU (3 models)
python simulate.py                   # -> assets/foundation4d_simulation.gif
```

## Core idea (src/foundation4d.py)
Tokens `[B, V cameras, T frames, N patches, D]`. Each block does three cheap attentions instead of one dense one:
1. **spatial** — within one image (length N)  2. **cross-view** — across cameras, same frame (length V·N)
3. **temporal** — across frames, same camera & patch slot (length T).
Heads: depth (log-depth per pixel), per-pixel Gaussians (opacity/scale/RGB), and a **geometry-guided BEV lift**
(project each BEV cell into every camera, bilinear-sample features) feeding detection heat-map + 3D occupancy.

## Results on held-out synthetic scenes (n=128; NOT comparable to the paper)
| Model | Det F1 | Occ IoU | Depth AbsRel | Params | Train s |
|---|---|---|---|---|---|
| Joint, factorized attention (new way) | 0.884 | 0.489 | 0.077 | 672K | 389 |
| Detection-only (old way: specialist) | 0.883 | — | — | 672K* | 398 |
| Joint, dense attention | 0.900 | 0.480 | 0.082 | 272K | 198 |

\*unused heads still instantiated. Attention pairs/layer: factorized 92,928 vs. dense 147,456 (−37%).
**Honest read:** co-training gave no detection gain here (paper: +1.2 mAP on real data), and dense attention was as good at
this tiny token count; the factorization win is analytic and grows with scale. Details in `SOURCING.md`.
