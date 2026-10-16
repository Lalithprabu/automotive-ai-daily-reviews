# Day 42 — RadarGSOcc-style Radar Semantic Occupancy (PyTorch reconstruction)
Review of arXiv:2610.18686 (RadarGSOcc: Doppler-Aware 3D Semantic Occupancy Prediction Using 4D Radar Tensor...). **Read SOURCING.md first**: only abstract-level details are paper-sourced; this is a small synthetic-data reconstruction and did **not** reproduce the paper's gain.

## Structure
```
day-42-radargsocc/
├── config.yaml
├── models/radargsocc.py   # DeformAttn3D, AttnBlock, RadarGSOcc, RadarOccBaseline
├── src/data.py            # synthetic 8-channel radar tensor -> 5-class occupancy (Doppler physics)
├── src/metrics.py         # occ-IoU, weighted mIoU, simplified EBKL
├── train.py               # trains new + old on identical data
├── simulate.py            # GIF: radar input | GT | old | new + live IoU telemetry, snow in 2nd half
├── tests/test_model.py    # 8 tests (shapes, attention axis order, gradients, overfit, metrics)
└── assets/simulation.gif
```
## Run
`pip install torch numpy matplotlib pillow pytest pyyaml && pytest tests && python train.py && python simulate.py`
(train.py: ~25 min CPU, 2 cores.)

## Results (synthetic, single seed — not the paper's numbers)
| | params | occ-IoU | w-mIoU | vehicle IoU | occ-IoU (snow) |
|---|---|---|---|---|---|
| Old (early-fusion conv) | 138,925 | 0.970 | 0.968 | 0.700 | 0.731 |
| RadarGSOcc-style | 84,813 | 0.965 | 0.963 | 0.668 | 0.731 |

A first, easier dataset gave ~0.998 for both (no separation) and was hardened. Tests: 8/8 pass.

## Core layer
`DeformAttn3D`: each voxel query predicts K offsets + softmax weights, samples the value map with trilinear `grid_sample`, and sums: out(q)=Σ_k a_qk · V(p_q + d_qk). Self-attention when value==query; cross-attention when value is the other branch.
