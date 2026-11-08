# Day 64 - VFlash: Flash Primitive Decoding (toy reconstruction)
Paper: arXiv 2611.08449 (cs.RO, submitted 2026-11-05). Reconstructed from the abstract only (full text not retrievable); drafter internals are my own minimal design.
```
vflash/model.py   VFlashDrafter (one-pass) + ARBaseline (L-pass)
vflash/data.py    synthetic lead-vehicle forecasting task
vflash/train.py   python -m vflash.train 600
simulate.py       python simulate.py  -> assets/vflash_sim.mp4
tests/            pytest -q
```
Results: see RESULTS section appended after running train. This is a synthetic toy; numbers are NOT the paper's.

## RESULTS (toy, CPU, single seed, 600 steps - NOT the paper's numbers)
| | ADE | latency, 1 scene | latency, 512 scenes |
|---|---|---|---|
| AR baseline (16 passes, teacher-forced training) | 0.212 | 21.5 ms | ~500 ms |
| VFlash-style drafter (1 pass) | 0.088 | 2.0 ms | ~55 ms |

The ADE gap is inflated by AR exposure bias (teacher forcing -> free-running rollout); one seed, no significance testing. Paper claims (abstract): lower error vs AR across 2D/3D robot trajectory tasks, ~25x faster 3D point forecasts (preliminary), 70.2% vs 62.6% MolmoSpaces success.
