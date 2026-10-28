# Day 53 - CWT-RUL-Mini: early-cycle battery life from compressed wavelet scalograms
Reconstruction of the *idea* in arXiv:2610.34159 (see SOURCING.md - abstract only). Synthetic data; not the authors' code/numbers.

## Structure
config.yaml, train.py, simulate.py, src/{cwt,model,synthetic,metrics}.py, tests/test_all.py, assets/cwt_rul_simulation.gif, SOURCING.md, SIMULATION_PROMPT.md

## Run
pip install -r requirements.txt && pytest -q && python train.py && python simulate.py [--live]

## Synthetic results (held-out cells, 48,399 params, MAPE of cycle life)
| cycles observed | 1 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|
| MAPE % | 55.5 | 48.0 | 40.5 | 30.1 | 26.1 |
Mean-life baseline: 58.0%. Gains are real but modest; the toy leaks life info via a cell-quality latent plus drift.
Fix during build: BatchNorm made eval unstable across window sizes (MAPE 15-480%) -> GroupNorm.

## Core code
See src/cwt.py (MorletCWT, ScaleCompression) and src/model.py (CWTRULNet).
