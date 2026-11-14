# Day 70 - FlowBatt: flow matching for probabilistic battery-degradation forecasting (toy re-implementation)

Paper: *FlowBatt: Flow Matching for Probabilistic Battery Degradation Prediction*, Eivazi, Sharaawy, Wittek, Hebenbrock,
Ginster, Blömeke. [arXiv 2611.03150](https://arxiv.org/abs/2611.03150) (2026-11-02, NeurIPS 2026 poster).
Official code: https://github.com/HamidrezaEiv/FlowBatt. **This repo is NOT the authors' code**: it is a compact
re-implementation of the core idea on a synthetic cell fleet. See `SOURCING.md` for what is paper fact and what is my choice.

```
early capacity matrix Q(cycle 1..100, V)  --CNN (LeakyReLU)-->  c_enc  ┐
observed SOH prefix (10 points)           --MLP-------------->  c_pre  ├─ c ──adaLN──> DiT ──> v(x_t, t | c)
flow time t                                --sinusoid + MLP--->  c_t    ┘
x_t = (1-t)·noise + t·trajectory,  loss = ||v - (trajectory - noise)||²,  sample: Euler 0 → 1, 100 draws → 5-95 % band
```

## Folder structure
```
day-70-flowbatt/
├── README.md  SOURCING.md  SIMULATION_PROMPT.md  config.yaml  requirements.txt
├── flowbatt/
│   ├── world.py      # synthetic 3-chemistry fleet: capacity matrices + 256-pt SOH trajectories, hidden late defect, random knee
│   ├── data.py       # cached splits (by cell), per-point fade normaliser (train stats only)
│   └── model.py      # CapacityEncoder, PrefixEncoder, DiTBlock (adaLN-Zero), TrajectoryDiT, FlowBatt,
│                     # DiffBattLite (DDPM, same DiT+encoder), DetTransformer, DetMLP
├── train.py          # python train.py --model {flowbatt,diffbatt,transformer,mlp} --seed 0
├── evaluate.py       # trajectory RMSE, RUL MAE/MAPE, 90 % coverage + width, RUL coverage, NFE sweep, latency (+ RF baseline)
├── simulate.py       # OpenCV "AI in action": input -> flow generation -> truth reveal, fleet RUL board, telemetry
├── tests/test_flowbatt.py   # 10 unit tests
├── checkpoints/      # weights, training logs, metrics_s0_a.json / metrics_s0_b.json / metrics_s1.json, run1_n640/ (failed first run)
└── assets/           # flowbatt_simulation.gif / .mp4 / flowbatt_still.png
```

## Quick start
```bash
pip install -r requirements.txt
pytest -q tests                                   # 10 passed
for m in flowbatt diffbatt transformer mlp; do python train.py --model $m; done   # ~8 min each, CPU, 1 thread
python evaluate.py --seed 0
python simulate.py --out assets/flowbatt_simulation.gif --mp4 assets/flowbatt_simulation.mp4   # add --live for a window
```

## Toy results (synthetic test fleet: 320 unseen cells, 3 chemistries, 80 % EOL)
Train 2,400 cells, 2,500 steps, batch 64. Point forecast for generative models = element-wise median of 20 samples. Interval
metrics use 100 samples on the first 96 test cells. Latency = 1 cell × 100 samples, CPU, 1 thread.

| Model | Traj SOH RMSE (%-pts) | RUL MAE (cycles) | RUL MAPE | 90 % band coverage | Band width | RUL in 5-95 % (hidden-defect / typical) | Params | Latency |
|---|---|---|---|---|---|---|---|---|
| **FlowBatt (32 Euler steps)** | **3.70** | 126 | **13.0 %** | 0.816 | 8.2 | 0.79 (0.64 / 0.87) | 1.38 M | **1.53 s** |
| DiffBatt-lite (DDPM, 100 steps, x0-clip) | **3.65** | **125** | 13.3 % | **0.850** | 9.0 | **0.88** (0.85 / 0.89) | 1.38 M | 3.94 s |
| Deterministic Transformer (same trunk, MSE) | 4.55 | 146 | 15.8 % | - | - | - | 1.34 M | - |
| MLP head | 4.58 | 146 | 15.2 % | - | - | - | 0.28 M | - |
| Random forest (Severson-style ΔQ features) | 4.23 | 150 | 14.5 % | - | - | - | 300 trees | - |

Seed 1 (new init and data order): FlowBatt 3.98 RMSE / 0.828 coverage / 0.865 RUL coverage; Transformer 4.50.
FlowBatt NFE sweep (seed 0, 96 cells, RMSE of 20-sample median): 1 step 4.04 · 2 steps 3.80 · 4 steps 3.80 · 8 steps 3.88 · 32 steps 3.98 · 64 steps 4.00 (Heun 8 steps: 3.96).
A **single** sample scores 4.87 RMSE: one draw is a scenario, not a forecast.

### Reading the toy honestly
* **Generative beats deterministic**: 3.70 vs 4.55 (−19 %) against the same-trunk Transformer and −13 % against RF, on both seeds. The deterministic models overfit (val loss 0.30–0.35 vs train 0.02). Here the noise-conditioned objective also acts as a regulariser.
* **FlowBatt did NOT beat DiffBatt here.** Same backbone and encoder: 3.70 vs 3.65 RMSE, a tie, and DDPM is *better* calibrated (85 % vs 82 % coverage). In this toy, flow matching wins on **speed**: 2–4 Euler steps already reach the 32-step accuracy, against 100 DDPM steps. The paper's MIX-20 gap (0.88 vs 1.15) is not reproduced. The paper also notes the objective's share of its gain is not isolated.
* **Coverage below nominal, as in the paper**: 82–83 % vs a nominal 90 %. It is worst on cells with a hidden late defect (64 % of those get their true EOL inside the band), the cases the early data cannot see.
* **Two failures worth knowing about**: (1) with only 640 training cells, FlowBatt **memorised** (val loss U-turn after step 1,000), and its band collapsed to 58 % coverage and 23 % RUL coverage (`checkpoints/run1_n640/`). One trajectory per condition is a memorisation trap for conditional generators, and the paper's benchmarks have only 41–256 training cells. (2) DDPM without x0 clipping produced divergent samples (RMSE 7.7, cell sd 28.9). The DiffBatt row uses the standard clip.
* **Not tested**: real MATR / HUST / MIX data, held-out chemistries, temperature or current conditioning, the paper's DiT size, calibrated intervals (conformal etc.).

## Core code (flowbatt/model.py, excerpt: the full file has DiffBatt-lite and the deterministic baselines)
See `flowbatt/model.py`. It contains the CNN capacity encoder (two channels Q and ΔQ = Q_n − Q_1, LeakyReLU 0.3,
GroupNorm, avg+max pool), adaLN-Zero DiT blocks over 16 patches of 16 SOH points, the linear-path flow-matching loss and
the Euler/Heun sampler. The sampler can return the whole path, which is how the simulation animates noise turning into futures.
