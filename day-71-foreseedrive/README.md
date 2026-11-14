# Day 71: ForeseeDrive, future-token conditioning for anticipatory generative driving (toy PyTorch re-implementation)

Paper: *ForeseeDrive: Future-Token Conditioning for Anticipatory Generative Driving*, Wu et al., arXiv [2611.14286](https://arxiv.org/abs/2611.14286) (submitted 2026-11-10).
**This is not the authors' code.** It is a small, CPU-trainable re-implementation of the core mechanism on a synthetic 2-lane world. [SOURCING.md](SOURCING.md) separates what comes from the paper from what I chose.

```
present scene ─► [frozen] base generator ─► K candidates + logits                     (the old way: score what you got)
      └─► token predictor ─► future tokens (actor | route | occupancy | TTC-margin), each (e, c, r)
                              └─► u = Pool(Attn(tokens, scene))   ──► z̃_k = z_k + α_l·A_l(z_k, u)  at every decoder layer
                                                                  └─► ℓ̃_k = ℓ_k + b_k(u)
      A_l(z, 0) = 0, α_l = 0 at init, b_k(0) = 0  =>  all-zero tokens give EXACTLY the base planner
      + candidate-preservation loss + TTC-margin guard (train hinge + inference reject of safe->unsafe candidates)
```

## Folder structure
```
day-71-foreseedrive/
├── README.md / SOURCING.md / config.yaml / config_k6.yaml / requirements.txt
├── foresee/
│   ├── world.py      # synthetic street with hidden futures (hard-braking lead, pedestrian step-out, cut-in), expert, PDMS-like metric
│   ├── tokens.py     # teacher future tokens from the LOGGED future: actor / route / occupancy queries
│   ├── data.py       # dataset generation + cache
│   ├── model.py      # SceneEncoder, BaseGenerator, TokenPredictor, TokenPool, ZeroGatedAdapter, ForeseeDrive
│   └── losses.py     # WTA imitation, distillation, margin tokens, preservation, TTC guard (train + inference)
├── train.py          # --stage base | predictor | cond --variant full | scorer | naive
├── evaluate.py       # all configs + token diagnostics -> checkpoints/metrics_s*.json
├── simulate.py       # OpenCV "AI in action": BEV input vs ground truth vs base vs ForeseeDrive + tokens + telemetry
├── tests/test_foresee.py   # 9 tests (exact zero-token no-op, guard rule, torch/numpy metric match, ...)
├── checkpoints/      # weights, logs, metrics_s0/s1/k6_s0/s0_lambdap1.json
└── assets/           # foresee_simulation.gif / .mp4 / still.png
```

## Run
```bash
pip install -r requirements.txt
python train.py --stage base && python train.py --stage predictor
for v in full scorer naive; do python train.py --stage cond --variant $v; done
python evaluate.py --seed 0
python simulate.py --scenes 5 --out assets/foresee_simulation.gif --mp4 assets/foresee_simulation.mp4   # add --live for a window
pytest -q tests
```
The whole pipeline takes about 15 minutes on 2 CPU cores.

## Toy results (3,000 test scenes; 1,812 contain a hidden future event; K = 16)
PDMS-like = NC·DAC·(5·TTC + 2·Comf + 5·EP)/12, scored against the TRUE future. The expert scores 0.998.
Each cell shows seed 0 / seed 1. Seed 1 retrains only the conditioning stage; the base and predictor are shared.

| Config | PDMS | Collision % | PDMS, hidden-event scenes | Best-of-K PDMS | Candidates degraded (>0.05) |
|---|---|---|---|---|---|
| Base generator (frozen) | 0.714 | 25.5 | 0.552 | 0.993 | – |
| Scorer-only (logit path), predicted tokens | 0.726 / 0.719 | 24.3 / 25.0 | 0.569 / 0.559 | 0.993 | 0 |
| **ForeseeDrive, predicted tokens** | **0.724 / 0.722** | **24.5 / 24.4** | 0.567 / 0.565 | 0.993 / 0.992 | 1.6 / 3.2 % |
| ForeseeDrive, shuffled tokens | 0.695 / 0.706 | 27.1 / 25.9 | 0.533 / 0.549 | 0.992 | 2.7 / 3.9 % |
| ForeseeDrive, logged-future ("oracle") tokens | 0.900 / 0.876 | 5.7 / 6.0 | 0.854 / 0.825 | 0.987 / 0.984 | 12.4 / 17.2 % |
| Scorer-only, oracle tokens | 0.881 / 0.882 | 8.3 / 8.1 | 0.826 / 0.831 | 0.993 | 0 |
| Naive (no preservation, no guard), predicted | 0.667 / 0.653 | 26.3 / 29.7 | 0.517 / 0.485 | 0.934 / 0.945 | 45.6 / 52.1 % |
| Naive, oracle tokens | 0.735 / 0.713 | 15.5 / 20.9 | 0.671 / 0.623 | 0.911 / 0.927 | 56.8 / 58.0 % |

Zero-token control: max |τ − τ_base| = **0.0** and max |ℓ − ℓ_base| = **0.0**, with the same selection in 100% of scenes.
Token predictor: confidence accuracy 0.958, boundary-risk recall 0.83, descriptor MAE 0.12.
Parameters: base 242 K, predictor 189 K, adapters 77 K (only the adapters are trained in stage 3).
CPU latency (batch 1, 1 thread): base 1.7 ms vs predictor + ForeseeDrive 7.6 ms.
λ_p = 1 instead of 10: predicted 0.714 (no gain), oracle 0.850.
K = 6: base 0.717, predicted 0.715, oracle 0.847, scorer-only oracle 0.872.

### Reading the toy honestly
* **The mechanism works when the tokens are right.** Logged-future tokens cut collisions from 25.5% to 5.7–6.0% and raise PDMS by 16–19 points.
* **The deployable gain is small: +0.8–1.0 PDMS points.** The predictor sees the same present scene as the base model, and hidden events such as a hard brake with no visible cue cannot be forecast from it. This is the paper's own caveat: its diagnostics use logged-future tokens and do not test the predictor.
* **The model does read the tokens.** Aligned beats shuffled by 1.6–2.9 points, and shuffled tokens fall *below* the base. The paper's aligned − shuffled gap is much smaller (+0.0002 PDM), but its diagnostics differ (fixed candidates, headroom splits).
* **Preservation is not optional.** Without it, 46–58% of candidates get worse and PDMS falls below the base, even with oracle tokens. λ_p = 1 was too weak.
* **The support-expansion story does NOT reproduce cleanly here.** The base already covers a good plan in 99% of scenes (best-of-K 0.993), so the logit path alone recovers most of the oracle gain (0.881 vs 0.900; 0.882 vs 0.876 on seed 1). Geometry conditioning still lowers collisions (5.7–6.0% vs 8.1–8.3%). Shrinking to K = 6 did not change this. It matches the paper's "oracle-best quality is mixed".
* **The guard almost never fires** (≤ 0.06% of candidates), because preservation already keeps candidates close to the base. That agrees with the paper calling the guard a local safety repair.
* **Not tested:** diffusion sampling, cameras/LiDAR, NAVSIM/nuScenes/Bench2Drive, or the authors' token definitions and sizes.
