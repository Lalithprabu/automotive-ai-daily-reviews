# Day 57 - eCoMAE-style evidential masked pretraining for VRU intent (synthetic re-implementation)
**Paper:** *Observability-Aligned Evidential Masked Pretraining for Reliable VRU Motion Prediction* - arXiv [2610.39958](https://arxiv.org/abs/2610.39958). See `SOURCING.md` for what is paper-sourced vs mine (summary-level read only).

## Idea
```
mask 60% of valid time steps in blocks -> encode ONLY visible tokens -> decoder inserts mask tokens -> per step/feature: recon x^ and NIG (mu, nu, alpha, beta)
sigma^2 = beta(1+nu)/(nu(alpha-1))  = aleatoric beta/(alpha-1) + epistemic beta/(nu(alpha-1))
L = rec(masked) + evi(masked: NIG NLL + 1e-2*|y-mu|(2nu+alpha)) + 0.05*mean-consistency(visible vs no-grad reference, 1/(var_ref+tau) weights)
    + 0.05*relu(2nu+alpha-2)(masked) + 0.10*(1-cos) between pooled overlap-visible embeddings of mask views
then: fine-tune encoder + linear head on few labels (intent: straight/left/right/stop)
```
## Layout
```
day-57-ecomae/
  ecomae/synthetic.py  VRU world, 3 weather envs (clear/rain/fog), dropped steps, distractor glances
  ecomae/model.py      block_mask, eCoMAE (visible-only encoder, NIG decoder), nig_nll, pretrain_loss
  ecomae/metrics.py    accuracy, ECE, AURC
  train.py             pretrain -> fine-tune -> ID vs OOD(fog) for sup / Ti-MAE-style / eCoMAE-style
  simulate.py          'AI in action' GIF/MP4: input vs ground truth vs prediction + uncertainty
  tests/test_core.py   8 pytest checks (NIG constraints, mask validity, encoder never sees hidden steps, ...)
  assets/              ecomae_demo.gif / .mp4 / .json      SIMULATION_PROMPT.md   SOURCING.md
```
Run: `pip install -r requirements.txt && python train.py && python simulate.py && pytest -q` (full train ~30 min on 2 CPU cores; `--quick` ~40 s)

## Results (synthetic, 3 seeds, mean±std)
| method | ID acc | OOD acc | OOD ECE | OOD AURC |
|---|---|---|---|---|
| supervised | 0.995 | 0.888±0.034 | 0.090±0.031 | 0.028±0.017 |
| Ti-MAE-style | 0.994 | 0.902±0.020 | 0.075±0.019 | 0.027±0.012 |
| eCoMAE-style | 0.994 | 0.909±0.027 | 0.074±0.023 | 0.022±0.008 |

Same ordering as the paper, but all gaps are inside seed noise - treat as "not contradicted", not "reproduced".
