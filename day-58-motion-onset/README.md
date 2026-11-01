# Day 58 - Meta-learned scene-graph GNN for motion-ONSET prediction (synthetic re-implementation)
**Paper:** *Open-World Object Motion Anticipation via Meta-Learned Semantic Adaptation* - arXiv [2610.38993](https://arxiv.org/abs/2610.38993) (warehouse robotics; used here for road-user onset). See `SOURCING.md` for what is paper-sourced vs mine (summary-level read only).

## Idea
```
per frame: nodes = visible objects (+1 persistent scene node); edge = kNN(k<=5, <=10 m) + scene<->object
node_in = Linear[pos, vel, valid] + Linear(semantic emb) + Linear(h_{t-1})
2 x GATv2(4 heads) -> GRUCell(h_{t-1}) -> MLP -> sigmoid = P(static object moves within 3 s)
loss = BCE over query set (static, visible, velocity observed); truncated BPTT; no weight update at test time (adaptation = GRU memory)
```
## Layout
```
day-58-motion-onset/
  motion_onset/synthetic.py  kerbside world, hidden per-episode class mobility + tempo + queue-release cue, novel classes
  motion_onset/model.py      dense GATv2, MotionOnsetGNN (graph build, rollout w/ TBPTT), masked_bce
  motion_onset/metrics.py    Brier skill score, PR-AUC (AP), ECE (equal-mass bins)
  train.py                   full vs stateless vs class-frequency, 3 seeds, val + novel-class sets
  simulate.py                'AI in action' GIF/MP4 (input vs prediction halos vs ground truth + ego advisory)
  tests/test_core.py         8 pytest checks (causality, permutation equivariance, invisible-node isolation, memory on/off, ...)
  assets/                    onset_demo.gif/.mp4/.json   SIMULATION_PROMPT.md      SOURCING.md
```
Run: `pip install -r requirements.txt && python train.py && python simulate.py && pytest -q` (full train ~40 min on 2 CPU cores; `--quick` ~1 min)

## Results (synthetic, 3 seeds)
| method | val BSS | val PR-AUC | novel-class BSS | novel PR-AUC |
|---|---|---|---|---|
| class-frequency | 0.081 | 0.236 | 0.000 | 0.157 |
| stateless GNN | 0.086±0.005 | 0.250±0.005 | 0.012±0.009 | 0.189±0.009 |
| full recurrent GNN | 0.104±0.005 | 0.286±0.003 | 0.039±0.007 | 0.228±0.009 |

Same ordering as the paper; absolute skill is low because onset is a stochastic hazard here. Paper's drastic no-recurrence collapse (BSS -2.1) NOT reproduced. Details in `SOURCING.md`.
