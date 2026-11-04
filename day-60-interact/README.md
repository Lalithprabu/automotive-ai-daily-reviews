# Day 60 - INTERACT-style object-centric world model for interaction-aware prediction (synthetic re-implementation, negative result)
**Paper:** *INTERACT: Learning Interaction-Aware World Dynamics from Driving Videos* - arXiv [2611.02526](https://arxiv.org/abs/2611.02526). Abstract-level read only; see `SOURCING.md`.
```
frames -> recurrent Slot Attention (K=7 slots, SAVi-style first-frame init) -> slots S_t
(S_t, S_t+1) -> latent ego action z_t (5 discrete codes, no labels)      } learned from video only
(S_t, z_t) -> slot Transformer -> S_t+1 ; decode slots -> predicted frame  }
planner: score 5 candidate ego actions from S_t + predicted future slots (distilled from a privileged oracle) -> pick the cheapest
```
## Layout
```
day-60-interact/
  interact/synthetic.py   BEV world, IDM traffic, ego-reactive cut-ins, oracle cost
  interact/model.py       SlotAttention, SlotEncoder, SlotDecoder, LatentAction, FuturePredictor, SlotPlanner, DensePlanner
  interact/evaluate.py    grounding, open-loop prediction error, closed-loop driving
  train.py                stages 1-3 + evaluation -> checkpoints/results.json
  simulate.py             'AI in action' GIF/MP4 (slots, predicted vs true paths, costs, telemetry); --mock for an illustrative clip
  tests/test_core.py      7 pytest checks
  assets/                 interact_demo_mock.{gif,mp4}  interact_demo_trained.{gif,mp4}   SIMULATION_PROMPT.md   SOURCING.md
```
Run: `pip install -r requirements.txt && python train.py && python simulate.py && pytest -q` (full train ~20 min on 2 CPU cores; `--quick` ~4 min; `simulate.py --mock` needs no checkpoint).
**Status: pipeline works end to end, but the world model did not converge in my CPU budget; the dense baseline is the best planner here. Read `SOURCING.md` before using any number.**
