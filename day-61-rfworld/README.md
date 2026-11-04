# Day 61 - RFWorld-style rollout-free trajectory evaluation (toy re-implementation)
**Paper:** *Trajectory Evaluation via Rollout-Free World Model for End-to-end Autonomous Driving* - arXiv [2611.05072](https://arxiv.org/abs/2611.05072); official code [Rupertail/RFWorld](https://github.com/Rupertail/RFWorld). I could read only the repo README (arXiv was rate-limited): see `assets/SOURCING.md` before quoting any number.
```
camera + LiDAR + ego + nav -> shared scene memory (BEV map + 16 query tokens)   [temporal aux loss, training only]
256 candidate trajectories -> sparse readout (grid_sample at each waypoint + cross-attention) -> sub-scores -> PDMS
pick argmax PDMS.  No world-model rollout at inference.
```
## Layout
```
day-61-rfworld/
  rfworld/world.py     toy road + moving vehicles, 256 fixed anchors, analytic PDMS-style scorer, closed-loop step
  rfworld/model.py     SceneMemory, SparseReadout, TemporalHead, RFWorldToy, GlobalAnchorClassifier (baseline), loss
  rfworld/evaluate.py  open-loop selection quality + closed-loop driving
  train.py             trains both scorers (~6 min, 2 CPU cores; --quick ~1.5 min) -> checkpoints/
  simulate.py          real-time-style "AI in action" GIF/MP4: input BEV, anchors coloured by prediction, truth, telemetry
  tests/test_core.py   8 pytest checks
  checkpoints/         trained weights + results.json + train.log
  assets/              rfworld_demo.{gif,mp4,json}  SIMULATION_PROMPT.md  SOURCING.md
```
Run: `pip install -r requirements.txt && python train.py && python simulate.py && pytest -q`

## Toy results (NOT the paper's) - see assets/SOURCING.md
Picked-anchor PDMS 0.882 (sparse readout) vs 0.801 (global MLP) vs oracle 0.931; temporal aux ablation: no effect; closed-loop off-road worse for the sparse scorer (43% vs 21%).
