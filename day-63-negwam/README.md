# Day 63 - NegWAM-style negative-aware world-action model (PyTorch, synthetic)

Reconstruction of the idea in **arXiv:2611.08111** (*Reinforcing End-to-End Autonomous Driving Safety via Negative-Aware Planning in a Unified
World-Action Model*, submitted 2026-11-05). Read `assets/SOURCING.md` first: only the abstract-level description was available, so the
architecture here is my own reading of it and all numbers are from a synthetic car-following world, **not NAVSIM**.

```
day-63-negwam/
  negwam/world.py     synthetic IDM car-following world, safety checks, near-expert negative generator
  negwam/model.py     unified world-action transformer + training-only rider tokens (masked attention)
  negwam/losses.py    flow matching + world loss + repulsive-anchor hinge + rider classification
  train.py            baseline vs NegWAM-style training, eval -> results.json
  simulate.py         "AI in action" replay (expert vs baseline vs NegWAM overlay + telemetry) -> GIF/MP4
  tests/test_core.py  6 tests (incl. "riders change no inference output")
  assets/             demo GIF/MP4, SIMULATION_PROMPT.md, SOURCING.md
  checkpoints*/       trained weights + results.json (full-data run, low-data 3-seed run)
```
Run: `pip install -r requirements.txt && python train.py && python simulate.py --ckpt checkpoints && pytest -q`

## Core idea in code
```python
# model.py - who attends to whom (True = blocked). Riders see the scene + themselves; nobody sees riders,
# so removing them at inference changes nothing (zero overhead), yet their loss shapes the shared backbone.
m[sc, sc] = False                                   # scene  -> scene
m[wo, sc] = m[wo, wo] = False                       # world  -> scene + world   (predicts lead positions)
m[ac, sc] = m[ac, wo] = m[ac, ac] = False           # action -> scene + world + action  (unified WAM)
for r in range(R): m[i_r, sc] = m[i_r, i_r] = False # rider  -> scene + self    (training only)

# losses.py - rectified flow + repulsive anchors
x1_hat = xt + (1 - t) * v                           # clean-trajectory estimate
d_pos  = rms(x1_hat - expert);  d_neg = rms(x1_hat - negative)
l_rep  = (t * relu(d_pos - d_neg + margin)).mean()  # stay closer to expert than to any near-expert negative
loss   = l_flow + 0.5*l_world + 1.0*l_rep + 0.3*l_rider
```

## Results (synthetic; read honestly)
| setting | split | model | collision | off-lane | headway viol. | ADE (m) | toy score |
|---|---|---|---|---|---|---|---|
| full data (20k+5k scenes, 3000 steps, 2 seeds) | hard | baseline | 0.05% | 0.0% | 0.25% | 0.162 | 0.9958 |
| | hard | NegWAM-style | 0.0% | 0.0% | 0.20% | 0.166 | 0.9962 |
| low data (1.5k scenes, 1500 steps, 3 seeds) | hard | baseline | 0.27% | 0.0% | 1.10% | 0.266 | 0.9883 |
| | hard | NegWAM-style | 0.20% | 0.0% | 0.77% | 0.255 | 0.9905 |

**Takeaway:** with enough data the baseline already saturates this toy, so there is no measurable gain. In the low-data regime the negative-aware
model had fewer headway violations in all 3 seeds (0.9/0.7/0.7% vs 1.5/1.0/0.8%) and a +0.002 toy score, a small effect on 1,000 scenes
that I would not call significant. Collision counts are 0-5 scenes per run. The paper's +4.1% relative EPDMS on navhard is not reproduced or
contradicted here. `assets/negwam_demo_disagree.*` shows only scenes where the two models' outcomes differ, so it exaggerates the gap; `negwam_demo.*`
shows the first 6 hard scenes, where both models are safe.
