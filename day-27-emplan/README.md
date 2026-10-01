# Day 27 — EMPlan: sparse anchors + offsets + reward-guided unpaired preference tuning
Reconstruction (see SOURCING.md) of arXiv:2609.38862.
```
day-27-emplan/
├── config.yaml  train.py  simulate.py  multiseed.py
├── emplan/{model,reward,data}.py
├── tests/test_emplan.py      # 8 tests
└── assets/emplan_simulation.gif
```
Run: `pip install torch pyyaml matplotlib pillow pytest && pytest -q && python train.py && python simulate.py`

Core code: `emplan/model.py` (`EMPlan`: encoder → K anchor logits + K×T×2 offsets) and `train.py::stage2` (unpaired preference loss
`-log σ(β·y_k·(log π_θ(k) − log π_ref(k)))`, y_k∈{+1,−1} from rule reward, collisions always −1).

| (synthetic, 3 seeds) | collision | reward |
|---|---|---|
| single-mode regression | 53.9% | −2.88 |
| EMPlan stage 1 | 6.3% | +1.98 |
| EMPlan stage 2 | 2.5% | +2.36 |
