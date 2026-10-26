# Day 52 — SafeDriveVLA-Mini (conflict-aware instruction following via world-model "dreaming")
Reconstruction of the decision structure of arXiv:2610.20609 on a synthetic 3-lane toy. See SOURCING.md for exactly what is and isn't from the paper.

```
day-52-safedrivevla/
├── config.yaml
├── train.py            # trains dream model + no-dream ablation -> assets/results.json
├── simulate.py         # GIF or --live OpenCV overlay: input vs ground truth vs dream
├── src/{synthetic,safedrivevla,metrics}.py
├── tests/test_model.py # 6 tests
├── assets/             # checkpoint, results.json, safedrivevla_simulation.gif
├── SOURCING.md  SIMULATION_PROMPT.md  requirements.txt
```
Run: `pip install -r requirements.txt && python train.py && python simulate.py && pytest -q`

Synthetic results (val n=1500): mode acc 0.991 (dream) vs 0.986 (no dream); unsafe-instruction recall 0.993; false refusal 0.9%;
collision rate when instruction is unsafe: model 23.6% vs oracle 23.1% vs always-obey 100% (the 23% floor is toy ego-lane traffic; no braking modelled).
The consequence rollout did NOT give a meaningful accuracy gain here — the toy is too easy for a transformer to read off directly.
