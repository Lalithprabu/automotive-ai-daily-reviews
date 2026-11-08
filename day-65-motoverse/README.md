# Day 65 - MotoAnticipator (inspired by MotoVerse, arXiv 2611.06007)

Causal ego-rider maneuver anticipation + ego-motion-compensated agent trajectory forecasting for
motorised two-wheelers, with a synthetic real-time overlay simulation.

**Honesty note:** MotoVerse (Paturkar, Gangisetty, Jawahar; Nov 4 2026) is a *benchmark* (949 clips, >31k agents,
>16k MTWs; tasks ERMA + TATP). It does not publish this architecture. Everything here (model, toy generator, numbers)
is my own reference design built from the abstract only. Toy results are synthetic, 1 seed.

```
python -m pytest -q tests          # 5 tests
python train.py --steps 800        # ~30 s CPU; add --no-comp for the ablation
python simulate.py --out assets/demo.mp4 [--show]
```
Structure: models/motoanticipator.py, src/synthetic.py, train.py, simulate.py, tests/test_all.py, assets/, checkpoints/.
