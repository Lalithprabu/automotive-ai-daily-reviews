# Automotive AI Daily Tech Reviews

Daily technical breakdowns of the latest machine learning research in the automotive space — ADAS perception, trajectory/motion forecasting, EV battery analytics, V2X — each paired with a LinkedIn deep-dive post and a clean, tested PyTorch reference implementation of the paper's core architecture, complete with an animated simulation GIF.

| Day | Paper | arXiv | Category |
|---|---|---|---|
| 1 | [SV-WAM: An Efficient Surround-View World-Action Model for End-to-End Autonomous Driving](day-01-sv-wam/) | [2609.03602](https://arxiv.org/abs/2609.03602) | End-to-end driving / World-Action Models |
| 2 | [Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving](day-02-qwen-drive-1-0/) | [2609.00111](https://arxiv.org/abs/2609.00111) | VLM unifying ADAS perception + planning |
| 3 | [If It Moves, Radar Knows: A Physics-Aware Radar Transformer (PART)](day-03-part/) | [2609.02289](https://arxiv.org/abs/2609.02289) | ADAS radar perception |
| 4 | [One Diffusion Model, Two Roles: SSDS + DAPSE](day-04-ssds-dapse/) | [2609.04921](https://arxiv.org/abs/2609.04921) | Trajectory planning + safety-critical scenario generation |
| 5 | [DriveZero: End-to-End Driving Beyond Human Demonstrations](day-05-drivezero/) | [2609.06055](https://arxiv.org/abs/2609.06055) | Closed-loop RL / teacher-student distillation |
| 6 | [COSTER: Collision Snapshot Guided Time-Reversed Scenario Generation](day-06-coster/) | [2609.06433](https://arxiv.org/abs/2609.06433) | Safety-critical scenario generation (CVAE) |
| 7 | [MC-DeTra: Motion-Consistent Joint Detection + Trajectory Forecasting](day-07-mc-detra/) | [2609.11717](https://arxiv.org/abs/2609.11717) | Joint BEV perception + forecasting |
| 8 | [Marigold: Diffusion Models as Monocular Depth Estimators](day-08-marigold/) | [2312.02145](https://arxiv.org/abs/2312.02145) | Monocular depth estimation (general CV, evaluated for automotive fit) |
| 9 | [TrajFusionNet+: Transformer-Based Prediction of Pedestrian Crossing Intention via Fusion of Trajectory Representations and Scene Graphs](day-09-trajfusionnet-plus/) | [2609.10806](https://arxiv.org/abs/2609.10806) | ADAS pedestrian crossing-intention prediction — adds a Graph Attention Module onto TrajFusionNet |

Each `day-0N-*/` folder is self-contained and independently runnable:

```bash
cd day-0N-<name>
pip install -r requirements.txt
pytest tests/ -v                                    # shape + backward-pass tests
python train.py --steps 20                           # synthetic-data smoke-test training loop
python simulate.py --config tiny_*_cfg.yaml --out trajectory_simulation.gif   # renders the simulation GIF
```

Every day's README has its own "Architecture" (with a rendered diagram), "Old vs. New" framing, "Simulation" (with the animated GIF), "Limitations", and "Sourcing / reconstruction note" sections.

## Code provenance

Days 1–4's code was carried forward from each day's own original daily-review session and its own repo package. Days 5–7 (DriveZero, COSTER, MC-DeTra) were **freshly reconstructed** when this package was assembled, because each scheduled daily run executes in its own isolated container and the original sessions' files were not retrievable afterward — only the papers' architecture descriptions (and, for Days 6–7, one verbatim code excerpt each) survived in the shared project log. Every day's code and every `simulate.py` — reconstructed or original — was independently re-verified when this package was assembled: every `pytest` suite passes, every `train.py` runs to completion, and every `simulate.py` renders a real GIF. See each day's own README for its specific sourcing/reconstruction note and what is and isn't independently verified from the source paper itself.

This is an independent, best-effort reference implementation series for educational purposes — it is **not** any of the papers' authors' official code release.
