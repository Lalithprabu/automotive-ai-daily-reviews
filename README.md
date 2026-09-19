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
| 10 | [READ: Learning Risk-Informed Fields for End-to-End Autonomous Driving](day-10-read/) | [2609.12371](https://arxiv.org/abs/2609.12371) | ADAS / end-to-end driving — continuous, differentiable, coordinate-conditioned risk field replacing fixed-shape safety bubbles |
| 11 | [DRiF: Data-Driven Risk Fields for Safer End-to-End Autonomous Driving](day-11-drif/) | [2609.10377](https://arxiv.org/abs/2609.10377) | ADAS / end-to-end driving — dense BEV risk field trained by pairwise risk-ranking, sharing one BEV feature with map segmentation + planning; verified Bench2Drive gains |
| 12 | [BeamTransFuser: Robust Beam Prediction for V2X Networks with Multi-Modal Sensing](day-12-beamtransfuser/) | [2609.10200](https://arxiv.org/abs/2609.10200) | V2X beam prediction — hierarchical Transformer fusing camera/LiDAR/radar/GPS with a generative modality-imputer for dropped sensors |
| 13 | [LDE: Learning from Distributed Eyes: Leveraging Collaborative Perception for Automated Model Adaptation](day-13-lde/) | [2609.18511](https://arxiv.org/abs/2609.18511) | ADAS perception + V2X — unsupervised domain adaptation using a collaborator vehicle's perception as pseudo-labels over a bandwidth-gated link |
| 14 | [Sparse-BEVNet: Bi-Level Routing and Sparse Spatial Attention based Multi-View BEV 3D Object Detection](day-14-sparse-bevnet/) | [2609.14185](https://arxiv.org/abs/2609.14185) | ADAS vision perception — efficient camera-only BEV 3D detection via routed/sparse attention instead of dense BEVFormer-style sampling |
| 15 | [MM-Future: Multi-Mode Joint World-Action Modeling for Autonomous Driving](day-15-mm-future/) | [2609.20377](https://arxiv.org/abs/2609.20377) | Joint action+scene generative planning — bidirectional flow-matching Transformer, K-means action prior, future-conditioned proposal scorer |

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

Days 1–4's code was carried forward from each day's own original daily-review session and its own repo package. Days 5–7 (DriveZero, COSTER, MC-DeTra) and Days 10–14 (READ, DRiF, BeamTransFuser, LDE, Sparse-BEVNet) were **freshly reconstructed** when each package was assembled, because each scheduled daily run executes in its own isolated container and the original sessions' files were not retrievable afterward — only the papers' architecture descriptions (and, for some days, verbatim code excerpts) survived in the shared project log. Day 15 (MM-Future) was built and pushed directly in the same live session that has GitHub write access, the first day not routed through a zip hand-off. Every day's code and every `simulate.py` — reconstructed or original — was independently re-verified when its package was assembled: every `pytest` suite passes, every `train.py` runs to completion, and every `simulate.py` renders a real GIF. See each day's own README for its specific sourcing/reconstruction note and what is and isn't independently verified from the source paper itself.

This is an independent, best-effort reference implementation series for educational purposes — it is **not** any of the papers' authors' official code release.
