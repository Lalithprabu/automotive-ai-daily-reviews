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
| 16 | [Vehicle Trajectory Prediction via Neural Fusion of Multiple EKF-Based Trajectory Candidates](day-16-ekf-neural-fusion/) | [2609.19813](https://arxiv.org/abs/2609.19813) | Multi-modal trajectory forecasting — late fusion of 4 classical EKF motion models with a neural (Trajectron++-style) predictor via learned per-timestep attention |
| 17 | [RiskWorld: Risk-Aware World Modeling with Flow-Guided Occupancy Evolution for Selective Trajectory Planning](day-17-riskworld/) | [2609.18442](https://arxiv.org/abs/2609.18442) | Trajectory + occupancy forecasting / risk-aware planning — flow-guided occupancy evolution fused with a spatial risk field, gated into a selective trajectory-replacement planner |
| 18 | [ChargeIC-LSTM: Intelligent Degradation Monitoring in Lithium-ion Batteries via Discharge Incremental Capacity Feature Estimation](day18-ic-degradation-lstm/) | [2609.22843](https://arxiv.org/abs/2609.22843) | EV battery degradation monitoring — first battery-category entry; predicts a battery's discharge IC curve directly from the charging-phase signal |
| 19 | [Prediction-Aided V2X Safety Message Recovery via Uncertainty-Aware LDPC Decoding](day19-v2x-ldpc-recovery/) | [2609.25609](https://arxiv.org/abs/2609.25609) | V2X safety-message reliability — propagates probabilistic motion-prediction uncertainty into a second, prior-guided LDPC belief-propagation decode pass |
| 20 | [ForeDrive: Foresight-Guided End-to-End Autonomous Driving with a Planning-Relevant Latent World Model](day20-foredrive/) | [2609.26299](https://arxiv.org/abs/2609.26299) | End-to-end driving / latent world models — JEPA-style multi-horizon forecasting asymmetrically coupled via stop-gradient into a Diffusion Transformer planner |
| 21 | ["Bend the Clock": ChronoFuse — Predicting Ahead to Beat Latency in Event-Based Object Detection](day21-chronofuse/) | [2609.26919](https://arxiv.org/abs/2609.26919) | ADAS vision perception — first event-camera / latency-compensation entry; predicts object state for the moment a detector's own output becomes available |
| 22 | [S2Planner: Multi-Scale Semantic Planner for End-to-End Autonomous Driving](day22-s2planner/) | [2609.29813](https://arxiv.org/abs/2609.29813) | End-to-end driving / vision-foundation-model planning — plans directly from DINOv3 features via geometry-guided camera cross-attention, no world model or risk field |
| 23 | [PriorMapBEVNet: Leveraging Vision-Based Point Cloud Map Priors for Camera-Based 3D Object Detection and Online Vectorized HD Mapping](day-23-priormapbevnet/) | [2609.26325](https://arxiv.org/abs/2609.26325) | ADAS vision perception — first entry using a persistent, cross-traversal vision-built scene memory (no LiDAR) fused with live camera BEV features |
| 24 | [INTERACT: Interactive Planning for Autonomous Driving via Anchor-Conditioned Prediction and Trust-Region Refinement](day-24-interact/) | [2609.31137](https://arxiv.org/abs/2609.31137) | Trajectory forecasting + interactive planning — predicts other agents' reactions once per map-derived intent anchor (not once per candidate trajectory), then refines with trust-region CEM |
| 25 | [ECO: Guiding End-to-End Driving Models with Endpoint-Constrained Trajectory Optimization](day-25-eco/) | [2609.31383](https://arxiv.org/abs/2609.31383) | End-to-end driving / trajectory post-processing — training-free layer that keeps the policy's predicted endpoint, anchors to executed history, and re-solves intermediate waypoints for smooth, trackable paths |
| 26 | [WALT: Learning World-Model-Aligned Latent Trajectories for Autonomous Driving](day-26-walt/) | [2609.30436](https://arxiv.org/abs/2609.30436) | End-to-end driving / latent trajectory planning — a dual-branch trajectory autoencoder borrows knowledge from a frozen driving world model, and the planner predicts in that learned latent space instead of raw waypoints |

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
