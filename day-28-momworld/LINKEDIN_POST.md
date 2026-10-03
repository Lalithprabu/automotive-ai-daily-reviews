🎬 **AI in action:** one road, one lead vehicle ahead, same 4 seconds of history fed to every planner. 🔴 Red = "stale momentum" (keeps extrapolating the old acceleration trend). 🟡 Amber = single-latent-state model. 🔵 Blue = momentum-aware latent rollout. 🟢 Dotted green = expert. Watch red drive to 1.6 m from the lead car while the learned plans stay ~9 m clear.

🚀 **[Day 28] Deep Dive: MomWorld — Momentum-Aware Latent World Model (arXiv:2609.33737)**

**What is it?**
A latent world model for long-horizon driving plans that carries a "momentum" state alongside the scene state, so recent motion trends survive a 6-second rollout. A scene-adaptive reset gate drops that momentum when the scene changes abruptly. Evaluated on NAVSIM, nuScenes and Bench2Drive.

🏗️ **The Architecture Shift**
* **The Blueprint:** history encoder → momentum extractor → latent rollout predicting configuration AND momentum → base trajectory → MoFlow refiner.
* **Core Innovation:** MoFlow, a momentum-conditioned flow-matching module that refines the base trajectory in a few integration steps, with horizon-aware residual fusion (near-term stability vs. long-range correction).

🔄 **The Evolution: Single-latent rollout vs. MomWorld**
* **Old Way:** one latent state rolled forward loses motion information, which destabilises near-term decisions.
* **New Way:** momentum persists across the horizon; the reset gate suppresses it when it's outdated (a lead car brakes, a scene flips).

📈 **Paper claims**
* 🎯 12.2% lower average collision rate vs. MomAD over a 6 s horizon.
* ⚡ Refinement in only a few flow steps. I could not access the full results table, so I quote nothing else.

🧪 **My toy reproduction (synthetic, NOT paper numbers; 3 seeds):**
* Non-learned "stale momentum" extrapolation: 83% collisions in event scenes, ADE 4.7 m there.
* Single-latent baseline ADE 0.42 m vs. my MomWorld-style 0.46 m: **the paper's gain did NOT reproduce**; momentum did not beat the baseline here.
* My reset gate barely discriminates (keep 0.50 event vs 0.47 free). MoFlow was the one piece that helped (0.50 → 0.46 m).
* Bug caught: my fusion weights were untrained, which faked a 0.51 vs 1.78 m "win" until I fixed it.

⚠️ **Limitations**
* Toy 2D world, 113K params; a GRU already encodes the trend, so the single-latent baseline isn't handicapped.
* I only had abstract-level detail; equations and sizes are my own reconstruction.
* Collision rate saturates at ~0% for every learned model, so it can't separate them.

🎞️ **Simulation Script/Prompt**
"Dark top-down BEV, 3 panels. Left: grey history trail, green dotted expert path, a purple lead vehicle moving ahead; red, amber and blue paths drawn waypoint by waypoint every 0.5 s over 6 s, each with a car outline. Red path ignores the lead car and a red COLLISION banner fires when centres are under 6 m. Middle: live distance-to-lead lines with a dashed 6 m threshold. Right: running ADE per planner plus a small chart of reset-gate keep and momentum magnitude. Cut to a second 'free road' scene where all paths overlap."

💡 **Visualizing the Concept**
"Diagram: history → GRU → two branches (momentum extractor, scene context) → rollout cell with a gate icon between m_{k-1} and m_k → base waypoints → flow-matching refiner (4 small arrows) → fused plan. Below it, a faded single-latent rollout for contrast."

🔗 Complete Code & Architecture Breakdown here: [INSERT GITHUB LINK]

#AutomotiveAI #MachineLearning #ADAS #EndToEndDriving #WorldModels #AutonomousVehicles #TechReview
