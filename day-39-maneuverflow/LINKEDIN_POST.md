## POST BODY (paste this)

🚀 **[Day 39] Deep Dive: ManeuverFlow — One-Step Intent-Diverse Planning With Future Evidence**

🎬 **AI in action:** a car approaches a lane with a cut-in vehicle and a lead that may brake. One forward pass fans out candidate paths per intent. The old multi-query planner (red) collides. ManeuverFlow (blue) picks a safe lane change close to the true path (green).

**What is it?**
ManeuverFlow (Chongqing University + Geely, arXiv 2610.13409) is an end-to-end planner that generates trajectories for distinct driving intentions in one step, then scores and refines them using current and predicted future scene evidence. The paper reports 90.6 PDMS on NAVSIM v1 and 90.1 EPDMS on NAVSIM v2.

🏗️ **The Architecture Shift**
* **The Blueprint:** A TransFuser-style backbone feeds a maneuver selector. Gaussian-mixture noise goes through a maneuver-routed mixture-of-experts MeanFlow field to give intention-specific paths.
* **Core Innovation:** Each candidate gathers "trajectory-specific evidence" from current features and world-model-predicted future features along its own path. That evidence predicts PDM factors and drives Frenet refinement; refined and retained paths are re-scored together.

🔄 **The Evolution: Multi-Query / Anchor Planners vs. ManeuverFlow**
* **Old Way:** Multi-query regression tends to collapse to one mode. Anchor/generative planners trade coverage against cost: more candidates, more scoring.
* **New Way:** Intent routing spends candidates on relevant maneuvers; one-step MeanFlow keeps cost flat.

📈 **Key Improvements**
* 🎯 Paper: 90.6 PDMS (NAVSIM v1), 90.1 EPDMS (NAVSIM v2), top PDMS* on WorldEngine 3DGS closed loop.
* ⚡ My reconstruction (synthetic, NOT the paper's data): no-collision rate 50.5% → 83.7%, synthetic PDMS 0.31 → 0.43. Future evidence added +0.02 PDMS over current-only.

⚠️ **Current Limitations**
* Paper full text was blocked; the code is my abstract-level reconstruction.
* In my run the old regressor fits the human path better (minADE@6 1.57 m vs 2.49 m). The gain is safety, not imitation accuracy.
* Raw one-step samples are jagged: 19% of picks left the road before refinement.
* Evidence needs a world model; mine beats constant velocity by only ~10%.
* Single seed, small synthetic world.

💡 **Visualizing the Concept**
Architecture chart: scene features split into a maneuver router and a world-model branch. The router picks top-3 intents, each with a Gaussian blob that one MeanFlow step turns into a path fan. Waypoints query current and predicted-future tokens to produce score bars, then a Frenet refiner nudges paths. The old pipeline sits beside it.
*Check out the full code and simulation script in my GitHub repository linked below!*

🔗 Complete Code & Architecture Breakdown here: [INSERT GITHUB LINK]

#AutomotiveAI #MachineLearning #ADAS #TrajectoryPrediction #DeepLearning #AutonomousVehicles #TechReview

---

## SIMULATION SCRIPT / PROMPT (for a video/rendering AI — not part of the post body)

**Title:** "Many intents, one step: choosing a safe path with future evidence"  **Length:** 18 s, 16:9. Top 45%: wide top-down BEV of a 3-lane road (light grey, dashed lane lines). Bottom: two small telemetry charts.

**Scene:** A black ego box at the left drives right at ~10 m/s. Grey ghost boxes mark other cars' current positions: a slow lead ahead in the ego lane, a left-lane car beside it with a gold "signal" tag, a right-lane car. A grey dotted trail shows the ego's last 4 s.

**Phase 1 — Observe (0–3 s):** The history trail draws in. Agent ghosts fade in. Gold labels flag cars with turn signals or brake lights.

**Phase 2 — One-step generate and score (3–8 s):** In one flash, a fan of 8 translucent blue paths blooms from the ego. Their opacity equals their score. Dashed gold lines show the world model's forecast for each car, ending in a gold ×. Bottom-left: horizontal bars rank the candidates, labelled KEEP, LEFT, RIGHT, BRAKE, FOLLOW. The top bar turns solid blue. The red old-way path (dashed) shows the single-pass pick, straight into the lead's lane.

**Phase 3 — Play-out of the true future (8–18 s):** Dark real cars start moving along their true futures: the left car cuts into the ego lane; the lead brakes. Green (truth), red (old) and blue (ManeuverFlow) paths draw as the ego advances, each with an outlined car box. At ~3.5 s the red box overlaps a dark car and flashes. Bottom-right: three clearance curves over time. The red curve dips below the 1.0 contact line; blue and green stay above. Final tags: "PDMS old 0.00 | new 0.72".

**Style:** flat, technical, thin lines, no photoreal textures. Legend: grey = observed, green = truth, red = old multi-query, blue = ManeuverFlow, gold = world-model forecast. Caption: "Synthetic data — illustrative reconstruction".
