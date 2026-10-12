# LinkedIn Post — Day 38

## POST BODY (paste this)

🚀 **[Day 38] Deep Dive: WZDrive — Predicting Where a Driver Goes Through a Road Work Zone**

🎬 **AI in action:** watch a car approach a cone-lined closure. Gaze and steering telemetry replay first, then two predictors fire: a conventional kinematics+scene model (red) and WZDrive (blue), against the true path (green).

**What is it?**
WZDrive is a multimodal ego-trajectory predictor for urban work zones, built with a new human-in-the-loop dataset of 1,183 driving segments from 33 drivers. The dataset records driver gaze and control signals alongside the vehicle's path, so a predictor can use more than motion history.

🏗️ **The Architecture Shift**
* **The Blueprint (my reconstruction; the paper's exact design wasn't retrievable):** Ego history, driver controls, gaze and scene objects (cones, workers) are fused by a transformer into K=6 trajectory hypotheses with confidences.
* **Core Innovation:** Driver-state signals (gaze, control) join motion and scene as inputs. The premise: intent shows up in the eyes before the wheel turns. A gaze-conditioned attention layer boosts the objects the driver is looking at.

🔄 **The Evolution: Kinematics-only Predictors vs. WZDrive**
* **Old Way:** Predictors trained on highway-style data see only past motion and map geometry. In a work zone, a driver about to merge looks identical to one about to continue.
* **New Way:** Driver gaze and control can disambiguate intent before the vehicle moves.

📈 **Key Improvements**
* 🎯 Paper: minADE@6 1.09 m, minFDE@6 2.12 m on WZDrive-Data, 7.18% / 13.18% lower than the best baseline per metric.
* ⚡ My PyTorch reconstruction (synthetic data, NOT the paper's): top-1 ADE 0.85 → 0.43 m (−49.6%), driven by late-merge scenes (2.02 → 0.38 m).

⚠️ **Current Limitations**
* The data comes from a driving simulator, so real-world gaze noise, glare and sensor latency are untested.
* Gaze needs an in-cabin eye tracker, which is rare in production fleets.
* In my reconstruction, minADE@6 did not improve (−7.8%): six modes already cover a four-intent world. The gain shows up in picking the right mode, not covering it.
* Paper full text was rate-limited, so the model code is my own reconstruction.

💡 **Visualizing the Concept**
Architecture chart: four input streams (ego kinematics, driver controls, gaze, scene objects) enter separate encoders. A gaze beam highlights the cone or worker the driver is looking at and boosts its attention weight. Everything merges into a transformer decoder with six mode queries, which output six fan-shaped paths with confidence bars.
*Check out the full code and simulation script in my GitHub repository linked below!*

🔗 Complete Code & Architecture Breakdown here: [INSERT GITHUB LINK]

#AutomotiveAI #MachineLearning #ADAS #TrajectoryPrediction #DeepLearning #AutonomousVehicles #TechReview

---

## SIMULATION SCRIPT / PROMPT (for a video/rendering AI — not part of the post body)

**Title:** "Reading the driver's eyes: work-zone trajectory prediction"  **Length:** 16 s, 16:9, light-grey top-down BEV on the left (2/3 of frame), telemetry on the right (1/3).

**Scene:** two-lane road, 50 m visible ahead. Orange cones fill the ego lane starting 35 m ahead; the left lane is open. A purple worker circle stands at the roadside. A small black ego square drives from the left edge at ~10 m/s.

**Phase 1 — Observe (0–6 s):** The ego's last 2 s of path draws as a dark grey dotted line. A translucent gold gaze wedge (12° wide) sweeps from the ego. It starts straight ahead, then swings toward the open left lane. Cones and workers in the wedge glow brighter. Right-top: live lines for gaze yaw (gold), steering (black), brake (red), with a moving "now" cursor.

**Phase 2 — Predict (6–7 s):** Freeze. Six faint blue fans bloom from the ego, one bold. In parallel, six faint red fans for the baseline, one bold dashed. Confidence labels appear at the path ends.

**Phase 3 — Play-out (7–16 s):** A green line draws the true future and the ego slides along it. Right-bottom: two growing error curves (red baseline vs blue WZDrive) with live metre readouts. The baseline's dashed path stays in the lane while the car merges; the blue path curves into the open lane.

**Style:** flat technical look, thin lines, no photoreal textures. Legend: dark grey = observed, green = truth, red = baseline, blue = WZDrive. Caption: "Synthetic data — illustrative reconstruction".
