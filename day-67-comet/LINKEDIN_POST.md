🚀 **[Day 67] Deep Dive: CoMET — Fixing a Roadside Camera's Pose Error with a Kalman Filter, Not a Neural Net**

🎬 **AI in action (watch the clip):** a top-down view of a junction. Left: the car's own detections (blue) next to the roadside unit's boxes (orange), all shifted by the same hidden pose error. Middle: ground truth. Right: the same fused scene before (red = false alarms) and after CoMET (green = hits), with a purple arrow showing the offset it is tracking live.

**What is it?**
CoMET (arXiv 2611.09121) tracks the one translation error that shifts *every* box in a roadside V2X message, then subtracts it, even from boxes the ego car can't see. The estimator has zero learned parameters.

🏗️ **The Architecture Shift**
* **The Blueprint:** match ego and roadside boxes of objects both can see (mutual nearest neighbours), turn each pair into an offset measurement, feed a per-sender Kalman filter, correct all roadside boxes.
* **Core Innovation:** treat pose error as *common-mode*, one state per sender, tracked over time. No ground truth needed at inference; if no pair matches, the filter carries the estimate forward.

🔄 **The Evolution: Online-pose late fusion vs. CoMET**
* **Old Way:** trust the roadside message's own pose and fuse. A 1-2 m shared shift silently misaligns every box.
* **New Way:** estimate that shift from the overlap, smooth it in time, apply it everywhere, including roadside-only objects.

📈 **Key Improvements**
* ⚡ Paper (V2X-Seq, 21 val sequences, delays 0-400 ms): 66.63 BEV AP@0.7 vs 63.18 (uncorrected online poses) and 62.96 (best intermediate fusion).
* 🎯 Across 5 detectors: +2.2 to +3.5 AP over online poses, +0.3 to +0.6 AP over using only the current message.
* 🧪 Audit finding: 28% of benchmark offset labels are zero placeholders; correcting labels also helps intermediate fusion.
* 🔬 My toy re-implementation (synthetic, 20 seeds x 4 delays): F1@1 m 0.34 → 0.88, vs 0.68 current-message-only (oracle 0.93).

⚠️ **Current Limitations**
* 38% of messages have no matched pair even at zero delay, so it often runs on carried-forward estimates.
* Translation only; no yaw or scale error.
* Validation sequences only; retraining gains were significant on 2 of 3 architectures.
* In my toy, a scene with almost no ego/roadside overlap gets no correction at all.

💡 **Visualizing the Concept**
Diagram: orange roadside boxes all shifted by one arrow. Matched pairs feed a Kalman block. A "carry forward" loop covers unmatched messages. The corrected boxes include roadside-only objects.
*Check the full simulation and PyTorch code in my GitHub repo below!*

🔗 Complete Code & Architecture Breakdown here: [INSERT GITHUB LINK]

#AutomotiveAI #V2X #CooperativePerception #KalmanFilter #MachineLearning #AutonomousVehicles #TechReview


---

## Simulation Script / Prompt (for an image/video AI or rendering engine)

**Title card (2 s):** "Day 67 - CoMET: tracking the shared pose error of roadside V2X messages. SYNTHETIC demo."

**Layout (16:9, dark UI, 1280x720):** six panels. Top: INPUT | GROUND TRUTH | UNCORRECTED FUSION. Bottom: COMET FUSION | OFFSET TELEMETRY | F1 TELEMETRY.

**Scene:** top-down ego-centred view (72 m x 52 m). Ego car (blue square) at the origin with a dashed 15 m blue range ring; roadside unit (orange triangle) at (22, 6) with a dashed 26 m orange ring. 10 vehicles drive curved paths at 3-9 m/s, 10 steps/s for 12 s. Message age 200 ms.

**Panels:**
1. INPUT - blue rectangles = ego detections, orange rectangles = roadside boxes, every one shifted by the same hidden error (about 1-3 m).
2. GROUND TRUTH - green rectangles with 1 s trails; orange dots on objects only the roadside sees.
3. UNCORRECTED - fused boxes coloured green (hit within 1 m) or red (false alarm); green '+' = true centres. Mostly red.
4. COMET - same colouring; purple arrow from the RSU showing the estimated offset; text "est. offset (x, y) m, pairs=n". Mostly green.
5. OFFSET TELEMETRY - x-component over time: red = true error, orange = current-message-only estimate (jittery), purple = CoMET (smooth). Grey bands where no pair matched.
6. F1 TELEMETRY - cumulative F1@1 m for uncorrected / current-message-only / CoMET.

**Beats:** (0-3 s) panels 3 and 4 both red as the filter initialises; (3-8 s) panel 4 turns green while panel 3 stays red; (8-12 s) grey carry-forward bands pass while the purple line holds. End card with GitHub link. Always show: "SYNTHETIC toy world - not V2X-Seq, not the authors' code".

**Honesty note for the voice-over:** toy F1@1 m averaged over 20 seeds and 4 delays is 0.34 uncorrected, 0.68 current-message-only, 0.88 CoMET. The paper's numbers (66.63 AP) are on V2X-Seq and are not reproduced here.

**Run it yourself:** `python simulate.py --seed 5000 --delay 200 --out assets/comet_simulation.gif --mp4 assets/comet_simulation.mp4`
