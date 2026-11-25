🚀 **[Day 80] Deep Dive: GGDriver, Planning by Letting Each Trajectory Choose What It Looks At**

🎬 **AI in action (watch the clip):** two cameras, same road. Left: 8 candidate paths scored on the same 9 summary tokens. Right: each candidate "gazes" at the camera patches along its own path (cyan cells). Both then drive their pick for 4 s: a random scene, a selected win, a selected loss.

**What is it?**
arXiv 2611.27330 (Sun et al., 23 Nov) targets a habit of end-to-end planners: scoring every candidate trajectory against one compressed scene summary. GGDriver lets each candidate pick its own visual evidence, using only the path and camera calibration.

🏗️ **The Architecture Shift**
* **The Blueprint:** a global glance proposes hypotheses from shared features. A corridor around each path is projected into the cameras, and the mask selects patch features at native resolution.
* **Core Innovation:** after refinement, the scorer re-projects the final path and ranks it on evidence along that path.

🔄 **The Evolution: Shared-Summary Scoring vs. Per-Path Gaze**
* **Old Way:** one compressed scene (registers, BEV grid, coarse tokens) for every candidate, so a small cone in one lane gets blurred away.
* **New Way:** a candidate only reads the patches it would drive through. Paper: 94.2 PDMS on NAVSIM-v1 and 72.18 Driving Score on Bench2Drive, ViT-S, no perception labels.

📈 **Key Improvements (my synthetic rebuild: 1,500 held-out scenes, 1.0M params)**
* 🎯 Rule score 0.87 vs 0.56 for glance-only. Collisions fell from 27.5% to 2.2%.
* 🔍 The scorer reads 17% of patches. It beats a random mask of the same size (0.87 vs 0.65) and all 240 patches (+0.027, CI +0.019..+0.036).
* ⚖️ Gaze for ranking beat gaze for refinement only, as the paper says, but by just +0.010.

⚠️ **Current Limitations**
* Most of the jump is "give the planner native patches at all": all-patch attention already reaches 0.84. The path mask adds the last ~3%.
* Confound: the gaze model's own glance head (shared tokens only) scores 0.85. The summary CAN hold the cone; the glance-only model never learned to.
* The gaze planner still crashed in 2.2% of scenes. In 14 of them the old planner was safe (clip, scene 3).
* Static scenes, one camera, a hand-written score. With calibration drift, a mis-projected corridor would hide the hazard.
* Abstract-only rebuild, 2k CPU steps, 1 seed (glance: 2). Not NAVSIM.

💡 **Visualizing the Concept**
Diagram: camera → 240 patches → glance (registers → top-8 paths) → per-path 3-D corridor projected into the image → masked refine → re-project → masked scorer → pick. Above it, the old way: all paths scored on the same 9 tokens.
*Simulation and PyTorch code in my GitHub repo below!*

🔗 Complete Code & Architecture Breakdown here: [INSERT GITHUB LINK]

#AutomotiveAI #EndToEndDriving #MotionPlanning #ADAS #AutonomousVehicles #MachineLearning #TechReview
