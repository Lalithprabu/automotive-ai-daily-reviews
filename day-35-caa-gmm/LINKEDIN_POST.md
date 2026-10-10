🚀 **[Day 35] Deep Dive: CAA-GMM — Attention-Weighted Gaussian Mixtures for Trajectory Forecasting**

🎬 **AI in action (watch the clip):** a car approaches a junction behind a lead vehicle. Left: a raster-CNN predictor. Right: CAA-GMM. Green = truth, fan = 6 Gaussian modes (opacity = probability). Halfway through, the lead vehicle's V2X message is lost. Watch what each model does.

**What is it?**
CAA-GMM (Raftari, Soorchaei, Fallah — UCF/AMD, arXiv 2610.09174, Oct 2026) predicts multimodal vehicle futures as a Gaussian mixture, so every mode carries its own uncertainty. A lightweight attention block weighs *who* matters in traffic and *which* map cues matter.

🏗️ **The Architecture Shift**
* **The Blueprint:** rasterized map + motion history → attention over neighbours and context → K Gaussian components.
* **Core Innovation:** attention decides how much to trust interactions vs. map cues. Paper also tests imperfect communication/perception.

🔄 **The Evolution: Raster CNN + MLP vs. CAA-GMM**
* **Old Way:** neighbours exist only as pixels; the net must infer "that car slows me down" from a blob.
* **New Way:** neighbours are tokens; the ego query attends to them, and a gate balances motion, interaction and map.

📈 **Key Improvements**
* 📄 Paper (abstract only, no tables retrievable): competitive or better than raster baselines on nuScenes & Argoverse 2 at low compute.
* 🧪 My toy rebuild (149K params, synthetic, 1 seed): minADE@6 0.055 vs 0.136 m; top-1 ADE 1.42 vs 2.24 m.

⚠️ **Current Limitations**
* Trained on clean comms, my CAA-GMM got WORSE than the baseline under outage: minADE 0.58 vs 0.21 m, and wildly overconfident (mixture NLL +102).
* Training with random message dropout fixed the cliff: at 60% drop, 0.142 vs 0.171 m. At total outage it only ties the baseline (0.175 vs 0.171).
* Attention on the true lead was only 0.45 vs 0.32 uniform: interpretability is partial. Synthetic data, one seed; not the paper's benchmarks.

💡 **Visualizing the Concept**
Two BEV panels side by side, a yellow route-hint lane, the lead car turning red-dashed when its message drops; below, a bar chart of attention on neighbours and the 3-way gate.

🎥 **Simulation Script/Prompt**
"Dark top-down BEV, 1 m/px. White ego car approaches a T-junction; grey cars; orange lead car ahead. Draw green ground-truth path, white history. Left panel: orange fan of 6 trajectories. Right panel: blue fan plus 2-sigma ellipses; opacity and labels = mode weight. Halfway, outline the lead red and dashed with 'MSG LOST'. Bottom: running minADE lines; attention bars."

🔗 Code, tests & sim: [INSERT GITHUB LINK]

#AutomotiveAI #TrajectoryPrediction #V2X #ADAS #MachineLearning #AutonomousVehicles #TechReview
