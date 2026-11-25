🚀 **[Day 81] Deep Dive: Trajectory Prediction When Sensors and V2X Lie a Little**

🎬 **AI in action (watch the clip):** highway traffic seen through noisy sensors (grey dots) and V2X messages (orange, hollow = packet lost). Same Transformer, two trainings: clean data (red) vs noise-augmented (blue). The noise ramps 0 → 4, then traffic plays out against the green truth.

**What is it?**
arXiv 2611.28128 (Salehi, Lusvarghi, Sepulcre, Gozalvez, 24 Nov) tests 5 Transformer predictors on noisy position, heading and speed, worst for V2X-received states. It asks which fix works: input denoising, or retraining on noisy data.

🏗️ **The Architecture Shift**
* **The Blueprint:** interaction-aware Transformers (per-agent temporal encoder → agent-to-agent attention → multi-modal decoder), trained on curated, offline-filtered tracks.
* **Core Innovation:** the shift is in the data, not the layers. Retrain on a noise-augmented dataset (released openly) instead of bolting a denoiser in front.

🔄 **The Evolution: Clean Training + Denoiser vs. Noise-Augmented Retraining**
* **Old Way:** train on clean tracks, filter inputs at runtime. The model still trusts its last observation, which may be biased or stale.
* **New Way:** the predictor sees the noise in training and learns how far to trust each input. Paper: up to 6x degradation without it, under 1.5x with it.

📈 **Key Improvements (my synthetic rebuild: 3,000 held-out scenes, 0.7M params, 2 seeds)**
* 🎯 At max noise, minADE6 1.07 m vs 2.83 m for the clean-trained model (-62%).
* 🧹 It beats both denoisers at every noise level: Kalman filter 2.46 m, learned Transformer denoiser 1.60 m.
* 📡 V2X-received agents: 1.53 m vs 3.81 m.

⚠️ **Current Limitations**
* "Under 1.5x" held only up to my mild level (1.32x). At max noise: 3.25x. At the level where the clean model hits ~6x, the augmented one is at 1.9x.
* Robustness costs accuracy on clean input: 0.33 vs 0.25 m (+30%). In 96 scenes it was clearly worse (clip, scene 3).
* V2X agents still have 2x the error of sensor-tracked ones. Retraining helps but does not remove GNSS bias or stale messages.
* My noise model is invented. A model trained on one noise profile may not hold under another sensor suite or V2X stack.
* Abstract-only rebuild: one architecture (paper: 5), synthetic highway, 5k CPU steps.

💡 **Visualizing the Concept**
Diagram: noisy states (sensor jitter, V2X bias, packet loss) feed three rows: A) clean-trained predictor, B) denoiser → same frozen predictor, C) noise-augmented retrain. Each row: temporal Transformer → social Transformer → 6 modes. On the right, error vs noise level, with C the flattest curve.
*Simulation and PyTorch code in my GitHub repo below!*

🔗 Complete Code & Architecture Breakdown here: [INSERT GITHUB LINK]

#AutomotiveAI #V2X #TrajectoryPrediction #ADAS #AutonomousVehicles #MachineLearning #TechReview
