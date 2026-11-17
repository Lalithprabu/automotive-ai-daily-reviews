🚀 **[Day 74] Deep Dive: Data-Driven Predictive Control on Changing Grip**

🎬 **AI in action (watch the clip):** three model-free race cars lap a track with two wet sectors. Grey dots show what each controller just saw; the green dashed line is where the car really goes. Inside the wet sector, the dry-data forecast (orange) drifts centimetres off; the pooled one (magenta) stays close.

**What is it?**
DPC (arXiv 2611.21378) steers a race car with no vehicle model, predicting from recorded driving alone. The paper tackles its weak spot, changing tyre-road friction, without a friction estimator: pool dry and wet data.

🏗️ **The Architecture Shift**
* **The Blueprint:** recorded steering/curvature inputs and lateral/heading errors fill a block-Hankel matrix. Willems' lemma: any trajectory is a linear combination g of its columns. Match the last 5 samples, add the planned steering, read off the next 0.75 s.
* **Core Innovation:** no friction estimate. Hankel matrices from different frictions are concatenated column-wise, and the initial-trajectory match decides how they mix.

🔄 **The Evolution: Model-based MPC / single-friction DPC vs. Pooled DPC**
* **Old Way:** NMPC needs a tyre model plus a live friction estimate. DPC recorded at one friction expects grip that isn't there.
* **New Way:** one pooled data set, no estimator, no re-identification when the surface changes.

📈 **Key Improvements**
* 🎯 Paper: pooled prediction error within 1.5x of matched data; mismatched 2.2-3.4x. Tracking 13.7 cm vs NMPC 13.3 cm (sim), 1.6 vs 1.5 cm on low-friction CPM Lab cars.
* 🔬 My 1:10 toy rebuild (20 held-out seeds): mismatch 2.7-3.9x, pooled 1.34x (wet) / 2.05x (dry).
* 🏁 Mixed-friction track: pooled 1.15 cm RMS vs 1.52 (dry data) / 1.77 (wet data), better in 20/20 seeds.

⚠️ **Current Limitations**
* Pooling is a compromise, not sensing: with l2 regularisation, 52% of |g| sits on dry columns whether it is dry or wet.
* Matched data still wins: on a uniform track pooled is 1.5-1.6x worse.
* An oracle linear MPC with the true friction beat all DPC variants on dry and mixed tracks (0.43 cm).
* Built from the abstract only; the paper's regularisation, data sizes and tuning may differ.
* Toy sim with two friction levels: no tyre wear, temperature or in-between grip.

💡 **Visualizing the Concept**
Diagram: dry and wet recordings → two block-Hankel matrices → concatenated side by side → [U_p; Y_p; U_f] g = [u_ini; y_ini; u_f] → Y_f g = 0.75 s forecast → steering QP → car. Inset: wet-only, dry-only and pooled forecasts against the real path in a wet corner.
*Check the simulation and PyTorch code in my GitHub repo below!*

🔗 Complete Code & Architecture Breakdown here: [INSERT GITHUB LINK]

#AutomotiveAI #AutonomousRacing #DataDrivenControl #MPC #VehicleDynamics #MachineLearning #TechReview
