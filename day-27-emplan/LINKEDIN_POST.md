🎬 **AI in action:** same scene, same inputs, two planners. 🔴 Red = classic single-trajectory regression. 🔵 Blue = EMPlan-style sparse anchors + offsets. 🟢 Dotted green = expert. Watch the red path split the difference between "brake" and "overtake" and drive into the lead car, while blue commits to a clean lane change. Live collision / reward / error meter on the right.

🚀 **[Day 27] Deep Dive: EMPlan — Efficient Multi-Modal Planning with Reward-Guided Preference Optimization (arXiv:2609.38862)**

**What is it?**
A trajectory planner for end-to-end driving that proposes a small set of sparse anchor trajectories, refines each with a learned offset, then fine-tunes using rule-based rewards instead of costly human preference pairs. Built for real-time use, evaluated on NAVSIM (non-reactive).

🏗️ **The Architecture Shift**
* **The Blueprint:** Scene encoder → sparse anchor candidates (low latency) → offset refinement → pick the best-scoring candidate.
* **Core Innovation:** Stage 2 fine-tunes with *unpaired* preference supervision from rule-based rewards (collision, drivable area, progress). Every candidate is labeled good or bad independently, so no annotated winner/loser pairs.

🔄 **The Evolution: Imitation regression vs. EMPlan**
* **Old Way:** One regressed trajectory trained by imitation. When demos are multi-modal (brake OR overtake), it averages the modes and inherits causal confusion.
* **New Way:** Keep the modes as anchors, score them, then let rule-based rewards push probability away from unsafe ones.

📈 **Paper claims**
* ⚡ Favorable accuracy/efficiency balance under real-time constraints on NAVSIM.
* 🎯 I could not access the paper's numeric table, so I am quoting none.

🧪 **My toy reproduction (synthetic 3-lane scenes, NOT paper numbers; 3 seeds):**
* Collision rate: regression 53.9% → anchors+offsets (imitation only) 6.3% → + reward-guided stage 2.5%.
* Mean rule reward: −2.88 → +1.98 → +2.36 (expert 2.52).
* Honest catch: distance to the expert's exact path *rose* slightly (2.57m → 2.84m). The planner often picks a different but safe mode.

⚠️ **Limitations**
* Reward-labeled preference only encodes what your rules can express; unmodeled hazards stay invisible.
* Anchor vocabulary caps what can be planned; offsets must cover the gap.
* Non-reactive benchmark: other agents don't respond to the ego.
* My reconstruction is abstract-level (architecture and loss are my own), on toy data.

🎞️ **Simulation Script/Prompt**
"Dark top-down BEV, 3 lanes, split panel. Left: road with yellow lead car; faint grey fan of 16 anchor trajectories; green dotted expert path; red regression path drifting between lanes and ending in the lead car; blue path committing to a smooth lane change. Cars advance every 0.5s over 4s. Right: live meters for collision flag, rule reward, and ADE per planner, red turning to a COLLISION banner."

💡 **Visualizing the Concept**
"Diagram: scene encoder → two heads (anchor scores, offsets) → candidate trajectories → argmax. Below: stage 2 loop where a rule-reward box stamps each candidate with a green check or red cross (unpaired labels), feeding back into the scorer."

🔗 Complete Code & Architecture Breakdown here: [INSERT GITHUB LINK]

#AutomotiveAI #MachineLearning #ADAS #EndToEndDriving #MotionPlanning #AutonomousVehicles #TechReview
