# Sourcing & honesty disclosure — Day 27 (EMPlan)
**Paper:** Efficient Multi-Modal Planning with Reward-Guided Preference Optimization for Autonomous Driving (EMPlan), arXiv:2609.38862, submitted 2026-09-30. Authors: Chenglin Chen, Lujia Wang, Xinhu Zheng, Jun Ma, Haoang Li.
## Paper-sourced (alphaxiv overview only; arxiv.org/abs returned HTTP 429)
- Hybrid architecture: sparse anchors for coarse candidates + offset module for refinement; two-stage training (pretrain, then reward-guided fine-tune) with rule-based rewards and unpaired preference supervision; NAVSIM non-reactive evaluation; no numeric results recoverable.
## This repo's own reconstruction (NOT paper-sourced)
Everything else: scene encoder, K-means anchors, offset head, the KTO-style unpaired loss with a frozen reference, the reward function, the synthetic world, all numbers in results.json. Not comparable to NAVSIM.
## Findings
- 3 seeds (multiseed_out.txt): collision 53.9±0.4% (regression) → 6.3±1.2% (stage 1) → 2.5±0.4% (stage 2); stage 2 reward gain 1.98→2.36.
- ADE to expert got slightly worse with anchors (2.57→2.84m): safe alternative modes, not errors.
- Regression's collisions are partly due to the toy's strict bimodality; the gap is likely exaggerated vs. real data.
- Stage 2 labels use the rule reward that also defines the expert and the eval metric: this is partly circular; real gains would be on held-out rule sets.
