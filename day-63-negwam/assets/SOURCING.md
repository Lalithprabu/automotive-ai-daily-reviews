# Sourcing & honesty disclosure - Day 63 (NegWAM)

**Paper:** Reinforcing End-to-End Autonomous Driving Safety via Negative-Aware Planning in a Unified World-Action Model (NegWAM),
arXiv:2611.08111, submitted 2026-11-05 (cs.RO; NeurIPS 2026 submission per the page). Authors: Zihao Zhang, Xinyu Zhang, Bing Zeng,
Xingyu Chen, Zhenning Li, Bin Yu, Ziyuan Pu (+2 not shown). Southeast University, Shenzhen Loop Area Institute, Guangxi University, Univ. of Macau, BUAA.

## What I could read
arxiv.org/abs/2611.08111 was HTTP 429 (rate limited) on every attempt. The alphaxiv overview page (a machine-generated third-party
summary) gave: the mechanism, the dataset names and three numbers. I did NOT see the paper's equations, architecture figure,
ablations or limitations section.

## Paper-sourced (via alphaxiv overview)
- Safety post-training framework for unified world-action models (WAMs) that jointly predict future traffic states and actions.
- Builds negative trajectories geometrically close to the expert but carrying safety risk; they act as repulsive anchors.
- Negatives are also encoded as scene-conditioned "rider" tokens; the rider branch is training-only and adds no inference overhead.
- Joint objective: flow matching + near-expert negative objective.
- NAVSIM v2 navhard 40.4 EPDMS (+4.1% relative over strongest scorer-free WAM baseline); navtest 93.3 EPDMS; HUGSIM zero-shot (no number given).

## This repo's own reconstruction (NOT paper-sourced)
The token layout and attention mask, hinge/triplet form of the repulsive loss, rider classification loss, all dimensions (d=96, 4 layers),
the synthetic car-following world, how negatives are generated, and every number in results.json. The synthetic toy score is not EPDMS.
