# Sourcing disclosure

This repository is an independent, third-party **reconstruction** of the mechanism
described in:

> **INTERACT: Interactive Planning for Autonomous Driving via Anchor-Conditioned
> Prediction and Trust-Region Refinement**
> Aron Distelzweig, Andreas Look, Faris Janjoš, Steffen Hagedorn, Luigi Palmieri,
> Joschka Boedecker. Robert Bosch GmbH / University of Freiburg / Coburg University.
> arXiv:2609.31137, submitted 2026-09-25.

It is **not** the authors' code, was built with **no access to the paper's full
text or figures** (arXiv full text returned a 429 rate-limit wall at fetch time,
consistent with this project's standing issue since Day 9), and contains **no
numbers copied from the paper** anywhere in code, README, or the LinkedIn post.

## What is actually paper-sourced (from the arXiv abstract page + one third-party mirror only)

| Claim | Status |
|---|---|
| Addresses interactive driving scenarios where other agents' responses depend on ego's trajectory/intent | Paper-sourced (abstract) |
| Decomposes planning into (1) prediction across a small set of diverse "intent"/anchor hypotheses derived from map geometry, and (2) CEM-based optimization within each intent, with trust-region penalties | Paper-sourced (abstract) |
| Key claim: agents respond to ego's *intent*, not exact path, so one reactive prediction stays valid across a family of plans sharing that intent -> predictor queried once per anchor (a handful of times), not once per candidate trajectory (potentially hundreds) | Paper-sourced (abstract) |
| Reports SOTA results on nuPlan and interPlan benchmarks, with largest gains in interactive scenarios | Paper-sourced (abstract) -- **no exact numeric table was recoverable**; this is a qualitative claim only |
| No public code release found | Verified absence (searched, none found) |
| Authors, affiliations, arXiv ID, submission date | Paper-sourced (arXiv metadata) |

**Nothing else about this paper is claimed to be paper-sourced.** In particular, the
paper's exact network architecture, exact loss function(s), exact anchor count K,
exact CEM hyperparameters, exact cost-function terms, exact benchmark numbers, and
exact ablation results were **not accessible** and are **not reproduced** anywhere
in this repo, the README, or the LinkedIn post.

## What is this project's own reconstruction default (everywhere else)

Every one of the following is a design choice made by this repository, not a fact
about the paper:

| Component | This repo's default | Paper says |
|---|---|---|
| Number of anchors K | 5 (`aggressive_merge`, `accelerate_past`, `hold_lane_delay`, `cautious_yield`, `decelerate_follow`) | unspecified |
| Anchor representation | 2D (target lateral offset, target speed delta) intent vector | unspecified (paper: "derived from map geometry") |
| Predictor architecture | GRU encoder + FiLM conditioning on the anchor vector + lightweight cross-attention decoder (`src/models/anchor_conditioned_predictor.py`) | unspecified |
| State representation | simplified 2D (lat, spd) trajectories, 4D per-timestep history features | unspecified |
| History / future horizon | H=8 past steps, T=10 future steps | unspecified |
| Cost function terms | collision + comfort (jerk) + progress + trust-region penalty, with specific weights in `config.yaml` | unspecified (paper says CEM + "trust-region penalties" qualitatively) |
| Trust-region penalty formula | `lambda * relu(||cond - anchor_center|| - radius)^2` | unspecified |
| CEM hyperparameters (n_iter, n_samples, elite_frac, init_std) | `config.yaml` defaults (4 iters, 32 samples, 20% elite) | unspecified |
| Dataset | fully synthetic merge/negotiation scenario (`src/data/synthetic_interactive_scenario.py`), with a hand-authored reactive rule (`k_speed_reaction`, `k_lateral_reaction`) | N/A -- paper trains/evaluates on nuPlan/interPlan, which this repo does not have access to |
| All numeric results (loss curves, ADE/FDE, predictor-call counts, wall-clock times) | measured by actually running `train.py` / `simulate.py` in this repo, on the synthetic dataset above | **not comparable to the paper's nuPlan/interPlan numbers**, which were never accessible |

## Bottom line

Read this repo as: *"here is a small, honestly-labeled, runnable illustration of
the mechanism the abstract describes, with its own synthetic benchmark and its own
measured numbers -- clearly not a reproduction of the paper's actual results."*
Every number in `README.md` and `LINKEDIN_POST.md` under "Key Improvements" is a
number this repo's own code measured on its own synthetic scenario, never a number
attributed to the paper.
