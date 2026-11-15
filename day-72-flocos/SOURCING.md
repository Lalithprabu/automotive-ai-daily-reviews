# Sourcing - what comes from the paper and what is mine

**Paper:** FloCOS: Predicting Compact Forward Occupancy Sets of Surrounding Vehicles via Conditional Flow Matching.
Chuansheng Dong, Yinxiao Zhan, Zongdian Li, Peng Hou, Jin Wang, Jiayu Fan. arXiv 2611.15488 [cs.RO],
submitted 11 Nov 2026 (v1; 8 pages, 4 figures, 3 tables). Affiliations per alphaXiv metadata: Soochow University and
Zhejiang University. No public code found.

## What I could read (2026-11-15)
* arXiv abstract page (title, authors, date, comments, category) - read.
* arXivSignals and alphaXiv overview pages - abstract-level only.
* arXiv HTML and PDF - **rate-limited (HTTP 429) on every attempt across ~40 minutes**; alphaXiv PDF mirror
  blocked by robots.txt. So **no table, no number, no architecture detail from the body of the paper is used anywhere.**

## From the paper (abstract level)
* Problem: forward occupancy prediction for surrounding vehicles either under-approximates the true occupancy
  (collision risk) or is over-conservative (shrinks feasible space, planning fails).
* Representation: a time-varying **union of half-space convex polygons**, parameterised by **temporal basis coefficients**.
* Model: **conditional flow matching** over those coefficients.
* Training: a **coverage loss** (missed occupancy) and a **union-area loss** (excess area).
* Use: the non-convex union enters **hyperplane-based collision-avoidance constraints** in trajectory optimisation.
* Evaluation: Waymo Open Motion Dataset; claims high coverage with compact sets and a higher **interaction completion
  rate** than the evaluated baselines; real-vehicle experiments in interactive scenarios.
  The abstract gives **no numbers**.

## My choices (not from the paper)
M = 6 components (noise draws), 8 fixed normals at 45 deg, Bernstein basis of degree 5 over a 3 s / 12-step horizon,
softplus offsets, GRU agent encoder, FiLM residual-MLP velocity field, rectified-flow path, Euler NFE = 4, the
two-stage schedule (FM pre-training on tubes fitted to the realised future, then FM + coverage + soft union-area
through the differentiable sampler), loss weights, the synthetic highway world, the baselines (DetReg, MTP-6,
physics reach set), and the ego-merge planner that filters candidates with the half-space constraints instead of
solving an optimisation.
