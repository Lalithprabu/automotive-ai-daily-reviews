# Sourcing / honesty note
* Paper: SafeDriveVLA: Navigation-Conditioned World Model Dreaming for Conflict-Aware End-to-End Autonomous Driving
  (Xie, Zhang, Wang, Qu, Liang, Kong, Lu, Christensen, Chen), arXiv:2610.20609, submitted 2026-10-14, cs.RO.
* Only the arXiv abstract page was read (no PDF, no code, no project-page URL). Everything about the paper's
  numbers comes from that abstract: Bench2Drive DS 88.16, SR 68.64%, +49.3% closed-loop language following,
  collisions under unsafe instructions 60 -> 38 vs. best SimLingo variant.
* NOT from the paper (this repo's own choices): the 3-lane synthetic world, the 3-class labelling rule, GRU dreamer,
  all dimensions, all losses, the toy 'expert' lateral profiles. The paper is a full VLA; this is a decision-structure toy.
* Results here are on synthetic data and are NOT comparable to the paper's.
