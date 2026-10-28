# Sourcing
Paper: "Single-Cycle Battery Remaining Useful Life Prediction with Compressed Continuous Wavelet Features and Deep Neural Networks"
(Kim, Jung, Kim, Park, Ryu; Ajou University), arXiv:2610.34159, submitted 2026-10-25.
Read: abstract only (arXiv listing + alphaXiv summary; the PDF could not be rendered/was rate limited).
From the abstract: CWT on cycle-wise voltage/current/capacity -> scalograms; scale compression; 8 benchmark datasets from 7 public sets;
outperforms prior single-cycle RUL methods; robust for 1-100 input cycles; Grad-style explanations show focus on transition regions.
NOT known (not in abstract): backbone, wavelet type, scale counts, loss, any numeric result. Everything below those lines
(Morlet CWT, 32->8 scale average-pool, GroupNorm CNN, mean-pool over cycles, log-life regression) is THIS REPO's own design.
All numbers in README/results.json come from a synthetic generator, not real cells and not the paper.
