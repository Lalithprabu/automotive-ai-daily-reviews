# Sourcing & honesty notes (Day 47)

**Paper:** *Temporal Residual Bottleneck for Robust Asynchronous Collaborative Perception* — Yazgan, Abouelazm, Zöllner (FZI Research Center for Information Technology; Karlsruhe Institute of Technology). arXiv:2610.10090, submitted 2026-10-07, cs.CV. Authors state code "will be publicly released" (url.fzi.de/8dk38) — not found at run time; this repo is an independent reconstruction.
Access: the arXiv HTML full text was fetched (via a summarising fetch tool, not read verbatim), so architecture hyper-parameters and result tables below come from that summary.

## Paper-sourced
- Pose-warped anchor B; Δt-conditioned xLSTM (m-s-m, k=4, Fourier delay features + FiLM, pose/time token); attention-free gated residual bottleneck, zero-init output, residual scale 1.5; object gate from stale∪warped objectness, dilation 4, floor 0.05, detached; F̂ = B + G⊙δF; L = L_det + 0.02·L_app with four terms (directional/magnitude/improvement/over-application; weights 1/0.25/0.25/0.1, margin 5e-5, ρ=1.5); no feature-reconstruction loss.
- Results quoted in the post: DAIR-V2X irregular-500ms AP@0.7 0.635 (LRCP 0.611); payload 1.01 MB vs 6.45 MB (LRCP); OPV2V irregular-300ms 0.884/0.818 vs CoDynTrust 0.857/0.795; ablation 0.640 (full) vs 0.622 (pose-warp only) at DAIR-V2X irregular-300ms; sync AP@0.7 0.664 vs LRCP 0.669. Limitations list is the authors'.

## This repo's own (NOT from the paper)
Everything else: BEV size 32×32×8, channel widths (d=32, bottleneck 32/64 vs paper 64/256), dilated-conv encoder, the *exact formulas* of the four L_app terms (the paper names them; formulas are my reading), the discrepancy/agreement inputs' exact definitions, the xLSTM cells (compact re-derivation from Beck et al. 2024, no up-projection/conv blocks), the objectness head (trained on fresh features, stand-in for the paper's detector objectness), the ego-fusion head, the whole synthetic world (CTRA objects, moving collaborator, ego sees only x<0), metrics, thresholds.

## Measured here (synthetic data, ONE training seed, 600 test scenes = 6 delays × 100; threshold chosen on val)
| Variant | params | F1 | blind-zone recall | feature MSE |
|---|---|---|---|---|
| stale (no compensation) | 28,978 | 0.784 | 0.610 | 0.0049 |
| pose-warp only | 28,978 | 0.883 | 0.774 | 0.0021 |
| TRB, no gate | 144,858 | 0.883 | 0.781 | 0.0067 |
| TRB (full) | 144,858 | 0.881 | 0.777 | 0.0030 |

**Not reproduced:** the paper's gain of full TRB over pose-warp alone. Here they are statistically indistinguishable (single seed, differences ≤0.5 pt). TRB's feature MSE is *worse* than pose-warp at 0-delay (0.0027 vs 0.0009) and roughly equal at 0.75 s (0.0043 vs 0.0042). What *is* reproduced: pose-warping is a strong anchor (+10 F1 over raw stale features), and the gate sharply reduces spurious residual energy (MSE 0.0067→0.0030, −55%).
Untested hypotheses for the gap: only 700 steps at batch 16 (paper: 30–60 epochs); a 5×-smaller bottleneck; per-pixel recurrence with a small receptive field; synthetic motion that is almost linear over ≤0.75 s so the pose-warp residual is small. No tuning was done to chase the paper's number.
Demo caveat: `simulate.py` shows 24 consecutive held-out scenes at the *hardest* delay (0.75 s), not cherry-picked; running blind recall there is old 0.358 vs new 0.389 (n=24 scenes, within noise).
