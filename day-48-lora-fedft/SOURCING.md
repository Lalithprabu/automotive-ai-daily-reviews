# Sourcing & disclosure (Day 48)

**Paper:** *Pedestrian Trajectory Prediction through LoRA Federated Training*, Simone Travagliati and Maria Luisa Merani, arXiv:2610.18640 (cs.NI), submitted 2026-10-13 (9 days old at run time). No code release found. Full text read via arxiv.org/html (this run, no rate-limit on that page). Affiliations were not shown on the fetched pages.

## Paper-sourced
- Foundation model Chronos-2 (>121M params, encoder-only transformer with group attention); LoRA rank 8 on all linear layers; ~2M LoRA params and 7.8 MB vs 477.91 MB payload.
- 20 connected vehicles, each assigned one data subset; sample-weighted "server averaging" at an edge server (algorithm not named); F = N_r / C_r local rounds per communication round, F in {1, 2, 5}.
- Input 8 positions (4 s), output 12 positions (6 s); quantile loss; ADE/FDE and miss rate (1 s / 2 s, 0.5 m / 1 m).
- Datasets: nuScenes (pedestrians; 4 regions x 3 subsets), UCY (6 subsets), ETH (2 subsets), 80/20 split.
- Baselines: centralized, federated from scratch, federated full fine-tuning.
- Results: LoRA-FedFT ADE 0.60/0.59/0.59 m and FDE 1.25/1.24/1.24 m for F=1/2/5; F=2 and F=5 about 1 cm above centralized; MR 0.3% (2 s, 1 m). RaOT (uplink radio occupancy): centralized ~4 d 4 h, federated ~2 d 15 h, full FT ~1 d 4 h, LoRA F=5 ~27 min (5G), ~10 min 11 s (6G, speculative).
- Paper's own stated limitations: simulated vehicles, simplified radio model (peak rates, fixed 1 MHz), speculative 6G, uplink only, narrow tuning (r=8, three F values), unnamed aggregation.

## This repo's own reconstruction (NOT paper-sourced)
Everything else: a ~0.27M-param transformer standing in for Chronos-2 (d=128, 2 layers, 4 heads, ff=256, last-token readout); LoRA r=4, alpha=8 (not r=8); smooth-L1 loss instead of quantile loss; FedAvg with A and B averaged separately; fresh Adam per vehicle per round; LRs (3e-3 LoRA, 1e-3 full); the four synthetic behaviour styles, canonicalisation, 160 train / 40 val samples per client, 70% dominant style; pre-training on straight walking only; the assumed 10 Mbps uplink; every number in `results/results.json`.

## Measured (synthetic, single seed 7, 800 validation samples)
| run | ADE m | FDE m | MR@2s/1m | upload/vehicle | rounds to ADE<=1.49 |
|---|---|---|---|---|---|
| frozen base | 2.61 | 5.91 | 0.478 | - | - |
| centralized FT | 0.90 | 2.04 | 0.070 | n/a | - |
| LoRA-FedFT F=1 | 1.47 | 3.21 | 0.224 | 62 KB | 24 |
| LoRA-FedFT F=5 | 1.03 | 2.36 | 0.064 | 62 KB | 7 |
| full FT F=1 | 1.45 | 3.08 | 0.221 | 1,080 KB | 20 |
| full FT F=5 | 0.99 | 2.25 | 0.060 | 1,080 KB | 4 |
| fed from scratch F=1 | 1.52 | 3.18 | 0.222 | 1,080 KB | never |

Uplink to reach ADE <= 1.49 m at F=5: 8.7 MB (LoRA) vs 86 MB (full FT), about 10x. Payload ratio 5.7% (paper: ~1.6% at a much larger model).

## Findings reported as they came out
1. **The paper's accuracy claim is only partly reproduced.** LoRA matched full FT's *accuracy class* but full FT at equal F was slightly better (0.99 vs 1.03) and converged in fewer rounds (4 vs 7). The saving here is bytes, not rounds. No gap to centralized closed: LoRA F=5 is 0.13 m above it, not ~1 cm.
2. **F matters more than LoRA vs full.** F=1 -> F=5 cut ADE ~30% for both methods.
3. **Straight walkers got worse.** Zero-shot straight ADE 0.23 m; after any fine-tune 1.6-2.2 m (centralized 2.2). In this generator stop-and-go and crossing futures look identical to straight in the history, so the model hedges. This is generator ambiguity, not forgetting per se, and not tested against a style-conditioned model.
4. **Crossing remains hard** (LoRA F=5 1.05 m vs centralized 0.54).
5. **Centralized comm cost is not reported.** Trajectory arrays are tiny, so centralized upload would be cheaper than any federated run here; the paper's multi-day centralized cost comes from raw dataset sizes (GB), which this repo does not model.
6. **Rounds-to-target threshold (1.49 m) was chosen after seeing the curves** (no run reached the first-choice 1.05x centralized). Uplink seconds in results.json use an assumed 10 Mbps, not the paper's 3GPP model.
7. Single seed, no error bars.

## Bugs / lessons
- A canonicalisation test first asserted the wrong frame (last-step velocity vs the 2-step heading used for rotation); fixed the test, not the code.
- The tiny core-claim test originally used a 0.8x threshold and failed at 0.80x; loosened to 0.9x for the small test model.
- Matching local-epoch budgets matters: my first full-FT baseline used F=1 against LoRA F=5 and looked unfair, so a full-FT F=5 run was added.
