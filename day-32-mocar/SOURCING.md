# Sourcing & disclosure — Day 32 (MoCAR)

**Paper:** *MoCAR: Motion-code Coordinate-aware AutoRegression for Continuous Trajectory Forecasting* — arXiv:2610.06210 (listed 2026-10-06 in cs.CV; ~1 day old at run time). Authors/affiliations were NOT recovered.

## Access
`arxiv.org/abs/2610.06210` and `arxiv.org/html/2610.06210` returned HTTP 429 ("do not fetch again") and were not read. The only paper content used is the alphaxiv.org mirror's abstract-level summary.

## Paper-sourced (abstract level only)
- Decoder-only framework recasting trajectory forecasting as sequential prediction of codes in a learned *continuous* latent space (no clean token exists in trajectory forecasting, unlike language).
- Codes learned from endpoint-normalized trajectory segments; each code jointly captures local geometry and the reference-frame transition induced by the segment.
- Historical motion codes as teacher-forced prefix; future codes generated autoregressively; temporal/map/agent/mode interactions; predicted codes kept in latent memory while scene context updates (no trajectory-space re-tokenization).
- Argoverse: top-tier single-stage results, strong AV2→AV1 zero-shot transfer, better on turn-heavy scenarios. Ablations: learned continuous codes, latent alignment, weak KL, joint tokenizer-predictor optimization are essential.
- **No numeric result was recoverable; none is quoted.**

## This repo's own reconstruction (NOT paper-sourced)
Segment length (5 steps), code dim 16, d_model 64, 2 layers, 3 modes; VAE-style tokenizer MLPs; pose embedding of the segment end pose; per-mode learned embeddings; PointNet lane encoder; no agent-agent interaction (synthetic scenes have a single agent); loss = tokenizer reconstruction + weak KL + winner-take-all trajectory loss + mode CE + latent-alignment MSE (teacher-forced and free-running passes); the one-shot baseline; the entire synthetic junction/road simulator; every number below.

## Honest findings (synthetic, 1 seed, 1,200 held-out scenes, minADE@3 in metres)
| Model | all | turns | road |
|---|---|---|---|
| Constant velocity (ADE) | 1.591 | – | – |
| One-shot baseline | **0.133** | **0.218** | **0.077** |
| MoCAR, no coordinate-aware embedding | 0.221 | 0.337 | 0.179 |
| MoCAR (full) | 0.176 | 0.258 | 0.149 |

- Coordinate-aware pose embedding helps (−20% minADE@3 vs. the same model without it), consistent in direction with the paper's claim that frame awareness matters.
- The paper's headline claim (autoregressive code prediction beats strong baselines, esp. on turns) is **NOT reproduced here**: the simple one-shot baseline is better on every bucket. Plausible, untested causes: tiny model/short training, a synthetic task that is easy for one-shot regression (lane-following with 3 known candidates), exposure bias in free-running rollout, no agent-interaction data.
- Top-1 (highest-score) ADE is ~0.9 m for all models because junction intent is genuinely unpredictable from the history alone in this simulator (aleatoric); minADE@3 is the meaningful multimodal metric.
