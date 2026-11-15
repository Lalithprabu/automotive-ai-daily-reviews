# Code provenance and sourcing notes

Moved here from the main README during the November 2026 reorganisation (text unchanged).

Days 1–4's code was carried forward from each day's own original daily-review session and its own repo package. Days 5–7 (DriveZero, COSTER, MC-DeTra) and Days 10–14 (READ, DRiF, BeamTransFuser, LDE, Sparse-BEVNet) were **freshly reconstructed** when each package was assembled, because each scheduled daily run executes in its own isolated container and the original sessions' files were not retrievable afterward — only the papers' architecture descriptions (and, for some days, verbatim code excerpts) survived in the shared project log. Day 15 (MM-Future) was built and pushed directly in the same live session that has GitHub write access, the first day not routed through a zip hand-off. Every day's code and every `simulate.py` — reconstructed or original — was independently re-verified when its package was assembled: every `pytest` suite passes, every `train.py` runs to completion, and every `simulate.py` renders a real GIF. See each day's own README for its specific sourcing/reconstruction note and what is and isn't independently verified from the source paper itself.

**Days 31–37** were each built in a scheduled cloud session from an abstract-level read of the paper (arXiv full text was rate-limited, HTTP 429, every day), so every architecture dimension, dataset and number in those folders is the repo's own disclosed reconstruction; each folder's `SOURCING.md` lists exactly what is paper-sourced. Weekly summary carousel: *The AI Road Trip, Leg 5* (Days 31–37).

This is an independent, best-effort reference implementation series for educational purposes — it is **not** any of the papers' authors' official code release.
