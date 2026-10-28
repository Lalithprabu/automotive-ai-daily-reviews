# Simulation Script / Prompt (image/video AI or rendering engine)
**Title:** "One cycle in, lifetime out" - wavelet battery-life forecasting, 14 s, 16:9, dark lab-HUD style.
**Layout:** left-top: charge-voltage curve of a featured cell (grey = cycle 1, white = latest cycle). Middle-top: its compressed scalogram (magma colormap, 8 scale rows x time). Right 40%: a fleet of 12 horizontal bars sorted by life.
Bottom-left: MAPE-vs-cycles telemetry line.
**Three layers every frame (INPUT vs GROUND TRUTH vs MODEL PREDICTION):**
1. INPUT: the raw voltage curves and the scalogram updating as a "cycles observed" counter climbs 1 -> 50.
2. GROUND TRUTH: neon-green hollow bars = each cell's true cycles-to-80%-capacity (hidden from the model).
3. MODEL PREDICTION: cyan dots = predicted life; thin connector turns red when error > 30%; MAPE line (cyan) falls against a dashed red mean-life baseline.
**Story beats:** 0-2 s n=1: dots scattered, bright red connectors, banner "ONE CYCLE". 2-9 s counter ticks up; dots slide toward the green bar edges, scalogram ridges sharpen. 9-12 s long-life cells still lag (hold on them). 12-14 s honesty card.
**Honesty card:** "Synthetic demo - reconstruction of the idea in arXiv:2610.34159 - not the authors' model or numbers."
Reference render: `assets/cwt_rul_simulation.gif` (from `simulate.py`).
