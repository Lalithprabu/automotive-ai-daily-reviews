# Simulation Script / Prompt (for an image/video AI or rendering engine)
**Title card (1 s):** "Day 41 — How sure is the model? X-TRACK-UQ". Dark charcoal background.
**Scene:** 1100x520, side-scrolling 4-lane highway, top-down, lateral axis stretched ~3.5x so lane changes are visible. Dashed white lane lines.
**INPUT (0-1 s):** A grey trail 3 s long behind a rectangular ego-target vehicle, labelled "INPUT: 3 s history". Neighbour cars optional, faded.
**GROUND TRUTH (1-5 s):** a thick green polyline extends forward 5 s following the true path (lane change or hard brake); a solid green car glyph rides its tip.
**PREDICTION:** a blue polyline = model mean path, hollow blue car glyph at its tip. Every 0.6 s drop a pair of ellipses centred on the blue path: thin RED outline = raw "95%" region (visibly too small at long horizons), filled translucent CYAN = conformally calibrated 95% region. Ellipses grow nonlinearly with horizon and tilt with the curve.
**TELEMETRY (bottom 170 px):** text "RAW 95% region: truth INSIDE/MISS" and "CONFORMAL 95%: truth INSIDE/MISS" in green/red; sigma_x, sigma_y, position error; two horizontal coverage bars (red raw, cyan conformal) with a white tick at 95%, filling as scenes accumulate (~90% vs ~95%).
**Ending (2 s):** freeze on a frame where the red ring misses the green car but cyan contains it; caption "Calibrated uncertainty = safer gap selection". Footer: "Synthetic data - reconstruction of arXiv:2610.11580".
Runnable version: `python simulate.py --mp4`.
