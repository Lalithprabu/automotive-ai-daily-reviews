# Simulation Script / Prompt (for an image-video AI or rendering engine)

**Title:** "One model, four jobs" — Foundation4D-style perception overlay, 10 s, 16:9, dark HUD style.

**Layout:** left 40% = 2×2 grid of surround cameras (FRONT, RIGHT, REAR, LEFT) of a dusk suburban street with a few parked and moving cars.
Centre 30% = top-down BEV, ego triangle in the middle. Right 30% = 3D occupancy (z-averaged) + telemetry strip.

**Three layers on every frame (the input vs. ground truth vs. prediction overlay):**
1. INPUT: clean camera video, no overlays for the first 1.5 s.
2. GROUND TRUTH (neon green, 2 px): a 3D box on each car in the camera views, matching green rectangles in the BEV and a green outline on the occupancy map.
3. MODEL PREDICTION: turbo-colormap depth wash (35% opacity) on each camera; cyan "+" markers in the BEV at each predicted object centre over a magma heat-map; blue occupancy voxels that gradually fill and snap onto the green outline. Cyan markers lag the green boxes by ~0.3 s at first, then lock on as time-context accumulates (frame 2 attends to frame 1).

**Motion:** ego creeps forward; two cars cross laterally; one car passes from FRONT into RIGHT camera — its cyan marker hands off between camera views with no dropout (show the "cross-camera attention" as a thin glowing line between the two camera tiles for 0.4 s).

**Telemetry (bottom-right):** "t=NN | det TP 2/2 FP 0 | running depth AbsRel 0.13" updating each frame, plus a four-icon strip (DEPTH · DETECT · OCCUPANCY · GAUSSIANS) all lit at once with caption "ONE FORWARD PASS".

**Honesty card (last 1.5 s):** "Synthetic demo · reconstruction of arXiv:2610.30340 · not NVIDIA's model or numbers."

Reference render: `assets/foundation4d_simulation.gif` (produced by `simulate.py`).
