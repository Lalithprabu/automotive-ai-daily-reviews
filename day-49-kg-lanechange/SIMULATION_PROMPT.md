# Simulation Script / Prompt (for an image/video AI or rendering engine)

**Title card (1 s):** "Day 49 — Will it cut in? KG-Embedding + Bayesian lane-change prediction".

**Scene (dark UI, 16:9, 1170x504 reference).** Three panels, synchronized, 11 s of sim time at 7 fps playback.

1. **Input / Ground truth / Prediction overlay (top-left, wide).** Top-down 3-lane highway, white solid edges, dashed lane lines. A blue "cut-in vehicle" drives in the left lane, 18 m/s, and drifts right into the ego lane with a bell-shaped lateral velocity (peak at the lane-line crossing, t = 5.25 s). Two overlapping ego cars at 28 m/s: a semi-transparent RED ghost labeled "ego (reactive)" and a solid GREEN car labeled "ego (KG-Bayes)". **Prediction overlay:** a gold halo around the cut-in vehicle whose radius and line weight grow with P(enter lane), plus text "P(enter lane)=0.xx". **Ground truth:** the actual lateral path is the blue car's real motion; flash "REACTIVE: COLLISION" in red when the red ghost's bumper gap hits 0 (~t = 6 s).
2. **Live posterior (top-right).** Horizontal bars for LLC / LK / RLC, labelled with probabilities, updating each frame. Early frames: LK ~0.95. Around t = 3.2 s the LLC bar surges past 0.6 and stays; show a small caption "evidence: lateral accel LEFT, front TTC HIGH" next to the bar.
3. **Telemetry (bottom, full width).** Deceleration (m/s²) vs. time, red (reactive, jumps to 9.0 at ~5.6 s) vs. green (KG-Bayes, soft ramp to ~2.3 at ~3.3 s). Dashed gap-to-intruder curves on a right axis; the red one hits 0. White vertical cursor sweeps with time.

**Camera/motion:** BEV follows the green ego. **Palette:** bg #0e1117, text #e6edf3, red #ff5c5c, green #3ddc97, blue #58a6ff, gold #f2cc60. **End card:** "Same sensors. Same cut-in. Intent known 2 s earlier."

*A real render of this exact overlay is produced by `python simulate.py` (assets/simulation.gif).*
