# Simulation Script / Prompt (for an image-video AI or rendering engine)
**Title:** "Dream before you obey" — SafeDriveVLA-style conflict-aware driving, 12 s, 16:9, dark HUD.
**Layout:** left 55% = top-down 3-lane road (ego = blue car, lane centre; 4 grey traffic cars with velocity tags and faint constant-velocity tails).
Right 45% = telemetry: 8 vertical bars "DREAMED COLLISION RISK" (0.5 s … 4.0 s), three horizontal bars FOLLOW / ADAPT / REFUSE, a risk sparkline, a mono-font readout line.
**Three layers every frame (INPUT vs GROUND TRUTH vs MODEL PREDICTION):**
1. INPUT: clean traffic scene + the spoken/typed instruction in a banner ("LEFT", "RIGHT", "KEEP").
2. GROUND TRUTH (neon green, 1 px): ghost ego rectangles showing the TRUE consequence of literally obeying the instruction, a red "X" on the step where it hits a car; green outlines on the risk bars.
3. MODEL PREDICTION: cyan ghost path of the ego "dreaming" the strict-follow manoeuvre; each dream step turns solid red when predicted risk > 0.5; banner colour = chosen mode (green FOLLOW / amber ADAPT / red REFUSE).
**Story beats:** 0–2.5 s KEEP, all clear. 2.5–8 s instruction "LEFT" while a car sits beside in the left lane: cyan dream collides at ~1.5 s, bars go red, banner flips to REFUSE, ego holds lane. 9–10 s "KEEP". 10+ s "RIGHT" after the right-lane car has passed: dream stays cyan, FOLLOW lights up, ego glides into the right lane. Final beat: "RIGHT" again from the rightmost lane → REFUSE (off-road).
**Honesty card (last 1.5 s):** "Synthetic demo · reconstruction of arXiv:2610.20609 · not the authors' model or numbers."
Reference render: `assets/safedrivevla_simulation.gif` (from `simulate.py`).
