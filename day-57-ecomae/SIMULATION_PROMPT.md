# Simulation Script / Prompt (for an image/video AI or rendering engine)
**Format:** 16:9, 1240x640, ~10 s at 7 fps, light technical-dashboard look.
**Left (top-down scene, INPUT -> GROUND TRUTH -> PREDICTION):** a pedestrian walks up the screen toward a grey kerb/crossing band. INPUT = black dots for observed sensor steps; hollow grey circles where the sensor dropped a step (about 1 in 5 in fog). GROUND TRUTH = thick green curve for the true future path (straight / left / right / stop). PREDICTION = two sets of arrows from the pedestrian, one per intent, length proportional to predicted probability: blue = evidential-MAE model, orange = supervised-only (offset slightly so both are visible).
**Right top:** grouped bars for the 4 intents (straight, left, right, stop), orange vs blue, green band behind the true intent.
**Right middle:** purple line = epistemic uncertainty from the evidential head over the observed window, dashed green vertical line = manoeuvre onset.
**Right bottom:** probability assigned to the TRUE intent over time, orange vs blue.
**Header:** "AI in action: pedestrian intent in FOG - sensor drops steps, an evidential MAE encoder reads the rest (synthetic)" + live readout line. Footer: "Synthetic demo, not IMPTC/Waymo/nuScenes/inD data".
