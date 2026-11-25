## Simulation Script / Prompt (for an image/video AI or rendering engine)

**Title card (1.5 s):** "Day 79 - One world model, three sensors, 2 seconds ahead. RGB + thermal + LiDAR depth forecast as ONE video, in fog and snow. SYNTHETIC demo."

**Layout (16:9, dark UI, 1280x720, 10 fps):**
* **Left: a 4 x 3 grid of sensor tiles (224 x 168 each).**
  * Rows: RGB camera, thermal camera (inferno palette), LiDAR depth. The LiDAR row shows 16 beams drawn as horizontal scan lines projected into the RGB view (near = red, far = blue, black = no return).
  * Columns: **INPUT** (observed, white header), **GROUND TRUTH** (green header), **TRI-SENSOR model** (blue header), **SINGLE-SENSOR models** (red header).
  * The scene is a forward view of a two-lane road with oncoming lanes on the left: a yellow centre line, dashed white lanes, roadside poles, lead and oncoming cars as coloured blocks with tail/head lights, and pedestrians on the verge.
* **Right panel (330 px):** three boxes. (1) LiDAR error for this clip vs horizon (+0.25 s to +2 s): blue = tri model, red = LiDAR-only, grey = persistence. (2) RGB error bars: tri vs RGB-only, labelled "the price of joint training". (3) Held-out table.
* **Bottom strip:** legend and disclaimer.

**INPUT:** 6 observed frames (1.5 s at 4 Hz) play in real time with green boxes on the true cars and pedestrians. The frame then freezes at t = 0.

**GROUND TRUTH (green):** the real next 8 frames (2 s), with green boxes and dotted green tracks marking where each car and pedestrian actually goes.

**PREDICTION:**
* **Blue column, tri model:** a single network forecasts all three tiles at once from one tiled canvas.
* **Red column:** three separate single-sensor models.
* **Cyan boxes:** hot blobs found in each model's forecast thermal frame by a simple threshold detector, a quick check of whether the forecast still contains the people and cars.
* **Look and feel:** forecasts should look slightly soft and blurry, as deterministic L1 forecasts do. Do not sharpen them.

**Acts per scene (about 6.4 s):**
1. **OBSERVE (1.8 s).** The input column plays. The other columns are dark and read "waiting for 1.5 s of input".
2. **FORECAST (2.4 s).** The clock runs from +0.25 to +2.00 s. Truth and both forecasts advance frame by frame, and the error curves draw themselves point by point.
3. **HOLD (2 s).** Final numbers.

**Scenes:**
* **Scene 1, fog, held-out clip #494 (MEDIAN tri-LiDAR error among fog clips, not the best).**
  * A lead car closes in through grey fog. In the truth it grows in the frame and its hot exhaust band brightens.
  * Both forecasts keep it small and faint: neither model predicts the closing gap well.
  * LiDAR error at +2 s, on returns measured in both frames: tri 0.23 m, LiDAR-only 0.21 m, persistence 0.18 m. Persistence wins on this clip; leave that visible.
  * RGB: tri 6.6 vs RGB-only 6.1 grey levels.
* **Scene 2, snow, held-out clip #591 (MEDIAN among snow clips).**
  * White ground, snowflake speckle, and spurious near LiDAR returns scattered above the horizon.
  * A close red car leaves the view and a purple car sits ahead. The thermal-only forecast keeps a hot blob on the far car (cyan box). The tri model's thermal forecast washes it out.
  * LiDAR at +2 s: tri 0.67, LiDAR-only 0.67, persistence 0.81 m. RGB: tri 11.5 vs RGB-only 8.9.
* **Scene 3, the same snow clip with the tri model's RGB INPUT BLANKED.**
  * The tri RGB tile is black with "BLANKED INPUT" in red.
  * The tri model's LiDAR error barely moves (0.67 → 0.68 m), showing that this toy model barely uses the camera for LiDAR.

**Held-out table (right panel 3):** "held-out 600 clips, error at +2 s", columns LiDAR m / object m / RGB / thermal:
* tri-sensor: 0.46 / 5.25 / 8.5 / 4.6
* LiDAR-only: 0.46 / 5.23 / - / -
* RGB-only: - / - / 7.7 / -
* thermal-only: - / - / - / 4.3
* persistence: 1.12 / 9.76 / 9.8 / 6.2

Footnote: "persistence LiDAR = last frame, on returns measured in both frames" (the 1.12 m is on all returns, where a missing return counts as 0).

**Bottom strip:** "green boxes + dots = true cars/pedestrians and their future track | cyan boxes = hot blobs detected in the model's forecast thermal frame | LiDAR tiles: 16 beams projected into the RGB camera | the model sees nothing after t = 0". Always show: "SYNTHETIC 3-sensor driving simulator, my re-implementation from the abstract of arXiv 2611.22669 - not DSERT-RoLL, not the authors' model or code".

End card with the GitHub link.

**Honesty note for the voice-over:**
* Both clips are median clips, not cherry-picked wins, and the table is the evidence.
* Post-training clearly beats persistence for every sensor.
* The paper's headline result did not reproduce here: joint training gave no LiDAR gain (0.46 vs 0.46 m), and it cost the cameras about 10%.
* Blanking a sensor did not hurt more in snow than in fog (1 of 6 pairs).
* This is a toy rebuilt from the abstract, trained for 1,000 steps on CPU.

**Run it yourself:** `python simulate.py --out assets/trimodal_simulation.gif --mp4 assets/trimodal_simulation.mp4`
