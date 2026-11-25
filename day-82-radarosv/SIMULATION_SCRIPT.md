# Simulation Script / Prompt (for an image/video AI or rendering engine)

**Title card (1.5 s):** "Day 82 - RadarOSV: one radar sees detail, the other sees speed. Imaging radar alone vs imaging
+ single-chip radar with velocity-candidate selection. SYNTHETIC demo."

**Layout (16:9, dark UI, 1280x720, 10 fps), two rows of four 304 x 304 panels:**

**Top row:**
* **A. INPUT 1: imaging radar (aliases > 2.5 m/s).** A 120° radar fan seen from above, ego car at the bottom tip,
  forward is up, 25.6 m deep. Brightness = strongest return; colour = its Doppler on a ±12 m/s scale (blue
  approaching, red receding, white ≈ 0). Walls and cars are sharp, but every colour stays pale: this radar can only
  report -2.5 ... +2.5 m/s, so a car closing at 7 m/s shows up as if it moved at -2 m/s.
* **B. INPUT 2: single-chip radar (±24 m/s).** The same fan, but blocky (7.5° beams, 0.8 m range cells) and patchy
  (40% of returns missing). The colours are strong and correct: deep blue for fast oncoming traffic.
* **C. BASELINE (red frame): imaging radar + external ego speed.** Bird's-eye 64 x 64 grid (0.4 m cells) with the
  predicted class colour per cell: grey static, orange car, dark orange truck, pink cyclist, yellow pedestrian.
* **D. RadarOSV-style (blue frame): both radars, ego speed estimated from radar.** Same view.
* In C and D, every TRUE moving object has a thin white box. From its centre come two arrows along the line of sight:
  green = true radial velocity, red/blue = the network's prediction (mean over the object's cells). The label reads
  "gt / pred" in m/s, e.g. "+5.6/+2.6" in the baseline and "+5.6/+5.6" in RadarOSV. Footer: "dynamic-cell vel MAE x.xx m/s".

**Bottom row:**
* **E. GROUND TRUTH (green frame):** true class per cell, white boxes, green radial-velocity arrows. Footer: true ego speed.
* **F. Live graph:** "radial-velocity MAE on dynamic cells (m/s)", red = baseline, blue = RadarOSV, drawn left to right
  over the 6 s scene, y-axis 0-8.
* **G. Live graph:** "ego speed (m/s): truth vs radar estimate", solid green truth and a dashed blue estimate, 0-12.
* **H. Held-out table:** "held-out 1500 synthetic scenes", two columns (baseline red, RadarOSV blue):
  * occ mIoU ≤12.8 m: 0.80 / 0.80
  * dyn class acc: 0.76 / 0.77
  * vel MAE all (m/s): 4.01 / 0.59
  * aliased cells: 4.24 / 0.60
  * unaliased cells: 3.37 / 0.57
  * ego speed MAE: ext. / 0.13
  * Footnote: "baseline is GIVEN the true ego speed; RadarOSV estimates it from radar".
* **Bottom strip:** colour legend, then always: "SYNTHETIC radar scenes, my re-implementation from the abstract of arXiv
  2611.30078 - not the authors' model, data or radars".

**INPUT vs GROUND TRUTH vs PREDICTION:** the input is panels A and B (raw polar spectra). The ground truth is panel E
plus the green arrows and white boxes. The predictions are the cell colours and red/blue arrows in C and D. Both networks
run live on every frame. The world moves: the ego drives forward at a fixed speed, walls slide down the screen, road
users move and respawn.

**Scenes (6 s each, rules fixed before rendering; a pool of 120 candidate scenes was scored over its first 0.8 s):**
1. **RANDOM (seed 82).** Ego at 8.0 m/s, two trucks and two cars. Almost every return is aliased in the imaging radar.
   Baseline arrows point the wrong way or are too short (scene average 4.56 m/s error); RadarOSV's arrows match the green
   ones (0.71 m/s). Its ego estimate wobbles about 0.45 m/s below truth: keep the dashed line visible.
2. **SELECTED win (seed 1088).** Median of the 107 pool scenes where RadarOSV is better by more than 1 m/s. Ego 4.5 m/s,
   a cyclist riding away at +5.6 m/s: baseline says +2.6, RadarOSV +5.6. Scene average 3.00 vs 0.51 m/s. Around t = 2.5 s
   both error curves spike together: neither model is perfect.
3. **SELECTED loss (seed 1044).** Median of the only 2 pool scenes where RadarOSV was WORSE in the 0.8 s scoring
   window. The ego car is almost stopped (0.1 m/s) and only slow pedestrians move, so nothing aliases. At t = 0.3 s
   the baseline is at 0.08 m/s error and RadarOSV at 0.30. Over the full 6 s RadarOSV ends up ahead (0.29 vs 0.49 m/s).
   Both networks mislabel some pedestrians as cyclists. The point: unwrapping only pays off when things move fast.

**Honesty note for the voice-over:**
* Scene 1 is random. Scenes 2 and 3 are selected, one win and one loss; the table is the evidence.
* The velocity gain (4.01 → 0.59 m/s) is bigger than the paper's (1.03 → 0.51) because 73% of my moving cells exceed
  the imaging radar's ±2.5 m/s window. Where the gain lands depends on that share.
* The paper's big class-accuracy gain (63.1 → 75.5%) did NOT reproduce: 76.0 → 76.7%.
* Radar-estimated ego costs a little: the same network with true ego speed scores 0.51 vs 0.57 m/s (seed 0).
* Invented radars and scenes, 2D grid, no occlusion or multipath, 3,000 CPU steps, 2 seeds for the main models.

**Run it yourself:** `python simulate.py --out assets/radarosv_simulation.gif --mp4 assets/radarosv_simulation.mp4`
