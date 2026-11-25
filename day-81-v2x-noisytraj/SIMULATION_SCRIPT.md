# Simulation Script / Prompt (for an image/video AI or rendering engine)

**Title card (1.5 s):** "Day 81 - Trajectory prediction under sensing and V2X noise. Same Transformer, trained clean vs
trained on noise-augmented histories. SYNTHETIC demo."

**Layout (16:9, dark UI, 1280x720, 10 fps):**
* **Top and middle: two stacked bird's-eye panels of the same 3-lane highway (1240 x 205 each).** Top one has a red
  frame and is labelled **CLEAN-TRAINED predictor (curated, noise-free training data)**. Middle one has a blue frame
  and is labelled **NOISE-AUGMENTED RETRAIN (same Transformer, trained on noisy histories)**.
  * Road: dark asphalt, dashed white lane lines, solid edge lines. The lateral axis is stretched x1.6 so lane changes
    are visible. The view follows a "focus" vehicle, about 130 m of road wide.
  * 8 vehicles, drawn as white outline boxes at their TRUE position. The focus vehicle has a thicker box and a
    "focus" tag.
  * Top-right corner of each panel: "focus agent minADE6 3.10 m".
* **Bottom-left (420 x 214): "held-out minADE6 (m) vs noise level"** with four lines: clean (red), +Kalman (pink),
  +learned denoiser (purple), noise-aug (blue). A white vertical cursor marks the current noise level.
* **Bottom-centre (400 x 214): "this frame: minADE6 (m)"**, four live bars: focus clean-trained, focus noise-aug,
  all 8 clean-trained, all 8 noise-aug. Below: "t0 position error of the input: 1.94 m".
* **Bottom-right (396 x 214): held-out table** (below).
* **Bottom strip:** legend and disclaimer.

**INPUT (identical in both panels):** 2 s of observed history per vehicle at 10 Hz.
* Grey dots: sensor-tracked vehicles, with correlated jitter.
* Orange dots: V2X-received vehicles, with a slowly drifting GNSS-like offset.
* Hollow dots: a lost V2X packet, so the previous value is held (stale). The last dot is larger.
* A new noise draw every frame, so the dots shimmer more as the level rises.

**GROUND TRUTH (green):** a dashed green line for each vehicle's true 3 s future, thicker for the focus vehicle. A
faint grey line shows the true (clean) past.

**PREDICTION:** 6 thin lines per vehicle (the 6 modes) plus one thick line (the top-scoring mode), red in the top panel
and blue in the middle one. Every line starts from the vehicle's last OBSERVED position, so a biased or stale
observation visibly displaces the whole forecast.

**Acts per scene (about 11 s):**
1. **LEVEL 0 (2 s).** Clean input. Both panels look almost the same: forecasts hug the green dashes.
2. **LEVEL 1 (2 s).** The dots start to jitter. Red forecasts begin to fan out sideways; blue ones barely move.
3. **LEVEL 2 (2 s).** Orange dots drift and some go hollow. The red top modes swing across lanes from frame to frame.
4. **LEVEL 4 (2 s).** Heavy noise (about 1.9 m mean error in the latest position). Red forecasts shoot diagonally across
   the road. Blue forecasts stay in lane, offset only by the input error.
5. **REVEAL (3 s).** Predictions freeze; the white boxes drive along the true future so viewers can see who was right.
   Header: "REVEAL t = +1.6 s (predictions frozen)".

**Scenes (rules fixed before rendering):**
* **Scene 1: RANDOM (#172, seed 81).** In a typical level-4 frame the noise-aug model wins over all 8 vehicles (1.03 vs
  2.81 m) but loses on the focus vehicle (2.04 vs 1.85 m). Keep that visible. (Live numbers change with every noise draw.)
* **Scene 2: SELECTED win (#1345).** Median of the 2,725 scenes where noise-aug is better by more than 1 m at level 4.
  In a typical level-4 frame: focus vehicle 2.09 vs 3.10 m, all vehicles 1.00 vs 3.04 m.
* **Scene 3: SELECTED cost (#1919).** Median of the 96 scenes where noise-aug is WORSE than clean-trained by more than
  0.3 m on clean input (level 0). Show the level-0 frames slowly: this is the price of robustness. At level 4 it still
  wins (1.10 vs 2.60 m).

**Held-out table (bottom-right):** "held-out 3000 scenes: minADE6 at level 0 / 2 / 4", plus degradation at level 4:
* constant velocity (raw): 0.89 / 2.88 / 5.18, 5.8x
* clean-trained: 0.25 / 1.65 / 2.83, 11.2x
* clean-trained + Kalman filter: 0.34 / 1.45 / 2.45, 9.7x
* clean-trained + learned denoiser: 0.42 / 0.87 / 1.60, 6.3x
* noise-augmented retrain: 0.33 / 0.62 / 1.07, 3.2x

Footnote: "degradation = error at level 4 / the same predictor on clean input".

**Bottom strip:** "grey = sensor-tracked history | orange = V2X-received | hollow = packet lost (stale value held) |
white box = TRUE position | green dashed = TRUE future | lateral scale x1.6". Always show: "SYNTHETIC highway traffic
+ synthetic noise model, my re-implementation from the abstract of arXiv 2611.28128 - not the authors' models, data or
noise".

End card with the GitHub link.

**Honesty note for the voice-over:**
* Scene 1 is random. Scenes 2 and 3 are selected, one win and one cost, and the table is the evidence.
* Noise-augmented retraining never lost to the clean-trained model at level 4 in any of the 3,000 scenes, but it is
  30% worse on clean input (0.33 vs 0.25 m).
* The paper's "below 1.5x" held in my toy only up to level 1 (1.32x); at level 4 it was 3.25x.
* The learned denoiser is a strong second (1.60 m at level 4), far better than a Kalman filter (2.46 m).
* One architecture, synthetic data, an invented noise model, 5,000 CPU steps, 2 seeds.

**Run it yourself:** `python simulate.py --out assets/noisytraj_simulation.gif --mp4 assets/noisytraj_simulation.mp4`
