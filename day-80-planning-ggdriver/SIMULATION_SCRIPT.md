## Simulation Script / Prompt (for an image/video AI or rendering engine)

**Title card (1.5 s):** "Day 80 - Glance at what is shared, gaze at what differs. Every candidate trajectory picks its own camera patches before it is scored. SYNTHETIC demo."

**Layout (16:9, dark UI, 1280x720, 10 fps):**
* **Top: two forward-camera feeds side by side (600 x 360 each, pixel-art upscaling of a 160 x 96 camera).**
  * Left, red frame: **GLANCE-ONLY planner (old way)**.
  * Right, blue frame: **GAZE planner (GGDriver-style)**.
  * The scene: a straight 2-4 lane road with dashed white lane lines, solid edge lines, green verges and a pale sky. Static objects: stalled cars (coloured blocks with a dark rear window and red tail lights), orange cones with a white stripe, low brown road debris, red barrels, pedestrians on the verge. Small hazards sit 22-62 m ahead, so they are only a few pixels tall.
  * A black label in each feed's top-left corner reads the pick: "pick: lateral +0.88 m, target 13.9 m/s, true score 0.89".
* **Bottom-left (300 x 262): bird's-eye view** of the same static scene. Road and lane lines are grey, objects are coloured rectangles, and the two picked paths are drawn in red and blue with the ego box moving along each.
* **Bottom-centre (470 x 262): "8 refined proposals: predicted score (bar) vs true rule score (|)".** Two columns (glance-only in red, gaze in blue). Each row is a bar of predicted score with a green tick at the true score. The picked row gets a white outline and a ">".
* **Bottom-right (438 x 262): held-out table** (see below).
* **Bottom strip:** legend and disclaimer.

**INPUT:** a single camera frame plus the ego speed. There is no history and no perception labels. Green boxes on the true objects are drawn for the viewer only; the planners never see them.

**GROUND TRUTH (green):** the true object boxes, plus a dotted green path showing the best trajectory under the rule score (the best refined (lateral offset, speed) in the vocabulary).

**PREDICTION:**
* **Red:** the glance-only pick, and its 8 hypotheses as thin grey lines.
* **Blue:** the gaze pick, with corridor edge lines at the ego width.
* **Cyan cells:** the camera patches inside the candidate's projected 3-D corridor (about 17% of the 240 patches). These are the only patches that candidate's scorer reads.

**Acts per scene (about 10 s):**
1. **GLANCE (1.5 s).** Both feeds show their 8 hypotheses as thin grey lines fanning out from the ego, labelled "glance: 8 hypotheses from shared scene features".
2. **GAZE (2.4 s).**
   * Right feed: the 8 candidates are shown one at a time. Each shows its cyan corridor cells and its refined path in cyan, captioned "candidate 5/8 gazes at 43 of 240 patches -> predicted score 0.34".
   * Left feed: dims slightly, captioned "all 8 candidates scored on the SAME 9 shared tokens".
3. **SCORE (1 s).** Each feed draws its pick as a thick line with the dotted-green best path. The bars panel outlines each planner's chosen row.
4. **EXECUTE (4 s, real time).** Each camera drives its own planner's path through the static scene, and the view re-renders from the moving ego pose. On a collision, that feed flashes red with "COLLISION at t = 3.10 s" and freezes. The ego box in the BEV turns solid red.
5. **HOLD (1.2 s).**

**Scenes (rules fixed before rendering):**
* **Scene 1: a RANDOM small-hazard scene (#1351, seed 80).**
  * Ego at 10.0 m/s. A small hazard sits in the ego lane and stalled cars sit in the lane to the left.
  * Glance-only picks lateral -2.63 m at 14 m/s and hits the car at t = 3.10 s (score 0).
  * Gaze picks a two-lane move left at 10 m/s, with no collision (score 0.66).
* **Scene 2: SELECTED win (#1109).** A scene where glance-only crashes and gaze does not; 161 of 1,500 scenes are like this, and this is the median gaze score among them.
  * Ego at 4.3 m/s. An orange barrel or cone sits in the ego lane.
  * Glance keeps the lane at 6 m/s and hits it at t = 3.55 s.
  * Gaze nudges +0.88 m and speeds to 13.9 m/s, passing beside it (score 0.89).
* **Scene 3: SELECTED loss (#194).** A scene where gaze crashes and glance does not; there are 14 such scenes.
  * Ego at 6.8 m/s with cones near the ego lane.
  * Gaze nudges -1.38 m at 14 m/s into a cone at t = 1.70 s.
  * Glance keeps the lane at 6 m/s (score 0.81).
  * Leave this failure fully visible.

**Held-out table (bottom-right):** "held-out 1500 scenes", columns score / crash / small-hazard score:
* keep lane at current speed: 0.490 / 31.7% / 0.384
* glance only: 0.564 / 27.5% / 0.568
* gaze, refine only: 0.851 / 2.9% / 0.832
* gaze, score only: 0.861 / 2.5% / 0.851
* gaze, refine + score: 0.869 / 2.2% / 0.853
* random patches (matched count): 0.649 / 18.9% / 0.630
* all 240 patches: 0.841 / 2.5% / 0.818
* oracle (best refined): 0.916 / - / 0.914

Footnote: "score = NC x DAC x (0.75 progress + 0.25 comfort)".

**Bottom strip:** "green boxes = true objects | dotted green = best trajectory under the rule score | cyan cells = patches inside the candidate's projected corridor | camera only, no perception labels". Always show: "SYNTHETIC driving scenes, my re-implementation from the abstract of arXiv 2611.27330 - not NAVSIM / WOD-E2E / Bench2Drive, not the authors' model or code".

End card with the GitHub link.

**Honesty note for the voice-over:**
* Scene 1 is random. Scenes 2 and 3 are selected, one win and one loss, and the table is the evidence.
* Most of the gain comes from letting the planner read native-resolution patches at all: all-patch attention reaches 0.84. The path-aligned mask adds +0.027 on top, and it clearly beats a random mask of the same size.
* The gaze model's own glance head scores 0.85 from shared tokens alone. So the compressed summary can hold the hazard; the glance-only model simply failed to learn it.
* Toy rebuilt from the abstract, 2,000 CPU steps, 1 seed (2 for glance-only).

**Run it yourself:** `python simulate.py --out assets/ggdriver_simulation.gif --mp4 assets/ggdriver_simulation.mp4`
