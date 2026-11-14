## Simulation Script / Prompt (for an image/video AI or rendering engine)

**Title card (2 s):** "Day 71 - ForeseeDrive: let the predicted future shape the plans, not just pick one. SYNTHETIC demo."

**Layout (16:9, dark UI, 1280x720):** the top band (1280x350) is a bird's-eye view of a straight 2-lane street. The bottom has three panels (each about 420x360): future-token strip, telemetry, and score tally.

**Scene:** a city street, ego in the right lane at 6-12 m/s, a 4-second horizon at 12 fps. There are five vignettes: (1) the lead car in the ego lane brakes hard at about 1 s with no warning; (2) a pedestrian at the kerb steps out (only a faint drift toward the road gives it away); (3) a left-lane car cuts in; (4) a second cut-in; (5) an honest failure where the predicted tokens miss and only logged-future tokens save it.

**BEV - INPUT:** grey dots behind every actor = 1.5 s of observed history. Actors are drawn at their present position as boxes (cars) or circles (pedestrians).

**BEV - GROUND TRUTH:** actors move along their *real* future in green. The expert plan is a green dashed line.

**BEV - PREDICTION:** 16 thin grey fans = base generator candidates; thick orange line plus a filled orange ghost car = the base plan it selects. 16 thin dark-cyan fans = the same frozen generator after future-token conditioning; thick cyan line plus a filled cyan ghost car = the ForeseeDrive plan. A magenta dashed line = ForeseeDrive fed logged-future tokens (diagnostic only). Translucent red cells = occupancy regions with high predicted boundary risk. Dots on the ego lane = route anchors, turning red when they are predicted to become occupied. A small tag over each ranked actor reads "A0 r 0.37 | true 1" (predicted risk vs teacher label). When a plan touches an actor, a red X stays at the contact point with "BASE collides at t = 2.9 s" (cyan text if ForeseeDrive collides).

**Token strip:** 36 vertical bars in four labelled groups (actor 4 | route 8 | occupancy 8 | margin 16, one per candidate). The top row is the teacher r and the bottom row is the predicted r: red intensity = risk, grey = suppressed (confidence < 0.5). Caption: "all tokens zero → u = 0 → output IDENTICAL to base (tested: max diff 0.0)".

**Telemetry:** (1) clearance to the nearest actor over time against the true future, one line each for base (orange), ForeseeDrive (cyan) and oracle tokens (magenta), with red lines at 0 m and 0.5 m. In win scenes the orange line dives below zero while cyan levels off. (2) Speed profiles for expert (green), base and ForeseeDrive.

**Tally:** running PDMS bars and collision counts over the vignettes shown. Below them, the full-split numbers: base 0.714 / 25.5%, ForeseeDrive (pred) 0.724 / 24.5%, shuffled tokens 0.695 / 27.1%, oracle tokens 0.900 / 5.7%.

**Beats:** (0-3 s) the lead car slams on its brakes and the base car rear-ends it, leaving a red X; cyan has already swung into the left lane. (3-6 s) the pedestrian steps out. The base collides, while cyan and magenta slow down and pass with clearance. (6-13 s) two cut-ins: the left-lane car merges; cyan is already braking. (13-16 s) the failure case: cyan and orange overlap and both hit the braking car, and only magenta (logged-future tokens) avoids it. Caption: "the future you can't see is the future you can't plan for". End card with the GitHub link. Always show: "SYNTHETIC toy world - not NAVSIM, not the authors' code".

**Honesty note for the voice-over:** the vignettes are picked by a fixed rule (first big win per event type, plus the first scene where only logged-future tokens avoid the crash). On all 3,000 synthetic test scenes, predicted tokens lift PDMS only from 0.714 to 0.724 (collisions 25.5% to 24.5%). Logged-future tokens reach 0.900 and 5.7%. The paper's numbers (NAVSIM 89.2 PDMS, nuScenes 0.41 m L2 / 0.05% collisions, Bench2Drive DS 76.50) come from real benchmarks and are not reproduced here.

**Run it yourself:** `python simulate.py --scenes 5 --out assets/foresee_simulation.gif --mp4 assets/foresee_simulation.mp4`
