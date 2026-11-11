## Simulation Script / Prompt (for an image/video AI or rendering engine)

**Title card (2 s):** "Day 68 - LOWP: a driving planner that looks only where it plans. SYNTHETIC demo."

**Layout (16:9, dark UI, 1280x720):** top band = two rows of four camera tiles (320x160 each); bottom-left = bird's-eye view (640x400); bottom-right = three stacked telemetry graphs (640x400).

**Scene:** a two-lane rural road (grey asphalt, white edge lines, dashed yellow centre line, green verge, pale-blue sky) that alternates straights with left and right bends. The ego car drives the right lane at 5-15 m/s. One to three red box-shaped cars are ahead in the same lane, some slowing to a stop, so the ego must brake. 10 frames per second for 15 s.

**Row 1 - INPUT + "where LOWP looks":** four camera feeds (FRONT, FRONT-LEFT, FRONT-RIGHT, BACK). Over them, small orange-to-yellow dots: the 512 3D sample points (16 anchors x 32 candidates) projected into each image. Dot size and brightness = attention weight. The dots should cluster on the lead car and on the inside of the next bend. The BACK tile stays empty, because nothing behind the car matters for the plan.

**Row 2 - global ray reader:** the same four feeds with a translucent magma heat-map of cross-attention. It is broad and soft, brightest on the road ahead and the lane markings of the coming curve.

**BEV panel - GROUND TRUTH vs PREDICTION:** top-down, forward = up, 6 px per metre. Road ribbon, red lead-car rectangles, blue ego rectangle at the bottom centre. Overlay three 3-second trajectories with a dot every 0.5 s: green = ground truth, cyan = LOWP (mixed), orange = dense-BEV baseline. Yellow dots = the 16 anchor positions after refinement, strung along the road ahead. Bottom caption: time, command (STRAIGHT / LEFT / RIGHT), current alpha (0.5 / 0.8).

**Telemetry:** (1) L2 error at 3 s per frame, cyan LOWP vs orange dense; (2) ego speed (dips as the ego brakes behind a lead car); (3) CPU latency per frame, both models.

**Beats:** (0-4 s) a straight road; dots spread along the lane and both trajectories overlay the green one. (4-9 s) a left bend; the command flips to LEFT and alpha to 0.8, the dots slide into the curve, and the orange line drifts wide while cyan hugs green. (9-15 s) a slower lead car; the dots lock onto it, the speed trace drops and the trajectories shorten. End card with the GitHub link. Always show: "SYNTHETIC toy world - not nuScenes/NAVSIM, not the authors' code".

**Honesty note for the voice-over:** on the 1,500-frame synthetic test split (seed 0), avg L2 is 0.351 m for LOWP, 0.423 m for the dense baseline and 1.64 m for a camera-blind model. In the toy, LOWP is not smaller or faster than the dense baseline. The paper's numbers (0.296 m nuScenes L2, 53% params, ~1.7x speed) are not reproduced here.

**Run it yourself:** `python simulate.py --seed 5000 --frames 150 --out assets/lowp_simulation.gif --mp4 assets/lowp_simulation.mp4`
