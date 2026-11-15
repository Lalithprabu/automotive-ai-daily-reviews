## Simulation Script / Prompt (for an image/video AI or rendering engine)

**Title card (2 s):** "Day 72 - FloCOS: predict the space a car might take, not just one path. SYNTHETIC demo."

**Layout (16:9, dark UI, 1280x720):** the top band (1280x380) is a bird's-eye view of a straight 3-lane highway centred on one surrounding "target" car. Lateral scale is exaggerated about 2x so the lanes stay readable. The bottom has three panels (each about 420x330): telemetry, scene outcome, and running tally.

**Scene:** the target drives at 8-26 m/s in the middle lane. An ego car in an adjacent lane wants to merge into the target's lane within 3 s, either ahead of it or behind it. Horizon 3 s, 12 prediction steps, about 12 fps. Six vignettes: (1) the target brakes hard with no brake light; (2) a lane change left; (3) a lane change right; (4) a sudden acceleration; (5) another hard brake; (6) an honest failure where the FloCOS ego also collides.

**BEV - INPUT:** grey dots behind the target = 1 s of observed history; a thin grey box = its present footprint. Lane markings are solid at the road edges and dashed between lanes.

**BEV - GROUND TRUTH:** the target moves along its *real* future as a thick green box labelled "TARGET (truth)". A green dashed line shows the full 3 s path.

**BEV - PREDICTION:**
* **FloCOS:** at the current step, 6 translucent cyan octagons (each one a half-space convex polygon) whose union is the occupancy set. Behind them, faint dark-cyan outlines of all 12 steps form a "tube" that widens as uncertainty grows: long along the lane for braking or accelerating, wide across lanes for lane changes.
* **DetReg (single-footprint baseline):** one orange rectangle outline.
* **Physics reach set:** a grey box drawn with a thick outline. It quickly grows wider than the road, so add a corner label: "physics reach set @t: 485 m²".

**BEV - PLANNING:** three filled ego cars plan against the three predictions. Cyan-dark = FloCOS ego, orange-dark = DetReg ego, grey = reach-set ego, each tagged "ego|<method>". The DetReg ego merges confidently and drives into the braking target: a red X stays at the contact point with "DetReg ego COLLIDES at t = 1.75 s". The reach-set ego stays in its lane under the tag "STUCK (no feasible merge)". The FloCOS ego slips in behind the target, or ahead of it when the set allows.

**Telemetry:** (1) coverage of the TRUE footprint over time for FloCOS (cyan), DetReg (orange) and the reach set (grey), with a green 100% line; in brake scenes the orange line falls to 0 while cyan stays high longer. (2) Occupied area on a log scale with a "1 car" reference line: the grey line climbs to hundreds of m², cyan stays at tens, and orange sits at about 9 m².

**Outcome panel:** per method, one of "merge completed" (in its colour), "COLLISION" (red) or "stuck: never merged". It also shows the current-step area and coverage, plus the constraint written out: "max_h n_h·(p − c_i(t)) − s_ih(t) > margin for all i, t, p".

**Tally:** running bars of merges completed and collisions over the vignettes shown (end state: FloCOS 5 completed / 1 collision, DetReg 1 / 5, reach set 0 / 0). Below the bars, a full-split table (3,000 agents) with coverage / area / completion / collision:
* DetReg 0.572 / 9.0 / 0.812 / 0.157
* MTP-6 0.820 / 32.4 / 0.373 / 0.041
* Reach 1.000 / 489.3 / 0.136 / 0.015
* FloCOS 0.921 / 38.8 / 0.548 / 0.020

**Beats:**
* (0-4 s) Hard brake: orange crashes, grey is stuck, cyan merges behind.
* (4-12 s) The two lane changes: the cyan tube bends across the lane line before the car does.
* (12-16 s) The acceleration and the second brake.
* (16-20 s) The failure scene: the cyan tube runs ahead of a car that brakes harder than expected, coverage drops to 0.07, and the FloCOS ego collides too. Caption: "statistical coverage is not a guarantee".

End card with the GitHub link. Always show: "SYNTHETIC toy world - not WOMD, not the authors' code".

**Honesty note for the voice-over:** the vignettes are picked by a fixed rule: the first scene per intention where FloCOS completes and DetReg collides, then the first scene where FloCOS itself collides. Across all 3,000 test agents, FloCOS completes 55% of merges with 2.0% collisions. DetReg completes 81% but collides 15.7% of the time. An oracle that knows the true footprint completes 93%. The paper's claims (WOMD, real-vehicle tests) are not reproduced here, and I could not read its tables.

**Run it yourself:** `python simulate.py --scenes 6 --out assets/flocos_simulation.gif --mp4 assets/flocos_simulation.mp4`
