## Simulation Script / Prompt (for an image/video AI or rendering engine)

**Title card (1.5 s):** "Day 78 - Every agent at the intersection, 20 futures each. SpecFlow: flow matching in the ACCELERATION SPECTRUM + endpoint-conditioned denoiser. SYNTHETIC demo."

**Layout (16:9, dark UI, 1280x720, 10 fps):**
* **Left, BEV (812x600):** a top-down 4-way intersection with 14 m-wide asphalt arms, dashed white lane lines, a yellow centre line, white stop lines and zebra crosswalks. Vehicles are drawn as oriented 4.6 x 1.9 m outline boxes with a nose dot; pedestrians are circles. Each agent carries a tag "#id car/ped NN km/h". The focus agent is outlined in pure white.
* **Right column (448 px):** three stacked panels. (1) Acceleration spectrum of the focus agent: 12 DCT coefficients c0..c11 as blue bars (10-90% range over the 20 hypotheses) with green truth ticks. (2) Telemetry: future speed (top) and signed acceleration (bottom) over 4.8 s, with ±6 m/s² dashed limit lines. (3) The held-out study table.
* **Bottom strip:** legend, scene-selection rule, disclaimer.

**INPUT (grey):** 8 measured positions per agent (3.2 s at 2.5 Hz) as grey dots joined by a thin trail. They stream in during act 1.

**GROUND TRUTH (green dashed):** the noise-free 12-step future (4.8 s). It is dim during denoising, then bright once the world plays forward. HOLD labels show minADE / minFDE per agent.

**PREDICTION:**
* **Blue, 20 thin lines per agent: SpecFlow** with top-1 endpoint conditioning.
* **Yellow star / diamonds: endpoint candidates.** Size scales with confidence; the star is the top-1 candidate that conditions the denoiser.
* **Red dashed (focus agent only): position-state flow**, i.e. the same network generating x,y directly.
* **Magenta (scene 2, focus agent): confidence-stratified endpoints.** The 20 hypotheses are spread over the K = 6 candidates in proportion to their scores.

**Acts per scene (about 12.5 s):**
1. **OBSERVE (3.2 s):** boxes move along their tracks and grey history dots appear.
2. **DENOISE (3 s):** the clock freezes at t = 0 and the caption reads "flow step k/10: noise -> acceleration spectrum -> integrate". The 20 blue fans start as smooth random sweeps (integrated noise) and tighten into lanes step by step. The spectrum bars shrink towards the green ticks.
3. **PLAY (4.8 s):** the caption reads "model input frozen at t = 0; world moves on along the ground truth". Boxes drive along the green truth while the fans stay fixed. A cursor sweeps the telemetry panel.
4. **HOLD (1.8 s):** per-agent minADE / minFDE labels. In scene 2 the focus agent also shows the stratified numbers.

**Scenes:**
* **Scene 1, test scene #1786 (MEDIAN-error scene, not the best):** five agents: a straight car, a left-turning car (focus #1), a right-turning car, another straight car and a waiting pedestrian. Beat: every blue fan for #1 runs straight west while the green truth bends left (minFDE 7.1 m). The straight car is almost perfect (0.02 m). The telemetry panel shows red position-flow acceleration lines zig-zagging past ±6 m/s² while the blue lines stay smooth on the green truth.
* **Scene 2, test scene #866 (chosen as a top-1 FAILURE case):** focus #2 is a right-turning car. Top-1 conditioning sends all 20 blue hypotheses straight north (minFDE 24.4 m). The magenta stratified fan splits: some go straight, some curve right and catch the turn (minFDE 1.75 m), and a few run backwards to junk candidates. Leave the junk visible.

**Telemetry table (right panel 3):** "held-out: 4000 scenes, 17429 agents, best-of-20", columns minADE / minFDE / jerk / |a|>6:
SpecFlow 2.04 5.43 1.6 0% | position state 2.55 6.67 5.1 6% | accel, no DCT 1.96 5.25 1.7 0% | no phase loss 1.95 5.20 1.7 0% | no endpoint 1.79 4.80 1.7 0% | stratified ep. 0.94 2.22 1.7 0% | const. velocity 3.67 9.72 | truth jerk 1.1. Footer: "paper (ETH/UCY avg): minADE 0.15 / minFDE 0.22".

**Bottom strip:** "boxes = tracked agents | grey dots = observed history (input) | green dashed = ground truth | blue = 20 SpecFlow hypotheses | stars/diamonds = endpoint candidates (star = top-1) | red dashed (focus agent) = position-state flow | magenta = confidence-stratified endpoints". Always show: "SYNTHETIC intersection simulator, my re-implementation from the abstract of arXiv 2611.26390 - not ETH/UCY/SDD/NBA, not the authors' model or code".

End card with the GitHub link.

**Honesty note for the voice-over:** scene 1 is a median scene and scene 2 is a deliberately chosen failure. The table is the evidence. Accelerations-plus-integration clearly beat positions (−0.53 m minADE, 3x less jerk, 0% vs 5.6% infeasible samples). But the DCT itself added nothing over time-domain acceleration. Top-1 endpoint conditioning collapsed best-of-20 diversity, and spreading hypotheses over the scored candidates fixed it (2.04 → 0.94 m). Everything is re-implemented from the abstract on a synthetic intersection.

**Run it yourself:** `python simulate.py --out assets/specflow_simulation.gif --mp4 assets/specflow_simulation.mp4`
