## Simulation Script / Prompt (for an image/video AI or rendering engine)

**Title card (1.5 s):** "Day 74 - Data-driven predictive control under varying friction: one Hankel matrix of dry + wet driving data, no friction estimator. SYNTHETIC demo."

**Layout (16:9, dark UI, 1280x720):**
* **Top-left (420x250):** track overview. A closed, rounded circuit (about 21 m lap, 1 m wide) drawn as a thick grey ribbon, with two sectors (15-45% and 65-85% of the lap) tinted light blue and labelled "wet, mu 0.5". Three small coloured dots mark the cars, and a white highlighted stretch marks the zoom window.
* **Main panel (836x330), below it:** an "unrolled chase view" in track coordinates. The horizontal axis is distance along the track, from 1.5 m behind the car to 3 m ahead. The vertical axis is lateral offset from the racing line, exaggerated x12, with grid lines at ±5 and ±10 cm. The racing line is a white dashed horizontal line. Wet stretches get a dark navy background tint, and the top-right reads "mu here = 0.5 (WET)" in light blue.
* **Right column (430 px):** three telemetry panels.
* **Bottom strip:** legend plus study numbers.

**Scene:** three 1:10-scale race cars on the same held-out seed, all driven by data-driven predictive control (DPC). Orange uses data recorded on dry tyres only. Magenta uses POOLED dry + wet data. Cyan uses matched data (an oracle that switches to the data set for the friction under the car). Speed follows a fixed profile (1.4-3.0 m/s). Control runs at 20 Hz; the prediction horizon is 0.75 s (15 steps).

**Main panel - INPUT:** five grey dots just behind the magenta car: the last 5 measured lateral offsets, i.e. the "initial trajectory" DPC matches.

**Main panel - GROUND TRUTH:** a thick green dashed line from the magenta car forward, showing where it REALLY goes over the next 0.75 s (revealed in hindsight).

**Main panel - PREDICTION:** from the SAME 5 samples and the SAME planned steering, three forecast polylines with dots at 0.25 / 0.5 / 0.75 s: cyan thin = matched-data predictor, orange thin = dry-only predictor, magenta thick = pooled predictor. In dry sectors all three overlap. In wet sectors the orange line bends away from the green truth by several cm while magenta stays near cyan.

**Main panel - TRACES:** solid lines behind each car show the path it actually drove. A filled circle marks each car's current offset. The orange car swings wider on entry to the wet sectors.

**Telemetry (right):**
1. "Live prediction error per step (cm)": 15 groups of three thin bars (orange / magenta / cyan) from 0 to 0.75 s, capped at 8 cm. Orange bars grow tall in wet corners.
2. "|lateral error| over time (cm)": three lines with a live RMS readout per car, and blue ticks along the bottom marking wet sectors. The orange line spikes on wet entries.
3. "Steering command (rad)": three overlaid steering traces.

**Bottom strip:** "prediction RMS (20 held-out seeds): on mu 0.5 matched 1.79 cm | dry-only 4.82 cm (2.7x) | pooled 2.40 cm (1.34x)" and "mixed track tracking RMS: dry-only 1.52 | wet-only 1.77 | pooled 1.15 | matched 0.73 | model MPC (true mu) 0.43 cm". Always show: "SYNTHETIC 1:10 toy world, my re-implementation from the abstract of arXiv 2611.21378 - not the authors' simulator, CPM Lab cars or code".

**Beats (about 18 s at 20 fps; times are video time):**
* (0-1.5 s) Title card.
* (1.5-3 s) Dry start: all forecasts overlap the truth, and the three cars are within a centimetre of each other.
* (3-5.3 s) First wet sector: the orange forecast fans away from the green truth, its error bars climb to 5-8 cm at the horizon end, and the orange car's |e_y| spikes. Magenta tracks cyan.
* (5.3-7.5 s) Dry again: all three converge.
* (7.5-9 s) Second, shorter wet sector: the same pattern, smaller.
* (9-11.7 s) Dry; lap 2 begins around 10 s.
* (11.7-14.1 s) First wet sector, lap 2: the pattern repeats, which shows it is systematic rather than a one-off.
* (14.1-18 s) Dry, then the start of the second wet sector. The RMS readouts settle near orange 1.5 / magenta 0.9 / cyan 0.8 cm for this seed.

End card with the GitHub link.

**Honesty note for the voice-over:** this seed (100) is simply the first held-out seed, not a picked best case. Over 20 held-out seeds, pooled beat both single-friction data sets on the mixed track every time, but it is still 1.5-1.6x worse than matched data on uniform tracks. The pooled predictor averages the two data sets rather than detecting friction (|g| splits 52/48 dry vs wet either way). The baseline here is an oracle linear MPC with the true friction, not the paper's NMPC, and it beat every DPC variant on the dry and mixed tracks.

**Run it yourself:** `python simulate.py --out assets/dpc_simulation.gif --mp4 assets/dpc_simulation.mp4`
