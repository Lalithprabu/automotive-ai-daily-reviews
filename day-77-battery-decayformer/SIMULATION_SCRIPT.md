## Simulation Script / Prompt (for an image/video AI or rendering engine)

**Title card (1.5 s):** "Day 77 - A battery's whole life from its first 100 cycles. DecayFormer: selective recurrence + state-query attention + master-curve head. SYNTHETIC demo."

**Layout (16:9, dark UI, 1280x720, 10 fps):**
* **Top-left, CYCLER FEED (836x300):** the discharge voltage vs SOC curve of the current cycle in bright white (2.7-4.3 V, SOC 0→1), with every 10th earlier cycle faded grey behind it. Ten vertical SOC bins split the curve. Under each bin a blue bar shows that cycle's dQ/dV, with a white tick at its cycle-1 height, so the mid-SOC peaks can be seen flattening. On the right: the aging condition (temperature, charge C-rate, depth of discharge) with "(never seen in training)", the cycle counter, capacity %, resistance (mΩ) and temperature rise.
* **Bottom-left, SOH TRAJECTORY (836x300):** x = cycles 0-3000, y = SOH 0.76-1.0, with a horizontal EOL line at 0.8. A shaded band marks the observed window (cycles 0..k), which grows as cycles arrive.
* **Right column (430 px):** three stacked panels: (1) the master curve in normalised life u = n/L; (2) where the model looks (attention over cycles + retention gate); (3) the held-out study table.
* **Bottom strip:** the legend, the cell-selection rule and the disclaimer.

**Scene:** two test cells whose aging condition was held out of training. Cell A is knee-dominated (35 °C, 1C charge, 60% DoD; true EOL 1,645 cycles). Cell B has no strong knee (45 °C, 2C, 100% DoD; true EOL 984). Each is chosen as the MEDIAN-error cell of its group. Each cell plays in three acts:
1. **Cycles 1-100 (10 s):** one cycle per frame. From cycle 20 on, every 5 cycles all three forecasters re-predict the full trajectory from the cycles seen so far.
2. **Fast-forward (4 s):** the model input freezes at cycle 100. The counter runs on to the true EOL, the readout switches to "true SOH" in green, and the true curve is revealed in bright green against the frozen forecasts.
3. **Hold (2.2 s):** each forecaster's EOL error is printed in %.

**INPUT (grey):** measured capacity dots for cycles 1..k, drawn on top of everything, together with the live V(SOC) curve and the dQ/dV bars in the feed panel.

**GROUND TRUTH (green dashed):** the noise-free SOH trajectory down to 80%. It is dim during streaming and turns bright as the fast-forward reveals it. A green diamond marks the true EOL on the 0.8 line. In the master-curve panel, the truth is re-plotted against u = n / L_true.

**PREDICTION:**
* **Blue (thick): DecayFormer.** The master-curve forecast is drawn up to its own predicted EOL, with a blue triangle under the EOL line at L_hat.
* **Red: Transformer (680k params, free head).** It is drawn up to its first crossing of 0.8. It visibly wiggles upward: the non-monotone failure.
* **Yellow dotted: sqrt-law fit** to the observed capacity.
* Each forecaster gets a triangle at its predicted EOL. A text block lists "EOL 1950 (+19%)"-style readouts.

**Telemetry (right):**
1. "master curve: SOH vs normalised life u = n / L": the blue 16-knot curve, with dots at the knots, against the green truth. A vertical line marks u = 1 and a horizontal line marks 0.8. Caption: "16 knots, softmax steps → monotone by construction".
2. "where it looks: state-query attention over cycles": 100 blue bars (attention averaged over the 4 state queries) and a thin white line for the mean retention gate a_t.
3. "held-out study: 288 cells, 48 conditions, 6 folds", with params / traj MAPE / life MAPE: DecayFormer 120k 1.06% 16.8%; free head 126k 0.88% 33.0%; no SOC feats 119k 1.34% 20.5%; no attention 54k 0.98% 16.4%; Transformer 680k 0.88% 41.8%; GRU 183k 0.86% 36.1%; sqrt fit 0.99% 68.8%. Footer: "paper (BatteryLife Li-ion): 2.25% MAPE".

**Bottom strip:** "grey dots = measured capacity (input) | green dashed = true SOH | blue = DecayFormer | red = Transformer (5x params, free head) | yellow dots = sqrt-law fit | triangles = predicted EOL". It also reads: "cell A (knee-dominated): test cell #167, fold 4; picked as the MEDIAN-error cell of its group, not the best". Always show: "SYNTHETIC aging simulator, my re-implementation from the abstract of arXiv 2611.19116 - not BatteryLife, not the authors' model or code".

**Beats (about 34 s at 10 fps):**
* **(0-2 s) Cell A, cycles 1-20.** Grey dots fall fast through the SEI break-in. Text reads "waiting for 20 cycles before the first forecast".
* **(2-10 s) Cell A, cycles 20-100.** All three forecasts track the observed part. DecayFormer already bends down with a knee near 1,700 cycles (EOL about 1,800 at cycle 46). The red Transformer wiggles. The yellow sqrt fit runs straight and keeps getting more optimistic.
* **(10-14 s) Fast-forward.** The green truth bends into its knee and crosses 0.8 at cycle 1,645. Final readouts: DecayFormer 1,950 (+19%), Transformer 2,970 (+81%), sqrt fit 3,000+ (+82%). DecayFormer is late but saw the knee; the others missed it.
* **(16-26 s) Cell B, cycles 1-100.** A hot cell with deep cycling and no strong knee. All three curves overlap the truth.
* **(26-32 s) Fast-forward.** Truth EOL 984. DecayFormer 918 (-7%), Transformer 1,000 (+2%), sqrt fit 975 (-1%). Leave this on screen: on easy cells the simple baselines win, and DecayFormer's last knot drops in a small cliff just before EOL.

End card with the GitHub link.

**Honesty note for the voice-over:** the two cells are median-error cells, and the 288-cell table is the evidence. DecayFormer wins clearly on lifetime error (16.8% vs 33-42% for free heads), but it LOSES on the paper's own metric, trajectory MAPE (1.06% vs 0.86-0.88%). The attention block added nothing in this toy. The simulator is synthetic and was built so that the SOC-interval features carry knee information. Code is re-implemented from the abstract only.

**Run it yourself:** `python simulate.py --out assets/decayformer_simulation.gif --mp4 assets/decayformer_simulation.mp4`
