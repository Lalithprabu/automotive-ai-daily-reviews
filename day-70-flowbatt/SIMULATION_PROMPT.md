# Simulation Script / Prompt (for an image/video AI or rendering engine)

**Title card (2 s):** "Day 70 - FlowBatt: a battery's future as a distribution. SYNTHETIC demo."

**Layout (16:9, dark lab-HUD, 1280x720):** top band = three panels (420x300 each); bottom-left = main SOH chart (640x400); bottom-right = telemetry stack (620x400).

**Panel 1 - INPUT:** discharge-capacity-vs-voltage curves Q(V) for cycles 1→100 drawn one by one, coloured viridis by cycle number (purple = cycle 1, yellow = cycle 100). Inset bottom-right: a magenta ΔQ(V) = Q_n − Q_1 curve whose dip deepens as cycles accumulate. Caption: cell ID, chemistry (LFP / NMC / NCA), latest prefix SOH.

**Panel 2 - GENERATE (the flow-matching moment):** 24 of 100 sample trajectories on SOH-vs-cycle axes (0-2550 cycles, SOH 0.5-1.05). At flow time t = 0 they are pure jagged noise filling the panel. Over 16 frames (Euler steps 0→32) they straighten along near-linear paths into smooth fading curves with a knee. Their colour shifts from dull gold to bright cyan as t → 1. Counter: "flow time t = 0.38 | Euler step 12/32". Red dashed line at SOH 0.80 (end of life).

**Panel 3 - FLEET RUL BOARD:** 8 rows of cells. Per row: cyan bar = 5-95 % range of sampled end-of-life cycles, cyan dot = median, orange dot = deterministic Transformer, green tick = true EOL (appears once revealed). The active cell row is highlighted.

**Main chart - INPUT vs GROUND TRUTH vs PREDICTION:** white thick segment = observed SOH prefix (cycles 0-90). Translucent cyan band = FlowBatt 5-95 %; solid cyan = median; orange dashed = Transformer point forecast. Down-triangles mark each model's predicted EOL on the 80 % line. ACT 3: a vertical cursor sweeps right and draws the green GROUND TRUTH curve behind it; a green diamond lands where truth crosses 80 %. Footer: "EOL true 894 | FlowBatt 828 [637-1026] | Transformer 766".

**Telemetry:** (1) |SOH error| at the cursor, FlowBatt median (cyan) vs Transformer (orange), with running means. (2) Histogram of the 100 sampled EOL cycles (cyan bars), orange line = Transformer, green line = truth once reached. (3) "truth inside 5-95 % band: 83.9 % of revealed points (nominal 90 %)". (4) "hidden defect z = −2.36 (NOT visible in input)".

**Beats (15 s, three cells × 5 s):** cell 1 typical NMC/LFP: band brackets truth, the Transformer lands early. Cell 2: hidden-defect NMC that fails at ~440 cycles; narrow band and both models close. Cell 3: long-life LFP (~2,000 cycles); wide band, the median runs late and truth sits near the band's lower edge. Linger here: "uncertainty is the product". End card with the GitHub link.

**Always on screen:** "SYNTHETIC toy cell fleet - not MATR/HUST/MIX, not the authors' code".

**Honesty note for the voice-over:** on 320 synthetic test cells, FlowBatt has 3.70 %-pt trajectory RMSE vs 4.55 for a deterministic Transformer with the same trunk (seed 1: 3.98 vs 4.50). Coverage is 82 % against a nominal 90 %. A DDPM with the same backbone ties it (3.65) and is better calibrated; FlowBatt is 2.6× faster. The paper's numbers (< 2.5 % on five real benchmarks; MIX-20 0.88 vs 1.15) are on real data and not reproduced here. On the three featured cells the Transformer's running error is actually lower (1.49 vs 1.62): the clip is about the band, not a cherry-picked win.

**Run it yourself:** `python simulate.py --out assets/flowbatt_simulation.gif --mp4 assets/flowbatt_simulation.mp4`
