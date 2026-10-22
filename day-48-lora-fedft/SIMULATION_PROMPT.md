# Simulation Script / Prompt (for a video or rendering AI; not part of the post body)

**Title:** "20 cars, one forecaster, 62 KB each"  **Length:** 18 s, 16:9, dark slate background, flat technical look, no photorealism. Caption on screen: "Synthetic data: illustrative reconstruction".

**Layout:** left 60% = three side-by-side top-down pedestrian scenes (curving walker, stop-and-go walker, sharp-turn crosser). Centre = radio panel. Right = telemetry chart.

**Input (0-3 s):** In each scene a grey dotted path (the observed 4 s) ends at a white dot. A round counter in the corner reads "Round 0 / 30". A red dashed line shoots straight ahead from the white dot (the frozen generic model, which only knows straight walking). A green line then draws the true 6 s future: it curves, stops short, or turns 80 degrees. The red line is visibly wrong. Show "ADE 2.6 m" in red.

**Federated rounds (3-15 s):** 20 small white car dots form a ring around a gold server square in the centre panel. Each round, every car emits a tiny blue square that travels along a thin line to the server (blue square area = 5.7% of a purple reference square labelled "full-FT packet, 1,080 KB"; the blue one is labelled "LoRA, 62 KB"). The server flashes once, the round counter ticks up, and in the left scenes a solid blue path (the LoRA-FedFT prediction) redraws, rotating and shortening toward the green truth. By round 7 the curving and stop-and-go blue paths overlap the green; the crossing scene improves more slowly and stays visibly off, which must be shown honestly.

**Telemetry (all along):** right chart, x = cumulative uplink MB on a log axis, y = validation ADE in metres. A blue line (LoRA) drops steeply and reaches about 1.5 m near 9 MB; a purple line (full fine-tuning, same local epochs) follows the same shape but about 10x further right; a grey line (federated from scratch) lags. Dotted green line = centralized (0.9 m), dotted red = frozen base (2.6 m). Live counters: "LoRA 8.7 MB | full FT 86 MB".

**Ground truth vs prediction overlay:** green = ground truth, red dashed = frozen base, blue = LoRA-FedFT, grey = observed history. Keep one legend, top-left.

**End card (15-18 s):** freeze on round 30; show "same accuracy class, ~1/17 of the bytes per upload". Add the disclosure line: "Full FT was slightly more accurate at equal rounds; LoRA wins on bytes."
