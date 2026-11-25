# Sourcing

**Paper:** "Beyond the Camera: A Tri-Modal Driving World Model for Future RGB, Thermal, and LiDAR Forecasting in Adverse
Weather" by Faysal Mehrab Chowdhury, Md. Mahfuzur Rahman, Ahmed Rafi Hasan, Kishor Datta Gupta, Mohd Ariful Haque and
Savannah Shannon, arXiv 2611.22669 (cs.CV), dated 18 Nov 2026 on the arXivSignals mirror. Please check the author list
against arxiv.org; this run could not open arxiv.org because of rate limiting.

**What I read:** only the abstract, as summarised on the arXivSignals mirror. I did not read the full text, the
figures or the authors' code (the mirror shows a "code released" badge).

**Taken from the abstract:**
* A pretrained video world model is post-trained for each sensor (RGB, thermal, LiDAR).
* LiDAR is represented as a depth video aligned with the RGB camera.
* A tri-sensor model sees all three sensors as one video.
* LiDAR error is measured in metres against measured returns, and forecasts are compared with persistence baselines
  run through the same tokenizer.
* Evaluation uses DSERT-RoLL: 58 validation sequences across six weather conditions.
* Reported findings: post-trained RGB and thermal forecasts beat persistence. Joint training lowers LiDAR error at 2 s
  by 2.1 m at a small cost in camera fidelity. Only the tri-sensor model's LiDAR beats persistence on mean error.
  Blanking one sensor degrades the others more in snow than in fog.

**My construction (not from the paper):**
* The synthetic 3-sensor simulator, including its six weathers and its range, dropout and clutter rules.
* The model: conv tokenizer, ConvGRU, one canvas-attention block, residual detokenizer, about 0.43M parameters. The
  authors post-train a large pretrained video model instead.
* The tiling layout: side-by-side [RGB | thermal x3 | LiDAR (d, d, valid)].
* Pretraining on clear RGB only, all losses and weights, step counts, and the 6 + 8 frame window at 4 Hz.
* Persistence is the raw last frame, not run through a tokenizer.

None of the numbers in this repo are comparable to the paper's.
