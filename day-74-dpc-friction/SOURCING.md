# What comes from the paper, what is mine

Paper: *Data-Driven Predictive Control for Autonomous Vehicle Racing under Varying Friction*,
Julius Beerwerth, Julius Tischbein, Bassam Alrifaee. arXiv [2611.21378](https://arxiv.org/abs/2611.21378) (eess.SY, v1 16 Nov 2026).
I could read the abstract only (via arXiv listings, arXivSignals and alphaXiv); the arXiv abstract and HTML pages
were rate-limited all four times I tried. The authors report released code, but I found no repository URL,
so none of this repo comes from their code.

**From the abstract:** DPC predicts future outputs by combining recorded input/output trajectories so they match a
measured initial trajectory and track a reference. Friction varies across track sections and tyre sets, and DPC
prediction gets worse when the data and the initial trajectory come from different friction levels. Instead of
estimating friction, the authors pool data recorded at different frictions. They compare matched, mismatched and
pooled data in simulation and on small-scale CPM Lab vehicles, with NMPC as the baseline. Their numbers:
pooled multi-step RMS prediction error within 1.5x of matched, mismatched 2.2x-3.4x; simulation tracking
13.7 cm (pooled DPC) vs 13.3 cm (NMPC); CPM Lab low-friction 1.6 vs 1.5 cm, high-friction 0.9 vs 1.1 cm.

**Mine (everything else):** the 1:10 single-track car with a simplified Pacejka tyre (mu scales peak force AND
cornering stiffness), the track and its wet sectors, the speed profile, the signals (u = steering + reference
curvature, y = lateral and heading error, shifted pairing), T_ini = 5 / N = 15, the l2 (ridge) regularisation and
its weights, the "mosaic" column-wise concatenation as my reading of "pool", the condensed steering QP, the
data-collection law, and the baseline. **The baseline is NOT NMPC**: it is a linear MPC on the single-track error
model with the TRUE local friction and true state (an oracle), which is a stronger-informed but linearised controller.
"Matched (oracle)" DPC switches data sets by the true friction under the car, which a real car cannot do.
