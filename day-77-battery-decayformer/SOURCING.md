# Sourcing: what comes from the paper and what is mine

**Paper:** Shrey Patel, Bhargav Srinivas Rao, Samet Ayhan, "A Hybrid State Space-Attention Architecture for
Forecasting the Trajectories of Battery Degradation", arXiv 2611.19116 (listed 17 Nov 2026; cs.AI, physics.chem-ph).
The listing says code was released; I did not use it.

**Read:** the abstract only, through the arXivSignals mirror (arxiv.org was rate-limited during this run).

## From the abstract
* Task: early-life forecasting of the FULL SOH trajectory of a cell from its first cycles.
* Critique of prior SOTA: depends on model scaling, LLM-derived aging-condition embeddings and per-fold tuning.
* DecayFormer ingredients: cycle-level selective recurrence with a single scalar state per channel; attention whose
  queries are generated from the recurrent state; SOC-interval statistics as input features; a master-curve head that
  jointly estimates a scalar lifetime and a monotonically non-increasing SOH profile parameterised by normalised
  lifetime.
* Protocol: BatteryLife, 22 aging-condition-exclusive folds, one run per fold, one fixed configuration.
* Results (mean test MAPE): Li-ion 2.25%, CALB 2.76%, Na-ion 0.95%, Zn-ion 7.19%.

## My construction (not in the abstract)
* The whole data set: a synthetic aging simulator (SEI sqrt-law + linear LAM + plating knee triggered by a LAM
  threshold), 48 conditions (T x charge C-rate x DoD), 6 cells each, 100 early cycles, 25 per-cycle features.
  The SOC-interval statistics (dQ/dV and mean V in 10 SOC bins) are my stand-in, and the simulator is BUILT so that
  they reveal the LAM rate (and therefore knee timing) earlier than capacity does.
* 6 condition-exclusive folds instead of 22; one chemistry instead of four.
* All sizes and equations below the abstract level: width 64, 2 recurrent layers, gate parameterisation
  (softplus step, exp(-delta * exp(A_log)) retention), 4 state queries, 4 heads, 16-knot softmax master curve,
  lifetime prior log(1000), losses (trajectory MAPE + 0.025 * |log L error|), 1,500 steps, random observed prefix
  20-100 cycles during training (added so the clip can stream; the paper uses a fixed early window).
* Baselines: a 3-layer d=128 Transformer and a 2-layer GRU with an unconstrained head, a sqrt-law capacity fit,
  and three ablations (free head, no SOC features, no attention). No LLM-embedding baseline (out of scope on CPU).
