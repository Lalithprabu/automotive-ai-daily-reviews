# Sourcing: what comes from the paper and what I chose

**Paper:** *ForeseeDrive: Future-Token Conditioning for Anticipatory Generative Driving*
Shaokai Wu, Xunchu Zhou, Pinzhi Wang, Bo Dong, Yuting Hu, Yanbiao Ji, Qiuchang Li, Xiao Cui, Qi Kong, Hongtao Lu, Yue Ding.
arXiv [2611.14286](https://arxiv.org/abs/2611.14286) (cs.RO, cross-listed to cs.CV), v1 submitted 2026-11-10. The abstract says code ships in the supplementary material. I found no public repo.

Read: the arXiv abstract, the arXiv HTML (method sections, Table 1 NAVSIM), and the alphaXiv overview (diagnostics and limitations).
Not read: the nuScenes per-horizon table, the Bench2Drive table beyond DS 76.50 / SR 53.06, appendix sweeps, query counts, K, token dimensions, loss weights, and latency. My HTML fetches were truncated before those parts, and the paper may not report some of them.

## Paper facts used
| Item | Paper |
|---|---|
| Problem | A downstream scorer can only choose among the candidates it is given. A maneuver that was never generated cannot be recovered. |
| Tokens | Q = Q_a ∪ Q_r ∪ Q_o ∪ Q_m (actor, route, occupancy, TTC-margin). Each z_i = (e_i, c_i, r_i): continuous descriptor, validity/confidence label, boundary-risk label. Fixed query identities give pointwise teacher-student matching. Missing entries are null-padded and masked. |
| Teacher / student | The teacher builds tokens from the logged future; the predictor h_ψ(x_t) estimates them from history, map, route and tracked actors. Loss = masked Smooth-L1(e) + BCE(c) + BCE(r), with a stop-gradient teacher. |
| Suppression | Low-confidence or invalid tokens are dropped before injection. |
| Scene vector | u = Pool(Attn(E_q(Ẑ^f), E_x(x_t))), with invalid queries masked. |
| Geometry path | z̃_k^(l) = z_k^(l) + α_l·A_l(z_k^(l), u), where A_l is bias-free with A_l(z, 0) = 0 and α_l is zero-initialised. |
| Logit path | ℓ̃_k = ℓ_k + b_k(u), with b_k(0) = 0. |
| Preservation | L_preserve = Σ_k w_k·ρ(τ_f^k − τ_b^k). w_k is high for good base candidates and low for candidates tied to high-confidence interaction tokens. |
| Guard | L_guard = Σ_k 1[m_TTC(τ_b^k) > ε_b]·max(0, ε_g − m_TTC(τ_f^k)). The same clearance check also serves as an inference-time rejection rule. |
| Training | The pretrained generator is frozen and only the adapters are trained. On NAVSIM the backbone is DiffusionDrive. |
| Results | NAVSIM navtest PDMS 89.2 (NC 98.5, DAC 97.5, TTC 95.0, Comf 100, EP 83.5), vs DiffusionDrive 88.1 and SeerDrive 88.9. These come from published tables, not paired reruns. nuScenes: avg L2 0.41 m, collision 0.05%. Bench2Drive: DS 76.50, SR 53.06. |
| Diagnostics | Zero tokens reproduce the candidates exactly. Aligned − shuffled selected PDM = +0.000205 (bootstrap [+0.000025, +0.000414]). Oracle-best quality is mixed. Naive conditioning degraded 3 of 14 candidates. In the TTC replay, a 3.7 cm shift erased a 1.1 cm clearance. |
| Limitations stated | Diagnostics use logged-future tokens, so they do not test the deployable predictor. Splits were selected for headroom. Benchmark rows are not same-backbone comparisons. The gain over shuffled tokens is small. The guard is a safety repair, not a gain specific to future tokens. |

## My choices (toy)
* **World:** a synthetic 2-lane street with a hidden future (lead car hard-braking, pedestrian stepping out with a weak lateral cue, cut-in with a weak drift cue, and left-lane traffic). The expert plans with the TRUE future over a {keep, change-left} × 2-stage acceleration lattice.
* **Base generator:** I replaced DiffusionDrive with a deterministic, anchor-based K-mode transformer decoder (K = 16 k-means anchors, 3 layers, d = 64, winner-take-all imitation). There is **no diffusion**.
* **Token sizes:** N_a = 4, N_r = 8, N_o = 8, N_m = K, e ∈ R^4. The injected vector is v = c·[e, r].
* **A_l:** a bias-free FiLM, A(z, u) = (W_a z) ⊙ (W_g u) + W_b u. α_l is a per-channel vector initialised to zero. b_k(u) = ⟨W u, z_k⟩/√d.
* **Pool:** 4 scene-conditioned queries attend over the live tokens. Values are bias-free and all-zero tokens are masked out, so u = 0 exactly when every token is zero.
* **m_TTC:** clearance against a constant-velocity projection of the actors (deployable). ε_b = 1.0 m, ε_g = 0.75 m.
* **Adapter training:** half the batches use teacher (logged-future) tokens and half use predicted tokens. λ_p = 10 (λ_p = 1 is kept as an ablation), λ_g = 1.
* **Metric:** a PDMS-like score = NC·DAC·(5·TTC + 2·Comf + 5·EP)/12, with TTC = clearance > 0.5 m against the TRUE future. This is not the NAVSIM simulator.
