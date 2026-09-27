"""
LateFusionNetwork: this repo's reconstruction of the paper's actual
contribution -- a learned, late-stage fusion of a neural trajectory
predictor's output with multiple classical EKF trajectory candidates.

Per the abstract: "this study proposes a framework that fuses the output of
Trajectron++ ... with extended Kalman filter (EKF)-based multiple trajectory
candidates at a late stage." The abstract does not disclose the fusion
architecture's internals (attention vs. simple weighting, per-timestep vs.
global, etc.) -- the design below (shared per-candidate encoder -> temporal
softmax attention -> residual refinement) is OUR reconstruction, chosen to be
architecturally consistent with "learned fusion" while remaining small and
auditable. Do not present these design choices as paper-sourced.

Inputs (B = batch size, T = future_len = 12, K_EKF = 4, K_TOTAL = K_EKF+1 = 5):
    neural_traj   : [B, T, 2]         neural predictor's best-guess future
    ekf_candidates: [B, K_EKF, T, 2]  4 EKF candidate futures (CV, CA, CTRV, CTRA)
    ekf_uncertainty:[B, K_EKF, T]     per-step covariance-trace uncertainty

Pipeline:
    1. Stack neural + EKF candidates -> candidates_all [B, 5, T, 2]
       Stack neural uncertainty (zeros, since the point-estimate neural
       predictor here has no explicit per-step variance head -- a disclosed
       simplification / limitation) with EKF uncertainty -> unc_all [B,5,T]
    2. Shared per-candidate MLP encodes (flattened trajectory + uncertainty)
       -> candidate_embed [B, 5, embed_dim]
    3. Learned per-timestep temporal embedding [T, embed_dim] is added to
       each candidate's embedding, then a small MLP scores each
       (candidate, timestep) pair -> attn_logits [B, 5, T]
    4. Softmax over the CANDIDATE axis (dim=1) at each timestep independently
       -> weights [B, 5, T], so early timesteps can favor the neural
       predictor while later/turning timesteps can favor whichever EKF model
       fits -- this is the "learned fusion" the paper's abstract describes.
    5. Weighted sum over candidates -> fused_traj [B, T, 2]
    6. A residual-correction MLP (conditioned on the fused trajectory and a
       pooled candidate embedding) predicts a small per-step correction,
       added back to `fused_traj` for the final refined prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LateFusionNetwork(nn.Module):
    def __init__(self, future_len: int = 12, num_ekf: int = 4,
                 embed_dim: int = 32, attn_hidden: int = 32, residual_hidden: int = 32):
        super().__init__()
        self.future_len = future_len
        self.num_candidates = num_ekf + 1  # +1 for the neural predictor

        # Per-candidate input = flattened trajectory (T*2) + uncertainty (T)
        cand_input_dim = future_len * 2 + future_len

        # --- (2) shared per-candidate encoder ---
        self.candidate_encoder = nn.Sequential(
            nn.Linear(cand_input_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # --- (3) learned per-timestep positional embedding for attention ---
        self.time_embed = nn.Parameter(torch.randn(future_len, embed_dim) * 0.1)

        self.attn_mlp = nn.Sequential(
            nn.Linear(embed_dim, attn_hidden), nn.ReLU(),
            nn.Linear(attn_hidden, 1),
        )

        # --- (6) residual refinement head ---
        # conditions on the flattened fused trajectory + a pooled (mean)
        # candidate embedding, outputs a per-step (dx, dy) correction.
        self.residual_mlp = nn.Sequential(
            nn.Linear(future_len * 2 + embed_dim, residual_hidden), nn.ReLU(),
            nn.Linear(residual_hidden, future_len * 2),
        )

    def forward(self, neural_traj: torch.Tensor, ekf_candidates: torch.Tensor,
                ekf_uncertainty: torch.Tensor):
        """
        neural_traj    : [B, T, 2]
        ekf_candidates : [B, K_EKF, T, 2]
        ekf_uncertainty: [B, K_EKF, T]

        Returns:
            fused_final : [B, T, 2]   final fused + residual-corrected trajectory
            weights     : [B, 5, T]   per-candidate, per-timestep softmax weights
        """
        B, T, _ = neural_traj.shape
        K_EKF = ekf_candidates.shape[1]
        assert T == self.future_len

        # (1) stack neural + EKF candidates along the candidate axis
        neural_unc = torch.zeros(B, 1, T, device=neural_traj.device, dtype=neural_traj.dtype)
        candidates_all = torch.cat([neural_traj.unsqueeze(1), ekf_candidates], dim=1)   # [B, 5, T, 2]
        unc_all = torch.cat([neural_unc, ekf_uncertainty], dim=1)                        # [B, 5, T]
        K = candidates_all.shape[1]  # = 5
        assert K == self.num_candidates

        # (2) shared per-candidate encoder
        traj_flat = candidates_all.reshape(B, K, T * 2)          # [B, 5, T*2]
        cand_input = torch.cat([traj_flat, unc_all], dim=-1)      # [B, 5, T*2+T]
        candidate_embed = self.candidate_encoder(cand_input)      # [B, 5, embed_dim]

        # (3) add per-timestep positional embedding, score each (candidate, t)
        # candidate_embed: [B, 5, 1, E] + time_embed: [1, 1, T, E] -> [B, 5, T, E]
        combined = candidate_embed.unsqueeze(2) + self.time_embed.unsqueeze(0).unsqueeze(0)
        attn_logits = self.attn_mlp(combined).squeeze(-1)          # [B, 5, T]

        # (4) softmax over the candidate axis (dim=1), independently per timestep
        weights = torch.softmax(attn_logits, dim=1)                # [B, 5, T], sums to 1 over dim=1

        # (5) weighted fusion: fused[b,t,:] = sum_c weights[b,c,t] * candidates_all[b,c,t,:]
        fused_traj = torch.einsum('bct,bctd->btd', weights, candidates_all)  # [B, T, 2]

        # (6) residual refinement
        pooled_embed = candidate_embed.mean(dim=1)                 # [B, embed_dim]
        residual_input = torch.cat([fused_traj.reshape(B, T * 2), pooled_embed], dim=-1)
        correction = self.residual_mlp(residual_input).reshape(B, T, 2)  # [B, T, 2]
        fused_final = fused_traj + correction

        return fused_final, weights
