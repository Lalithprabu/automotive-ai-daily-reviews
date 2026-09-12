"""
SSDS + DAPSE -- core architecture (reference re-implementation)
Reference implementation inspired by:
  "One Diffusion Model, Two Roles: Guided Trajectory Planning and
   Safety-Critical Scenario Generation in Closed-Loop Simulation"
  (arXiv:2609.04921, submitted Sept 4 2026, accepted ECCV 2026 workshop)

This is an original re-implementation of the architecture and sampling
algorithm *described* in the paper (module names, mechanisms, and the
DAPSE update rule as given in the text) -- written for the daily-review
series. It is not the authors' own code, and the authors' exact layer
widths/depths and closed-loop benchmark numbers were not independently
verified (see the accompanying README sourcing note).

Why this design exists (from the paper's framing):
  Autonomous-driving stacks traditionally need two *separate* systems: an
  ego motion planner, and a hand-built / adversarially-trained scenario
  generator used to stress-test that planner. This paper shows a single
  pretrained diffusion model over joint future traffic trajectories can
  serve both roles, switched purely at sampling time via a training-free
  guidance scheme (DAPSE) that injects arbitrary energy functions -- no
  retraining, no auxiliary classifier/discriminator network.

Pieces implemented here:
  1. sinusoidal_embedding            -- timestep / scalar embedding helper.
  2. DualStreamBlock                 -- trajectory tokens and scene-context
                                         tokens each get their own adaLN
                                         modulation + independent Q/K/V
                                         projections, but Q/K/V are
                                         concatenated for a single *joint*
                                         attention op (early, symmetric
                                         fusion) rather than one stream
                                         cross-attending into the other late
                                         in the network.
  3. SingleStreamBlock               -- once streams are fused, further
                                         blocks run standard self-attention
                                         + MLP over the concatenated token
                                         sequence (deep fusion stage).
  4. SSDSDenoiser                    -- full noise/x0-prediction network:
                                         embeds noisy trajectory tokens +
                                         scene context, runs N dual-stream
                                         blocks then M single-stream blocks,
                                         and reads off a per-agent x0
                                         (clean trajectory) prediction.
  5. dapse_guided_step               -- ONE outer diffusion timestep of
                                         Decoupled Annealing Posterior
                                         Sampling with Energy: takes the
                                         denoiser's x0 estimate and refines
                                         it with J inner Langevin steps that
                                         blend a reconstruction term, an
                                         arbitrary energy-function gradient,
                                         and injected noise -- entirely at
                                         the clean-sample (t=0) level, so it
                                         avoids the first-order approximation
                                         error of guiding at noisy x_t.
  6. Two example energy functions    -- comfort_energy (pulls the PLANNER
                                         role toward smooth, on-route
                                         trajectories) and
                                         adversarial_proximity_energy (pulls
                                         the SCENARIO GENERATOR role toward
                                         near-miss / collision-adjacent
                                         trajectories for the *other* agents)
                                         -- same frozen model, opposite sign
                                         and target of the same guidance
                                         mechanism.
"""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 0. Sinusoidal embedding for scalar conditioning signals (diffusion timestep)
# --------------------------------------------------------------------------- #
def sinusoidal_embedding(x: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    x:   [B]   scalar values (e.g. diffusion timestep index, possibly float)
    Returns: [B, dim] sinusoidal position/timestep embedding (standard
    transformer/diffusion convention -- half sin, half cos).
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=x.device, dtype=torch.float32) / half
    )                                                              # [half]
    args = x.float().unsqueeze(-1) * freqs.unsqueeze(0)             # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)     # [B, 2*half]
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb                                                       # [B, dim]


class AdaLNModulation(nn.Module):
    """
    Produces (shift, scale, gate) triples from a conditioning vector, used to
    modulate a stream's tokens before attention/MLP and to gate their
    residual contribution back in -- the standard adaLN-Zero recipe used in
    diffusion transformers (DiT-style), applied independently per stream.
    """

    def __init__(self, cond_dim: int, model_dim: int, n_outputs: int = 6):
        super().__init__()
        self.n_outputs = n_outputs
        self.model_dim = model_dim
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, n_outputs * model_dim))
        # Zero-init the final projection: at initialization every block is
        # the identity function (gate = 0), which stabilizes early training
        # of deep diffusion transformers.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond: torch.Tensor):
        """
        cond: [B, cond_dim]
        Returns a tuple of `n_outputs` tensors, each [B, model_dim].
        """
        out = self.net(cond)                                          # [B, n_outputs * D]
        return out.chunk(self.n_outputs, dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # x: [B, T, D], shift/scale: [B, D]
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# --------------------------------------------------------------------------- #
# 1. Dual-stream block: independent streams, one joint attention operation
# --------------------------------------------------------------------------- #
class DualStreamBlock(nn.Module):
    """
    Trajectory-stream tokens and context-stream tokens each get their own
    LayerNorm -> adaLN modulation -> independent Q/K/V linear projections.
    The two streams' Q, K, V are then *concatenated* along the token axis
    and a single softmax-attention is computed jointly over all of them --
    every trajectory token can attend to every context token AND every
    other trajectory token in one operation (symmetric, early fusion),
    unlike a standard DiT block where one stream would only *later*
    cross-attend into a frozen/separately-processed context stream.

    Each stream keeps its own residual stream and its own MLP afterward --
    only the attention operation is shared.
    """

    def __init__(self, model_dim: int, cond_dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        assert model_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads

        self.traj_norm1 = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.ctx_norm1 = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.traj_mod1 = AdaLNModulation(cond_dim, model_dim, n_outputs=6)
        self.ctx_mod1 = AdaLNModulation(cond_dim, model_dim, n_outputs=6)

        self.traj_qkv = nn.Linear(model_dim, 3 * model_dim)
        self.ctx_qkv = nn.Linear(model_dim, 3 * model_dim)
        self.traj_attn_out = nn.Linear(model_dim, model_dim)
        self.ctx_attn_out = nn.Linear(model_dim, model_dim)

        self.traj_norm2 = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.ctx_norm2 = nn.LayerNorm(model_dim, elementwise_affine=False)
        hidden = int(model_dim * mlp_ratio)
        self.traj_mlp = nn.Sequential(nn.Linear(model_dim, hidden), nn.GELU(), nn.Linear(hidden, model_dim))
        self.ctx_mlp = nn.Sequential(nn.Linear(model_dim, hidden), nn.GELU(), nn.Linear(hidden, model_dim))

    def _split_heads(self, qkv: torch.Tensor, b: int, t: int):
        # qkv: [B, T, 3D] -> q,k,v each [B, H, T, hd]
        qkv = qkv.view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def forward(self, traj: torch.Tensor, ctx: torch.Tensor, cond: torch.Tensor):
        """
        traj: [B, Tt, D]   trajectory-stream tokens
        ctx:  [B, Tc, D]   scene-context-stream tokens
        cond: [B, cond_dim]  (timestep + navigation embedding, shared)
        Returns updated (traj, ctx), same shapes.
        """
        b, tt, d = traj.shape
        tc = ctx.shape[1]

        t_shift1, t_scale1, t_gate1, t_shift2, t_scale2, t_gate2 = self.traj_mod1(cond)
        c_shift1, c_scale1, c_gate1, c_shift2, c_scale2, c_gate2 = self.ctx_mod1(cond)

        traj_mod = modulate(self.traj_norm1(traj), t_shift1, t_scale1)   # [B, Tt, D]
        ctx_mod = modulate(self.ctx_norm1(ctx), c_shift1, c_scale1)      # [B, Tc, D]

        traj_qkv = self.traj_qkv(traj_mod)                                # [B, Tt, 3D]
        ctx_qkv = self.ctx_qkv(ctx_mod)                                   # [B, Tc, 3D]

        tq, tk, tv = self._split_heads(traj_qkv, b, tt)                   # each [B, H, Tt, hd]
        cq, ck, cv = self._split_heads(ctx_qkv, b, tc)                    # each [B, H, Tc, hd]

        # ---- the joint attention operation: concatenate along the token axis ----
        q = torch.cat([tq, cq], dim=2)                                    # [B, H, Tt+Tc, hd]
        k = torch.cat([tk, ck], dim=2)                                    # [B, H, Tt+Tc, hd]
        v = torch.cat([tv, cv], dim=2)                                    # [B, H, Tt+Tc, hd]

        attn_out = F.scaled_dot_product_attention(q, k, v)                # [B, H, Tt+Tc, hd]
        attn_out = attn_out.transpose(1, 2).reshape(b, tt + tc, d)        # [B, Tt+Tc, D]

        traj_attn, ctx_attn = attn_out.split([tt, tc], dim=1)             # [B,Tt,D], [B,Tc,D]

        traj = traj + t_gate1.unsqueeze(1) * self.traj_attn_out(traj_attn)
        ctx = ctx + c_gate1.unsqueeze(1) * self.ctx_attn_out(ctx_attn)

        # ---- per-stream MLP (independent, gated adaLN as well) ----
        traj = traj + t_gate2.unsqueeze(1) * self.traj_mlp(modulate(self.traj_norm2(traj), t_shift2, t_scale2))
        ctx = ctx + c_gate2.unsqueeze(1) * self.ctx_mlp(modulate(self.ctx_norm2(ctx), c_shift2, c_scale2))

        return traj, ctx                                                   # [B,Tt,D], [B,Tc,D]


# --------------------------------------------------------------------------- #
# 2. Single-stream block: standard self-attn + MLP over the fused sequence
# --------------------------------------------------------------------------- #
class SingleStreamBlock(nn.Module):
    """
    Once the dual-stream stage has let trajectory and context tokens
    exchange information symmetrically, the single-stream stage concatenates
    them into one sequence and runs plain adaLN-conditioned self-attention +
    MLP -- deep fusion where every token can attend to every other token
    with a single shared set of weights (no more stream-specific
    projections).
    """

    def __init__(self, model_dim: int, cond_dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        assert model_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = model_dim // n_heads

        self.norm1 = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.mod1 = AdaLNModulation(cond_dim, model_dim, n_outputs=6)
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.attn_out = nn.Linear(model_dim, model_dim)

        self.norm2 = nn.LayerNorm(model_dim, elementwise_affine=False)
        hidden = int(model_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(model_dim, hidden), nn.GELU(), nn.Linear(hidden, model_dim))

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor):
        # tokens: [B, T, D]  (trajectory + context tokens concatenated)
        b, t, d = tokens.shape
        shift1, scale1, gate1, shift2, scale2, gate2 = self.mod1(cond)

        x = modulate(self.norm1(tokens), shift1, scale1)                  # [B, T, D]
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                    # each [B, H, T, hd]

        attn_out = F.scaled_dot_product_attention(q, k, v)                  # [B, H, T, hd]
        attn_out = attn_out.transpose(1, 2).reshape(b, t, d)                # [B, T, D]
        tokens = tokens + gate1.unsqueeze(1) * self.attn_out(attn_out)

        tokens = tokens + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(tokens), shift2, scale2))
        return tokens                                                        # [B, T, D]


# --------------------------------------------------------------------------- #
# 3. Full SSDS denoiser
# --------------------------------------------------------------------------- #
@dataclass
class SSDSConfig:
    traj_feat_dim: int = 3        # (x, y, heading) per waypoint per agent, flattened per-token
    ctx_feat_dim: int = 8         # per scene-context token raw feature width
    model_dim: int = 256
    cond_dim: int = 256
    n_heads: int = 8
    n_dual_blocks: int = 4        # "N" dual-stream blocks
    n_single_blocks: int = 4      # "M" single-stream blocks
    mlp_ratio: float = 4.0
    max_traj_tokens: int = 512    # upper bound on agents * horizon_steps


class SSDSDenoiser(nn.Module):
    """
    x0-prediction network for the joint diffusion model over future agent
    trajectories.

    Forward pass shapes:
      noisy_traj:    [B, A, H, 3]   noisy (x, y, heading) for A agents over
                                     an H-step future horizon, at diffusion
                                     timestep t
      ctx_tokens:    [B, C, ctx_feat_dim]  scene-context tokens (map polyline
                                     segments, agent history summaries, ego
                                     route/goal -- already tokenized upstream)
      timesteps:     [B]            diffusion timestep index per sample
      nav_embed:     [B, cond_dim]  navigation/goal conditioning vector

    Returns:
      x0_pred:       [B, A, H, 3]   predicted clean (x, y, heading) trajectory
    """

    def __init__(self, cfg: SSDSConfig):
        super().__init__()
        self.cfg = cfg

        self.traj_in_proj = nn.Linear(cfg.traj_feat_dim, cfg.model_dim)
        self.ctx_in_proj = nn.Linear(cfg.ctx_feat_dim, cfg.model_dim)

        # Learned per-(agent, horizon-step) positional embedding for the
        # trajectory stream, so the same MLP/attention weights can tell
        # "agent 3, step 7" apart from "agent 1, step 2".
        self.traj_pos_embed = nn.Parameter(torch.randn(1, cfg.max_traj_tokens, cfg.model_dim) * 0.02)

        self.timestep_mlp = nn.Sequential(
            nn.Linear(cfg.model_dim, cfg.cond_dim), nn.SiLU(), nn.Linear(cfg.cond_dim, cfg.cond_dim)
        )
        self.nav_proj = nn.Linear(cfg.cond_dim, cfg.cond_dim)

        self.dual_blocks = nn.ModuleList([
            DualStreamBlock(cfg.model_dim, cfg.cond_dim, cfg.n_heads, cfg.mlp_ratio)
            for _ in range(cfg.n_dual_blocks)
        ])
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(cfg.model_dim, cfg.cond_dim, cfg.n_heads, cfg.mlp_ratio)
            for _ in range(cfg.n_single_blocks)
        ])

        self.final_norm = nn.LayerNorm(cfg.model_dim, elementwise_affine=False)
        self.final_mod = AdaLNModulation(cfg.cond_dim, cfg.model_dim, n_outputs=2)  # shift, scale only
        self.out_proj = nn.Linear(cfg.model_dim, cfg.traj_feat_dim)

    def forward(
        self,
        noisy_traj: torch.Tensor,      # [B, A, H, 3]
        ctx_tokens: torch.Tensor,      # [B, C, ctx_feat_dim]
        timesteps: torch.Tensor,       # [B]
        nav_embed: torch.Tensor,       # [B, cond_dim]
    ) -> torch.Tensor:
        b, a, h, _ = noisy_traj.shape

        traj_tok = self.traj_in_proj(noisy_traj).view(b, a * h, self.cfg.model_dim)  # [B, A*H, D]
        pos = self.traj_pos_embed[:, : a * h, :]
        traj_tok = traj_tok + pos                                                     # [B, A*H, D]

        ctx_tok = self.ctx_in_proj(ctx_tokens)                                        # [B, C, D]

        t_emb = sinusoidal_embedding(timesteps, self.cfg.model_dim)                   # [B, D]
        t_emb = self.timestep_mlp(t_emb)                                              # [B, cond_dim]
        cond = t_emb + self.nav_proj(nav_embed)                                       # [B, cond_dim]

        for block in self.dual_blocks:
            traj_tok, ctx_tok = block(traj_tok, ctx_tok, cond)                        # shapes preserved

        fused = torch.cat([traj_tok, ctx_tok], dim=1)                                 # [B, A*H+C, D]
        for block in self.single_blocks:
            fused = block(fused, cond)                                                # [B, A*H+C, D]

        traj_out = fused[:, : a * h, :]                                               # [B, A*H, D]
        shift, scale = self.final_mod(cond)
        traj_out = modulate(self.final_norm(traj_out), shift, scale)                  # [B, A*H, D]

        x0_pred = self.out_proj(traj_out).view(b, a, h, self.cfg.traj_feat_dim)       # [B, A, H, 3]
        return x0_pred


# --------------------------------------------------------------------------- #
# 4. DAPSE: Decoupled Annealing Posterior Sampling with Energy
# --------------------------------------------------------------------------- #
def dapse_guided_step(
    x_t: torch.Tensor,
    denoiser: SSDSDenoiser,
    ctx_tokens: torch.Tensor,
    t_index: torch.Tensor,
    nav_embed: torch.Tensor,
    r_t: float,
    energy_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
    beta: float = 1.0,
    eta: float = 0.05,
    n_inner_steps: int = 4,
) -> torch.Tensor:
    """
    ONE outer diffusion timestep of DAPSE guidance. Given the current noisy
    joint trajectory x_t, this:
      1. Gets the network's unguided posterior-mean estimate
         x_hat0 = denoiser(x_t, t)  -- treated as the mean of a Gaussian
         approximate posterior q(x0 | x_t) ~= N(x_hat0, r_t^2 I).
      2. Refines a running clean-sample estimate x0 toward BOTH that
         Gaussian (reconstruction term) AND an arbitrary, possibly
         non-differentiable-friendly energy function E0 (energy term),
         using J steps of Langevin dynamics -- entirely at the t=0 (clean)
         level, so no first-order approximation of a noisy score is needed.

    This mirrors the update rule described in the paper:
        x0^(j+1) = x0^(j)
            - eta * grad[ ||x0^(j) - x_hat0||^2 / (2 * r_t^2) ]      (recon)
            - eta * beta * grad[ E0(x0^(j)) ]                        (energy)
            + sqrt(2 * eta) * noise^(j)                               (Langevin)

    Args:
      x_t:        [B, A, H, 3]  current noisy joint trajectory
      denoiser:   the SSDSDenoiser (frozen at sampling time)
      ctx_tokens: [B, C, ctx_feat_dim]
      t_index:    [B]  current diffusion timestep index
      nav_embed:  [B, cond_dim]
      r_t:        float, the posterior-approximation std at this timestep
                  (schedule-dependent; larger early in sampling, shrinking
                  toward 0 as t -> 0)
      energy_fn:  callable mapping a candidate clean trajectory
                  [B, A, H, 3] -> per-batch scalar energy [B]. Pass `None`
                  for the un-guided PLANNER-default behaviour (pure
                  reconstruction term = ordinary posterior sampling).
                  Use `comfort_energy` for extra planning-quality guidance,
                  or `adversarial_proximity_energy` to switch this exact
                  same frozen model into the SCENARIO GENERATOR role.
      beta:       energy-term weight.
      eta:        Langevin step size.
      n_inner_steps: number of inner MCMC iterations J.

    Returns:
      x0_refined: [B, A, H, 3] refined clean-sample estimate for this
                  timestep (the caller uses this, together with the
                  timestep's forward-process formula, to sample the next,
                  less-noisy x_{t-1}).
    """
    with torch.no_grad():
        x_hat0 = denoiser(x_t, ctx_tokens, t_index, nav_embed)     # [B, A, H, 3], no grad needed for the target

    x0 = x_hat0.clone().detach()

    for _ in range(n_inner_steps):
        x0 = x0.requires_grad_(True)

        recon_term = ((x0 - x_hat0) ** 2).flatten(1).sum(-1) / (2 * r_t ** 2)   # [B]

        if energy_fn is not None:
            energy_term = beta * energy_fn(x0)                                    # [B]
            total = (recon_term + energy_term).sum()
        else:
            total = recon_term.sum()

        (grad,) = torch.autograd.grad(total, x0)                                  # [B, A, H, 3]

        noise = torch.randn_like(x0)
        x0 = (x0.detach() - eta * grad + math.sqrt(2 * eta) * noise).detach()

    return x0                                                                       # [B, A, H, 3]


# --------------------------------------------------------------------------- #
# 5. Example energy functions -- same guidance mechanism, opposite intent
# --------------------------------------------------------------------------- #
def comfort_energy(candidate_traj: torch.Tensor, ego_index: int = 0) -> torch.Tensor:
    """
    PLANNER-role energy: penalizes jerky ego motion (large second-difference
    in position) so DAPSE nudges samples toward smoother, more comfortable
    ego trajectories without needing a separate comfort-classifier network.

    candidate_traj: [B, A, H, 3]  (x, y, heading)
    Returns: [B] scalar energy (lower = more comfortable).
    """
    ego_xy = candidate_traj[:, ego_index, :, :2]                 # [B, H, 2]
    vel = ego_xy[:, 1:, :] - ego_xy[:, :-1, :]                    # [B, H-1, 2]
    accel = vel[:, 1:, :] - vel[:, :-1, :]                        # [B, H-2, 2]
    jerk = accel[:, 1:, :] - accel[:, :-1, :]                     # [B, H-3, 2]
    return (jerk ** 2).flatten(1).sum(-1)                          # [B]


def adversarial_proximity_energy(candidate_traj: torch.Tensor, ego_index: int = 0, target_gap_m: float = 1.5) -> torch.Tensor:
    """
    SCENARIO-GENERATOR-role energy: the SAME frozen SSDS model, guided with
    the *opposite* intent -- rewards (i.e. gives LOW energy to) trajectories
    for the non-ego agents that come close to the ego's path at some point
    in the horizon, producing plausible-but-safety-critical near-miss
    scenarios for stress-testing a planner, without retraining anything.

    candidate_traj: [B, A, H, 3]
    Returns: [B] scalar energy (lower = closer near-miss = "more adversarial").
    """
    ego_xy = candidate_traj[:, ego_index : ego_index + 1, :, :2]     # [B, 1, H, 2]
    other_xy = torch.cat(
        [candidate_traj[:, :ego_index, :, :2], candidate_traj[:, ego_index + 1 :, :, :2]], dim=1
    )                                                                 # [B, A-1, H, 2]
    dist = torch.norm(other_xy - ego_xy, dim=-1)                      # [B, A-1, H]
    min_dist_per_batch = dist.amin(dim=(1, 2))                        # [B]
    # Energy is low (attractive) when min_dist is near target_gap_m,
    # rising on both sides -- "close enough to be a real stress-test,
    # not literally overlapping" (which the reconstruction term already
    # discourages via the likelihood prior).
    return (min_dist_per_batch - target_gap_m) ** 2                    # [B]
