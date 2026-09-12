"""
COSTER -- core architecture (reference re-implementation)
Reference implementation inspired by:
  "Collision Snapshot Guided Time-Reversed Safety-Critical Scenario
   Generation" (COSTER, arXiv:2609.06433, submitted Sept 6 2026)

RECONSTRUCTION NOTE: the original research session for this paper produced a
complete repo package (verified with a real train.py + pytest run), but
that session ran in a separate, isolated container that this session
cannot access -- only the paper's abstract, module names, and the
TimeReversedRolloutDecoder excerpt survive in the project log. This file is
therefore a *fresh* original reconstruction of the described architecture
(module names and data flow only, from the project log's own description),
not a byte-for-byte replay of the earlier package and not the authors' own
code. Only one quantitative claim from the paper is used anywhere in this
package: a 31% collision-rate reduction on safety-critical Waymo Open
Motion Dataset scenarios (abstract-level, cross-checked nowhere else) --
see the README sourcing note.

Why this design exists (from the paper's framing):
  Prior safety-critical scenario generators perturb a LOGGED trajectory with
  a hand-tuned adversarial objective -- limited plausibility and diversity,
  bounded by the allowed deformation. COSTER instead (1) predicts *where* a
  collision plausibly belongs along a target agent's future path from a
  learned traffic prior, (2) composes a "collision state" placing a new
  vehicle already in contact with the target at that point, then (3) rolls
  a conditional VAE decoder *backward* through time, one step per call,
  from the collision instant back to a plausible earliest state -- the
  mirror image of ordinary forward trajectory prediction.

Pieces implemented here:
  1. PolylineMapEncoder       -- VectorNet-style per-polyline point encoder
                                  (shared point MLP + max-pool) followed by
                                  self-attention across polylines.
  2. AgentHistoryEncoder      -- GRU encoder over each agent's past
                                  (x, y, heading, v) states.
  3. SceneContextEncoder      -- social cross-attention: the target agent's
                                  history embedding attends over all other
                                  agents' embeddings and all map-polyline
                                  embeddings, producing one scene-context
                                  vector per target agent.
  4. CollisionSnapshotHead    -- from scene context, predicts (a) a
                                  categorical distribution over *when* along
                                  the target's future path a collision is
                                  plausible, and (b) a regressed contact-pose
                                  offset (relative position + heading) for
                                  where an inserted vehicle should make
                                  contact.
  5. compose_collision_state  -- combines the target's state at the
                                  predicted collision time with the
                                  predicted contact-pose offset into the
                                  full [x, y, heading, v] collision-instant
                                  state that seeds the backward rollout.
  6. PosteriorEncoder          -- (training-time only) encodes the
                                  ground-truth lead-up trajectory into a
                                  latent Gaussian posterior q(z | x, scene).
  7. ConditionalPriorNet       -- (used at both train and inference time)
                                  predicts a latent Gaussian prior
                                  p(z | scene) from scene context alone, so
                                  inference never needs ground truth.
  8. TimeReversedRolloutDecoder -- the paper's core mechanism: an
                                  autoregressive GRU cell that starts at the
                                  collision-instant pose and rolls BACKWARD
                                  in time into a plausible lead-up.
  9. COSTERModel + coster_loss -- ties everything together and defines the
                                  CVAE reconstruction + collision-snapshot +
                                  KL training objective.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class COSTERConfig:
    map_point_dim: int = 7          # per map-polyline-point raw feature width (x,y,dx,dy,type-onehot...)
    agent_state_dim: int = 4        # (x, y, heading, v) per past timestep
    hidden_dim: int = 128
    latent_dim: int = 32
    state_dim: int = 4              # (x, y, heading, v) -- collision-state / rollout state width
    n_history_steps: int = 20       # observed past steps per agent
    t_rev: int = 30                 # backward rollout length (collision instant -> earliest state)
    n_collision_time_bins: int = 20  # discretized "when along the future path" classification
    n_heads: int = 4


# --------------------------------------------------------------------------- #
# 1. Map polyline encoder (VectorNet-style)
# --------------------------------------------------------------------------- #
class PolylineMapEncoder(nn.Module):
    """
    Encodes a set of map polylines (e.g. lane centerlines, boundaries) into
    one embedding per polyline: a shared per-point MLP followed by max-pool
    over points within each polyline, then a self-attention layer lets
    polylines exchange context (e.g. "this lane merges with that one").
    """

    def __init__(self, cfg: COSTERConfig):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(cfg.map_point_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.polyline_attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.n_heads, batch_first=True)
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, polylines: torch.Tensor, point_mask: torch.Tensor) -> torch.Tensor:
        """
        polylines:  [B, M, P, map_point_dim]  M polylines, P points each
        point_mask: [B, M, P]  True where a point is valid

        Returns: [B, M, hidden_dim]  one embedding per polyline.
        """
        b, m, p, _ = polylines.shape
        point_feats = self.point_mlp(polylines)                              # [B, M, P, H]
        point_feats = point_feats.masked_fill(~point_mask.unsqueeze(-1), float("-inf"))
        polyline_embed = point_feats.amax(dim=2)                             # [B, M, H]  max-pool over points
        polyline_embed = torch.nan_to_num(polyline_embed, neginf=0.0)        # all-invalid polylines -> 0

        attn_out, _ = self.polyline_attn(polyline_embed, polyline_embed, polyline_embed)
        return self.norm(polyline_embed + attn_out)                          # [B, M, H]


# --------------------------------------------------------------------------- #
# 2. Agent history encoder
# --------------------------------------------------------------------------- #
class AgentHistoryEncoder(nn.Module):
    """GRU encoder over each agent's observed past (x, y, heading, v) states."""

    def __init__(self, cfg: COSTERConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.agent_state_dim, cfg.hidden_dim)
        self.gru = nn.GRU(cfg.hidden_dim, cfg.hidden_dim, batch_first=True)

    def forward(self, agent_histories: torch.Tensor) -> torch.Tensor:
        """
        agent_histories: [B, N, T_hist, agent_state_dim]
        Returns: [B, N, hidden_dim]  final GRU hidden state per agent.
        """
        b, n, t, d = agent_histories.shape
        x = self.input_proj(agent_histories.view(b * n, t, d))               # [B*N, T, H]
        _, h_n = self.gru(x)                                                  # h_n: [1, B*N, H]
        return h_n.squeeze(0).view(b, n, -1)                                  # [B, N, H]


# --------------------------------------------------------------------------- #
# 3. Scene context encoder (social attention over agents + map)
# --------------------------------------------------------------------------- #
class SceneContextEncoder(nn.Module):
    """
    Produces one scene-context vector per TARGET agent by letting the
    target's history embedding cross-attend over every other agent's
    embedding and every map-polyline embedding.
    """

    def __init__(self, cfg: COSTERConfig):
        super().__init__()
        self.social_attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.n_heads, batch_first=True)
        self.map_attn = nn.MultiheadAttention(cfg.hidden_dim, cfg.n_heads, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 3, cfg.hidden_dim), nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

    def forward(self, target_embed: torch.Tensor, agent_embeds: torch.Tensor,
                map_embeds: torch.Tensor) -> torch.Tensor:
        """
        target_embed: [B, H]        the agent COSTER will generate a collision for
        agent_embeds: [B, N, H]     all agents' history embeddings (includes target)
        map_embeds:   [B, M, H]     polyline embeddings

        Returns: [B, H]  fused scene-context vector.
        """
        q = target_embed.unsqueeze(1)                                        # [B, 1, H]
        social_ctx, _ = self.social_attn(q, agent_embeds, agent_embeds)       # [B, 1, H]
        map_ctx, _ = self.map_attn(q, map_embeds, map_embeds)                 # [B, 1, H]

        fused = torch.cat([target_embed, social_ctx.squeeze(1), map_ctx.squeeze(1)], dim=-1)
        return self.fuse(fused)                                               # [B, H]


# --------------------------------------------------------------------------- #
# 4. Collision Snapshot Head
# --------------------------------------------------------------------------- #
class CollisionSnapshotHead(nn.Module):
    """
    From scene context, predicts:
      (a) collision_time_logits: which of n_collision_time_bins steps along
          the target's future path a collision plausibly occurs at.
      (b) contact_pose_offset: a (dx, dy, dtheta) offset describing where,
          relative to the target's predicted state at that time, an
          inserted adversarial vehicle should make contact.
    """

    def __init__(self, cfg: COSTERConfig):
        super().__init__()
        self.time_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2), nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, cfg.n_collision_time_bins),
        )
        self.contact_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2), nn.GELU(),
            nn.Linear(cfg.hidden_dim // 2, 3),   # (dx, dy, dtheta)
        )

    def forward(self, scene_context: torch.Tensor):
        # scene_context: [B, H]
        collision_time_logits = self.time_head(scene_context)                # [B, n_bins]
        contact_pose_offset = self.contact_head(scene_context)               # [B, 3]
        return collision_time_logits, contact_pose_offset


def compose_collision_state(
    target_future_states: torch.Tensor,   # [B, n_bins, state_dim]  target's own (x,y,theta,v) at each candidate future step
    collision_time_logits: torch.Tensor,  # [B, n_bins]
    contact_pose_offset: torch.Tensor,    # [B, 3]  (dx, dy, dtheta)
    hard: bool = False,
) -> torch.Tensor:
    """
    Combines the target's predicted state at the (soft- or hard-selected)
    collision time with the predicted contact-pose offset into the full
    collision-instant state [x, y, heading, v] that seeds the backward
    rollout decoder.

    During training, uses a soft (differentiable) expectation over time
    bins (Gumbel/softmax-style) so gradients reach `collision_time_logits`;
    at inference, `hard=True` selects the argmax bin directly.
    """
    if hard:
        idx = collision_time_logits.argmax(dim=-1)                            # [B]
        b = torch.arange(target_future_states.shape[0], device=target_future_states.device)
        base_state = target_future_states[b, idx]                             # [B, state_dim]
    else:
        weights = F.softmax(collision_time_logits, dim=-1).unsqueeze(-1)      # [B, n_bins, 1]
        base_state = (weights * target_future_states).sum(dim=1)              # [B, state_dim]

    x, y, theta, v = base_state.unbind(dim=-1)
    dx, dy, dtheta = contact_pose_offset.unbind(dim=-1)
    new_theta = torch.remainder(theta + dtheta + math.pi, 2 * math.pi) - math.pi
    collision_state = torch.stack([x + dx, y + dy, new_theta, v], dim=-1)     # [B, state_dim]
    return collision_state


# --------------------------------------------------------------------------- #
# 5 & 6. Posterior encoder (train-only) and conditional prior (train + infer)
# --------------------------------------------------------------------------- #
class PosteriorEncoder(nn.Module):
    """
    Training-time-only recognition network: encodes the GROUND-TRUTH
    lead-up trajectory (the real backward sequence from collision instant to
    earliest observed state) jointly with scene context into a latent
    Gaussian posterior q(z | x_gt, scene) -- standard CVAE recipe.
    """

    def __init__(self, cfg: COSTERConfig):
        super().__init__()
        self.traj_gru = nn.GRU(cfg.state_dim, cfg.hidden_dim, batch_first=True)
        self.to_stats = nn.Linear(cfg.hidden_dim * 2, cfg.latent_dim * 2)

    def forward(self, gt_lead_up_traj: torch.Tensor, scene_context: torch.Tensor):
        """
        gt_lead_up_traj: [B, t_rev, state_dim]  ground-truth backward-ordered trajectory
        scene_context:   [B, hidden_dim]
        Returns: (mu, logvar), each [B, latent_dim]
        """
        _, h_n = self.traj_gru(gt_lead_up_traj)                               # [1, B, H]
        traj_embed = h_n.squeeze(0)                                            # [B, H]
        stats = self.to_stats(torch.cat([traj_embed, scene_context], dim=-1))  # [B, 2*latent_dim]
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar


class ConditionalPriorNet(nn.Module):
    """Predicts a latent Gaussian prior p(z | scene) from scene context alone."""

    def __init__(self, cfg: COSTERConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim * 2),
        )

    def forward(self, scene_context: torch.Tensor):
        stats = self.net(scene_context)                                       # [B, 2*latent_dim]
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# --------------------------------------------------------------------------- #
# 7. Time-Reversed Rollout Decoder (the paper's title mechanism)
# --------------------------------------------------------------------------- #
class TimeReversedRolloutDecoder(nn.Module):
    """The paper's core mechanism: an autoregressive GRU cell that starts at
    the collision-instant pose and rolls BACKWARD in time, one step per
    call, predicting the residual state change to the chronologically
    *earlier* step. After `t_rev` steps it has reconstructed a full
    plausible trajectory arriving at the collision, ordered
    [collision instant, ..., earliest state].
    """

    def __init__(self, cfg: COSTERConfig):
        super().__init__()
        self.t_rev = cfg.t_rev
        self.z_to_h0 = nn.Linear(cfg.latent_dim + cfg.hidden_dim, cfg.hidden_dim)
        self.cell = nn.GRUCell(cfg.state_dim + cfg.hidden_dim, cfg.hidden_dim)
        self.to_delta = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim // 2, cfg.state_dim),
        )

    def forward(self, collision_state, z, scene_context):
        # collision_state: [B, state_dim] — t=0 (the collision instant)
        # z:               [B, latent_dim]
        # scene_context:   [B, hidden_dim]
        B = collision_state.shape[0]
        h = torch.tanh(self.z_to_h0(torch.cat([z, scene_context], dim=-1)))  # [B, H]

        cur_state = collision_state
        step_input = torch.zeros_like(collision_state)
        outputs = [cur_state]

        for _ in range(self.t_rev - 1):
            gru_in = torch.cat([step_input, scene_context], dim=-1)
            h = self.cell(gru_in, h)
            delta = self.to_delta(h)  # predicted backward-time residual [B, state_dim]

            # Integrate backward: earlier_state = current_state + delta.
            x, y, theta, v = cur_state.unbind(dim=-1)
            dx, dy, dtheta, dv = delta.unbind(dim=-1)
            new_theta = torch.remainder(theta + dtheta + math.pi, 2 * math.pi) - math.pi
            cur_state = torch.stack([x + dx, y + dy, new_theta, v + dv], dim=-1)

            outputs.append(cur_state)
            step_input = delta

        return torch.stack(outputs, dim=1)  # [B, t_rev, state_dim]


# --------------------------------------------------------------------------- #
# 8. Full COSTER model
# --------------------------------------------------------------------------- #
class COSTERModel(nn.Module):
    def __init__(self, cfg: COSTERConfig):
        super().__init__()
        self.cfg = cfg
        self.map_encoder = PolylineMapEncoder(cfg)
        self.agent_encoder = AgentHistoryEncoder(cfg)
        self.scene_encoder = SceneContextEncoder(cfg)
        self.snapshot_head = CollisionSnapshotHead(cfg)
        self.posterior = PosteriorEncoder(cfg)
        self.prior = ConditionalPriorNet(cfg)
        self.decoder = TimeReversedRolloutDecoder(cfg)

    def forward(
        self,
        polylines: torch.Tensor,          # [B, M, P, map_point_dim]
        point_mask: torch.Tensor,         # [B, M, P]
        agent_histories: torch.Tensor,    # [B, N, T_hist, agent_state_dim]
        target_idx: torch.Tensor,         # [B] index of the target agent within N
        target_future_states: torch.Tensor,  # [B, n_bins, state_dim] candidate future states along target's path
        gt_lead_up_traj: Optional[torch.Tensor] = None,  # [B, t_rev, state_dim], train-mode only
    ):
        b = polylines.shape[0]

        map_embeds = self.map_encoder(polylines, point_mask)                  # [B, M, H]
        agent_embeds = self.agent_encoder(agent_histories)                    # [B, N, H]
        target_embed = agent_embeds[torch.arange(b, device=agent_embeds.device), target_idx]  # [B, H]

        scene_context = self.scene_encoder(target_embed, agent_embeds, map_embeds)  # [B, H]

        collision_time_logits, contact_pose_offset = self.snapshot_head(scene_context)
        collision_state = compose_collision_state(
            target_future_states, collision_time_logits, contact_pose_offset,
            hard=not self.training,
        )                                                                       # [B, state_dim]

        if self.training:
            assert gt_lead_up_traj is not None, "gt_lead_up_traj is required in training mode"
            post_mu, post_logvar = self.posterior(gt_lead_up_traj, scene_context)
            z = reparameterize(post_mu, post_logvar)
        else:
            post_mu = post_logvar = None
            prior_mu, prior_logvar = self.prior(scene_context)
            z = reparameterize(prior_mu, prior_logvar)

        prior_mu_out, prior_logvar_out = self.prior(scene_context)            # always computed (for KL at train time)
        rollout = self.decoder(collision_state, z, scene_context)             # [B, t_rev, state_dim]

        return {
            "rollout": rollout,
            "collision_state": collision_state,
            "collision_time_logits": collision_time_logits,
            "contact_pose_offset": contact_pose_offset,
            "posterior_mu": post_mu,
            "posterior_logvar": post_logvar,
            "prior_mu": prior_mu_out,
            "prior_logvar": prior_logvar_out,
        }


# --------------------------------------------------------------------------- #
# 9. Training objective
# --------------------------------------------------------------------------- #
def coster_loss(
    outputs: dict,
    gt_lead_up_traj: torch.Tensor,          # [B, t_rev, state_dim]
    gt_collision_time_bin: torch.Tensor,    # [B] long, index of the true collision-time bin
    gt_contact_pose_offset: torch.Tensor,   # [B, 3]
    beta: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """
    L = reconstruction (backward-trajectory L1)
      + collision-time cross-entropy
      + contact-pose L1 ("NLL" under a fixed-variance Laplace, i.e. L1)
      + beta * KL(posterior || prior)
    """
    recon_loss = F.l1_loss(outputs["rollout"], gt_lead_up_traj)

    time_ce = F.cross_entropy(outputs["collision_time_logits"], gt_collision_time_bin)
    contact_l1 = F.l1_loss(outputs["contact_pose_offset"], gt_contact_pose_offset)

    post_mu, post_logvar = outputs["posterior_mu"], outputs["posterior_logvar"]
    prior_mu, prior_logvar = outputs["prior_mu"], outputs["prior_logvar"]
    # Closed-form KL between two diagonal Gaussians q(post) || p(prior).
    kl = 0.5 * (
        prior_logvar - post_logvar
        + (post_logvar.exp() + (post_mu - prior_mu) ** 2) / prior_logvar.exp()
        - 1.0
    ).sum(dim=-1).mean()

    total = recon_loss + time_ce + contact_l1 + beta * kl
    return total, {
        "recon_loss": recon_loss.item(),
        "time_ce": time_ce.item(),
        "contact_l1": contact_l1.item(),
        "kl": kl.item(),
        "total_loss": total.item(),
    }
