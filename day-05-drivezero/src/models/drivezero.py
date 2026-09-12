"""
DriveZero -- core architecture (reference re-implementation)
Reference implementation inspired by:
  "DriveZero: End-to-End Driving Beyond Human Demonstrations"
  (arXiv:2609.06055, submitted Sept 5 2026)

RECONSTRUCTION NOTE: the original research session for this paper produced a
complete repo package, but that session ran in a separate, isolated
container that this session cannot access -- only the paper's abstract,
module names, and a few verified numbers survive in the project log. This
file is therefore a *fresh* original reconstruction of the described
architecture (module names and data flow only), not a byte-for-byte replay
of the earlier package and not the authors' own code. Exact NAVSIM/HUGSIM
scores and the PPO reward function were never independently verified in the
original session either -- see the README sourcing note.

Why this design exists (from the paper's framing):
  Standard imitation-learning driving policies are capped by what a human
  demonstrator actually drove: they inherit demonstration quality limits,
  compounding rollout error, and causal confusion, and can never exceed the
  driver they imitate by construction. DriveZero instead trains a
  *privileged* closed-loop RL teacher (with access to ground-truth BEV /
  occupancy state) via PPO inside a simulator, then distills the teacher's
  own rollouts -- including under goals never seen in any human log -- into
  a camera-only student that can be deployed without privileged state.

Pieces implemented here:
  1. ModalityAdapter          -- small per-source projection that maps a
                                  frozen vision-foundation-model's raw
                                  feature tokens (DINOv3 / SigLIP2 / SAM /
                                  Depth Anything V2 -- each with a different
                                  native width) into one shared token width.
  2. DriveVFM                 -- fuses the four adapted modality streams
                                  through a shared self-attention fusion
                                  transformer into "unified driving tokens".
  3. PrivilegedTeacherPolicy  -- a Gaussian policy (+ value head) that reads
                                  BOTH the unified driving tokens AND
                                  privileged ground-truth BEV/occupancy
                                  state -- only available in simulation,
                                  which is exactly what makes it "privileged".
  4. ppo_clipped_update       -- the standard PPO clipped-surrogate policy
                                  loss + value loss + entropy bonus.
  5. CameraOnlyStudentPolicy  -- a Gaussian policy that reads ONLY the
                                  unified driving tokens (no privileged
                                  state) plus a goal/navigation embedding --
                                  this is what actually gets deployed.
  6. distillation_loss        -- goal-conditioned distillation: the student
                                  is trained to match the teacher's action
                                  distribution on the teacher's OWN rollout
                                  states (not just logged human states),
                                  including states reached under augmented
                                  goals the original logs never contained.
  7. value_guided_action_search -- test-time inference trick: sample K
                                  candidate actions from the student policy,
                                  score each with the teacher's value head,
                                  and execute the highest-scoring one.
"""

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Per-source modality adapter
# --------------------------------------------------------------------------- #
class ModalityAdapter(nn.Module):
    """
    Projects one frozen VFM's raw per-token feature width into the shared
    model width, with LayerNorm + GELU for a mild nonlinear re-mapping (the
    VFM backbone itself stays frozen; only this adapter is trained).
    """

    def __init__(self, in_dim: int, model_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        # A learned per-source embedding lets the fusion transformer tell
        # "this token came from SAM" apart from "this token came from DINOv3".
        self.source_embed = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.source_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N_tokens, in_dim] -> [B, N_tokens, model_dim]
        return self.proj(x) + self.source_embed


# --------------------------------------------------------------------------- #
# 2. DriveVFM: multi-source fusion into unified driving tokens
# --------------------------------------------------------------------------- #
@dataclass
class DriveVFMConfig:
    source_dims: Dict[str, int]     # e.g. {"dinov3": 1024, "siglip2": 1152, "sam": 256, "depth_anything_v2": 384}
    model_dim: int = 256
    n_heads: int = 8
    n_fusion_layers: int = 4
    mlp_ratio: float = 4.0


class _FusionBlock(nn.Module):
    """Standard pre-norm self-attention + MLP block over the concatenated
    multi-source token sequence."""

    def __init__(self, model_dim: int, n_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = nn.MultiheadAttention(model_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(model_dim)
        hidden = int(model_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(model_dim, hidden), nn.GELU(), nn.Linear(hidden, model_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class DriveVFM(nn.Module):
    """
    Fuses N frozen-VFM feature streams (already extracted upstream -- this
    module does NOT run DINOv3/SigLIP2/SAM/Depth-Anything-V2 itself, it
    consumes their output tokens) into one "unified driving token" sequence.

    forward:
        features: dict[str, Tensor], each [B, N_tokens_source, source_dim]
        -> per-source ModalityAdapter -> concat along token axis
        -> n_fusion_layers x self-attention fusion blocks
        -> unified_tokens: [B, sum(N_tokens_source), model_dim]
    """

    def __init__(self, cfg: DriveVFMConfig):
        super().__init__()
        self.cfg = cfg
        self.adapters = nn.ModuleDict({
            name: ModalityAdapter(dim, cfg.model_dim) for name, dim in cfg.source_dims.items()
        })
        self.fusion_layers = nn.ModuleList([
            _FusionBlock(cfg.model_dim, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.n_fusion_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.model_dim)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert set(features.keys()) == set(self.cfg.source_dims.keys()), (
            f"expected sources {sorted(self.cfg.source_dims)}, got {sorted(features)}"
        )
        adapted = [self.adapters[name](feat) for name, feat in features.items()]  # each [B, N_i, D]
        tokens = torch.cat(adapted, dim=1)                                         # [B, sum(N_i), D]
        for layer in self.fusion_layers:
            tokens = layer(tokens)
        return self.final_norm(tokens)                                             # [B, sum(N_i), D]


# --------------------------------------------------------------------------- #
# 3. Privileged closed-loop PPO teacher
# --------------------------------------------------------------------------- #
class PrivilegedTeacherPolicy(nn.Module):
    """
    Gaussian policy + value head operating on [pooled unified driving
    tokens || privileged ground-truth BEV/occupancy summary]. Privileged
    state is only available in closed-loop simulation, which is exactly
    what the teacher exploits and the deployed student cannot.
    """

    def __init__(self, token_dim: int, privileged_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = token_dim + privileged_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.action_mean = nn.Linear(hidden, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)
        self.value_head = nn.Linear(hidden, 1)

    def _pool(self, unified_tokens: torch.Tensor) -> torch.Tensor:
        return unified_tokens.mean(dim=1)  # [B, D] simple mean-pool over tokens

    def forward(self, unified_tokens: torch.Tensor, privileged_state: torch.Tensor):
        """
        unified_tokens:    [B, T, token_dim]
        privileged_state:  [B, privileged_dim]  (ground-truth BEV/occupancy summary)

        Returns: (dist: torch.distributions.Normal, value: [B, 1])
        """
        pooled = self._pool(unified_tokens)                       # [B, token_dim]
        h = self.trunk(torch.cat([pooled, privileged_state], dim=-1))  # [B, hidden]
        mean = self.action_mean(h)                                 # [B, action_dim]
        std = self.action_log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        value = self.value_head(h)                                 # [B, 1]
        return dist, value

    def act(self, unified_tokens: torch.Tensor, privileged_state: torch.Tensor):
        dist, value = self.forward(unified_tokens, privileged_state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value.squeeze(-1)


def ppo_clipped_update(
    policy: PrivilegedTeacherPolicy,
    unified_tokens: torch.Tensor,
    privileged_state: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    One PPO clipped-surrogate update step for the privileged teacher.

    L = -E[min(r * A, clip(r, 1-eps, 1+eps) * A)]
      + value_coef * MSE(V, returns)
      - entropy_coef * H[pi]

    where r = exp(new_log_prob - old_log_prob) is the probability ratio.
    """
    dist, value = policy(unified_tokens, privileged_state)
    new_log_probs = dist.log_prob(actions).sum(dim=-1)             # [B]

    ratio = torch.exp(new_log_probs - old_log_probs)                # [B]
    adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    surr1 = ratio * adv_norm
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_norm
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(value.squeeze(-1), returns)
    entropy = dist.entropy().sum(dim=-1).mean()

    total = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return {
        "ppo_loss": total,
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy.detach(),
    }


# --------------------------------------------------------------------------- #
# 4. Camera-only deployed student + goal-conditioned distillation
# --------------------------------------------------------------------------- #
class CameraOnlyStudentPolicy(nn.Module):
    """
    Gaussian policy over [pooled unified driving tokens || goal/navigation
    embedding] ONLY -- no privileged state, matching what's actually
    available on the deployed vehicle.
    """

    def __init__(self, token_dim: int, goal_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        in_dim = token_dim + goal_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.action_mean = nn.Linear(hidden, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def _pool(self, unified_tokens: torch.Tensor) -> torch.Tensor:
        return unified_tokens.mean(dim=1)

    def forward(self, unified_tokens: torch.Tensor, goal_embed: torch.Tensor):
        pooled = self._pool(unified_tokens)
        h = self.trunk(torch.cat([pooled, goal_embed], dim=-1))
        mean = self.action_mean(h)
        std = self.action_log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def sample_k(self, unified_tokens: torch.Tensor, goal_embed: torch.Tensor, k: int) -> torch.Tensor:
        """Draws K candidate actions per batch element for test-time search."""
        dist = self.forward(unified_tokens, goal_embed)             # batch_shape [B, action_dim]
        samples = dist.sample((k,))                                  # [K, B, action_dim]
        return samples.permute(1, 0, 2)                               # [B, K, action_dim]


def distillation_loss(
    student: CameraOnlyStudentPolicy,
    teacher_dist: torch.distributions.Normal,
    unified_tokens: torch.Tensor,
    goal_embed: torch.Tensor,
) -> torch.Tensor:
    """
    Goal-conditioned distillation: KL(teacher || student) evaluated on the
    teacher's OWN rollout states -- including states reached under
    augmented goals the human logs never contained, which is what lets the
    student learn from more than "what a human happened to drive".
    """
    student_dist = student(unified_tokens, goal_embed)
    kl = torch.distributions.kl_divergence(teacher_dist, student_dist)  # [B, action_dim]
    return kl.sum(dim=-1).mean()


# --------------------------------------------------------------------------- #
# 5. Value-guided test-time action search
# --------------------------------------------------------------------------- #
def value_guided_action_search(
    student: CameraOnlyStudentPolicy,
    teacher: PrivilegedTeacherPolicy,
    unified_tokens: torch.Tensor,
    goal_embed: torch.Tensor,
    privileged_state_estimate: torch.Tensor,
    k: int = 8,
) -> torch.Tensor:
    """
    Inference-time trick: sample K candidate actions from the (cheap,
    camera-only) student policy, score each with the teacher's value head
    using an estimated/approximate privileged state, and return the
    highest-scoring action per batch element. Trades K forward passes
    through the (small) value head for improved action quality without
    any additional training.

    Returns: [B, action_dim] the selected action per batch element.
    """
    b = unified_tokens.shape[0]
    candidates = student.sample_k(unified_tokens, goal_embed, k)        # [B, K, action_dim]

    # Repeat context K times so the teacher's value head can score each
    # candidate action in one batched forward pass.
    tokens_rep = unified_tokens.unsqueeze(1).expand(-1, k, -1, -1).reshape(
        b * k, unified_tokens.shape[1], unified_tokens.shape[2]
    )
    priv_rep = privileged_state_estimate.unsqueeze(1).expand(-1, k, -1).reshape(b * k, -1)

    with torch.no_grad():
        _, values = teacher(tokens_rep, priv_rep)                        # [B*K, 1]
    values = values.view(b, k)                                            # [B, K]

    best_idx = values.argmax(dim=-1)                                      # [B]
    selected = candidates[torch.arange(b, device=candidates.device), best_idx]  # [B, action_dim]
    return selected
