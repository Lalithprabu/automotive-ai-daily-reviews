"""
PlanningExpert -- Qwen-Drive-1.0's core novel piece: a diffusion-transformer
trajectory generator that does NOT run as a bolt-on planner behind a frozen
perception network. Instead, at every one of its 32 layers it directly
cross-attends into the *cached* key/value activations produced by the
backbone Vision-Language Model's own attention layers -- so the planner
reasons with the exact same scene representation the VLM uses for 3D
perception and visual question answering.

Reference implementation inspired by:
  "Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation
   Model for Autonomous Driving" (arXiv:2609.00111, Aug 2026)
"""
import torch
import torch.nn as nn

from src.models.ada_ln import AdaLNModulation, modulate
from src.models.planning_expert_layer import PlanningExpertLayer
from src.models.time_embed import sinusoidal_time_embed
from src.models.vlm_kv_cache import VLMKVCache


class PlanningExpert(nn.Module):
    """
    trajectory_tokens (noisy waypoints at flow-matching time t)
        -> N_LAYERS PlanningExpertLayers, layer i conditioned on
           VLMKVCache[i // LAYERS_PER_CACHE]
        -> linear head -> predicted clean trajectory (x-prediction)

    Waypoint tokens are (x, y, heading) at 10 Hz for HORIZON_SEC seconds,
    i.e. N_WAYPOINTS = HORIZON_SEC * 10 = 50 for the paper's 5-second horizon.
    """

    def __init__(
        self,
        model_dim: int = 1024,
        n_layers: int = 32,
        n_heads: int = 16,
        n_caches: int = 8,
        cond_dim: int = 1024,
        waypoint_dim: int = 3,       # (x, y, heading)
        n_waypoints: int = 50,       # 5s @ 10Hz
    ):
        super().__init__()
        assert n_layers % n_caches == 0, "n_layers must divide evenly across n_caches"
        self.layers_per_cache = n_layers // n_caches
        self.n_caches = n_caches
        self.n_waypoints = n_waypoints

        # Project noisy (x, y, heading) waypoints + a learned per-step
        # positional embedding into the model's token dimension.
        self.waypoint_in = nn.Linear(waypoint_dim, model_dim)
        self.step_pos_embed = nn.Parameter(torch.zeros(1, n_waypoints, model_dim))
        nn.init.trunc_normal_(self.step_pos_embed, std=0.02)

        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.layers = nn.ModuleList(
            [PlanningExpertLayer(model_dim, n_heads, cond_dim) for _ in range(n_layers)]
        )

        self.final_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.final_ada_ln = AdaLNModulation(cond_dim, model_dim)
        self.waypoint_out = nn.Linear(model_dim, waypoint_dim)
        nn.init.zeros_(self.waypoint_out.weight)
        nn.init.zeros_(self.waypoint_out.bias)

        self._time_embed_dim = model_dim

    def forward(
        self,
        noisy_waypoints: torch.Tensor,   # [B, n_waypoints, 3]  (x, y, heading) at flow time t
        t: torch.Tensor,                 # [B]  flow-matching timestep in [0, 1]
        kv_cache: VLMKVCache,            # 8 cached (K, V) pairs from the frozen VLM
        instruction_embed: torch.Tensor,  # [B, cond_dim] pooled language-instruction embedding
        ego_state_embed: torch.Tensor,    # [B, cond_dim] pooled ego speed/heading/history embedding
    ) -> torch.Tensor:
        assert len(kv_cache) == self.n_caches, (
            f"expected {self.n_caches} cached KV groups, got {len(kv_cache)}"
        )

        # ---- Build conditioning vector: time + instruction + ego-state ----
        t_embed = sinusoidal_time_embed(t, self._time_embed_dim)   # [B, D]
        t_embed = self.time_mlp(t_embed)                            # [B, cond_dim]
        cond = t_embed + instruction_embed + ego_state_embed        # [B, cond_dim]

        # ---- Tokenize the noisy trajectory ----
        x = self.waypoint_in(noisy_waypoints) + self.step_pos_embed  # [B, n_waypoints, D]

        # ---- Run through 32 layers, cycling through the 8 caches ----
        for i, layer in enumerate(self.layers):
            cache_idx = i // self.layers_per_cache
            x = layer(x, kv_cache.keys[cache_idx], kv_cache.values[cache_idx], cond)

        # ---- Final AdaLN + projection back to (x, y, heading) ----
        shift, scale, _, _, _, _ = self.final_ada_ln(cond)
        x = modulate(self.final_norm(x), shift, scale)
        x1_pred = self.waypoint_out(x)                               # [B, n_waypoints, 3]
        return x1_pred
