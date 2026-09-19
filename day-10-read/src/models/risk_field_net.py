"""RiskFieldNetwork: the paper's central contribution.

Classical planners wrap each obstacle in a hand-prescribed "safety
bubble" -- usually an isotropic Gaussian or a fixed-radius disk -- and
re-discretize a cost grid every time perception updates. READ instead
*learns* a continuous, differentiable, queryable spatiotemporal risk
field r(x, y, t) directly from data: any point in space-time can be
probed for its risk without re-running perception or building a grid,
and because the field is differentiable, a candidate trajectory can be
refined by simply descending its gradient.

This file is kept intentionally close to the verified reference
implementation from the isolated build session -- it is not redesigned
here, only reproduced.
"""

import torch
import torch.nn as nn

from src.utils.positional_encoding import FourierPositionalEncoding


class RiskFieldNetwork(nn.Module):
    """Coordinate-conditioned implicit field: (x, y, t) + scene -> risk.

    Classical safety fields hand-prescribe a risk shape (usually an
    isotropic Gaussian) around each obstacle. READ instead *learns* a
    continuous spatiotemporal risk field directly from data: given a
    query point in space-time, the network cross-attends to the scene's
    map + agent context tokens and regresses a scalar risk value.

    Input:
        query_xyt     # Shape: [B, Q, 3]   (Q query points per batch elem)
        scene_tokens  # Shape: [B, N, D]   (from SceneEncoder)
    Output:
        risk          # Shape: [B, Q]      scalar risk in [0, 1] per query
    """

    def __init__(self, hidden_dim=128, num_freqs=8, num_cross_layers=2, num_heads=4):
        super().__init__()
        self.pos_enc = FourierPositionalEncoding(num_input_dims=3, num_freqs=num_freqs)
        self.query_proj = nn.Linear(self.pos_enc.output_dim, hidden_dim)
        self.cross_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_cross_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_cross_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                          nn.Linear(hidden_dim * 2, hidden_dim))
            for _ in range(num_cross_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_cross_layers)])
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_xyt, scene_tokens):
        encoded = self.pos_enc(query_xyt)
        q = self.query_proj(encoded)
        for attn, norm, ffn, ffn_norm in zip(self.cross_layers, self.norms, self.ffns, self.ffn_norms):
            attended, _ = attn(query=q, key=scene_tokens, value=scene_tokens)
            q = norm(q + attended)
            q = ffn_norm(q + ffn(q))
        risk_logits = self.risk_head(q).squeeze(-1)
        return torch.sigmoid(risk_logits)


@torch.enable_grad()
def refine_trajectory_by_risk_descent(risk_field, scene_tokens, init_trajectory_xy,
                                       timestamps, num_steps=20, step_size=0.05):
    """Differentiable trajectory refinement: gradient-descend the (x, y)
    waypoints against the learned risk field, holding the field fixed.

    This is the mechanism that replaces "re-run perception, re-discretize
    a cost grid, re-search" with a handful of gradient steps: because
    RiskFieldNetwork is a differentiable function of (x, y, t), we can
    backprop risk straight into the trajectory's waypoint coordinates.

    Args:
        risk_field: a RiskFieldNetwork (weights held fixed here).
        scene_tokens: [B, N, D] context to cross-attend against.
        init_trajectory_xy: [B, Q, 2] candidate waypoints (x, y).
        timestamps: [B, Q] timestamp for each waypoint.
        num_steps: number of gradient-descent steps.
        step_size: gradient step size.
    Returns:
        refined trajectory: [B, Q, 2], detached.
    """
    trajectory = init_trajectory_xy.clone().detach().requires_grad_(True)
    for _ in range(num_steps):
        query_xyt = torch.cat([trajectory, timestamps.unsqueeze(-1)], dim=-1)
        risk = risk_field(query_xyt, scene_tokens).mean(dim=-1)
        grad = torch.autograd.grad(risk.sum(), trajectory)[0]
        trajectory = (trajectory - step_size * grad).detach().requires_grad_(True)
    return trajectory.detach()
