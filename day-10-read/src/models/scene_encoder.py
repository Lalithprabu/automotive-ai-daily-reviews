"""SceneEncoder: BEV-raster CNN + agent-history GRU fused by a shallow
Transformer, producing scene context tokens for the Risk Field Network to
cross-attend against.

READ needs a compact set of "scene tokens" that jointly summarize (a) the
static/semi-static map context (lane geometry, drivable area -- here a
rasterized BEV occupancy/lane grid) and (b) the dynamic agents' recent
motion history. We encode each with a small backbone, then let a shallow
Transformer encoder mix map tokens and agent tokens together so each
token's representation is already scene-contextualized before the risk
field cross-attends to it.
"""

import torch
import torch.nn as nn


class BEVRasterCNN(nn.Module):
    """Small conv backbone over a rasterized BEV grid.

    Input:  [B, C_in, H, W]  (e.g. lane/occupancy channels)
    Output: [B, num_map_tokens, D]  flattened spatial feature map, pooled
            down to a manageable number of tokens.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 128, pooled_size: int = 4):
        super().__init__()
        self.pooled_size = pooled_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # H/2
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # H/4
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),  # H/8
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))

    def forward(self, bev_grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_grid: [B, C_in, H, W]
        Returns:
            map_tokens: [B, pooled_size*pooled_size, hidden_dim]
        """
        feat = self.conv(bev_grid)  # [B, D, H/8, W/8]
        feat = self.pool(feat)  # [B, D, P, P]
        b, d, p1, p2 = feat.shape
        map_tokens = feat.flatten(2).transpose(1, 2)  # [B, P*P, D]
        return map_tokens


class AgentHistoryGRU(nn.Module):
    """Encodes each agent's recent (x, y, heading, speed) history into a
    single per-agent token via a GRU over time.

    Input:  [B, A, T, F_in]  (A agents, T past timesteps, F_in features)
    Output: [B, A, D]        one token per agent
    """

    def __init__(self, in_features: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_dim, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, agent_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_history: [B, A, T, F_in]
        Returns:
            agent_tokens: [B, A, D]
        """
        b, a, t, f = agent_history.shape
        flat = agent_history.reshape(b * a, t, f)  # [B*A, T, F_in]
        _, h_n = self.gru(flat)  # h_n: [1, B*A, D]
        h_n = h_n.squeeze(0)  # [B*A, D]
        agent_tokens = self.out_proj(h_n).reshape(b, a, -1)  # [B, A, D]
        return agent_tokens


class SceneEncoder(nn.Module):
    """Fuses BEV map tokens + agent-history tokens with a shallow
    Transformer encoder into a unified set of scene context tokens.

    Output: scene_tokens [B, N, D] where N = num_map_tokens + A.
    """

    def __init__(
        self,
        bev_channels: int = 3,
        agent_features: int = 4,
        hidden_dim: int = 128,
        pooled_size: int = 4,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bev_cnn = BEVRasterCNN(bev_channels, hidden_dim, pooled_size)
        self.agent_gru = AgentHistoryGRU(agent_features, hidden_dim)

        # Learned type embeddings so the fusion transformer can tell map
        # tokens apart from agent tokens.
        self.map_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.agent_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            activation="gelu",
            batch_first=True,
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

    def forward(self, bev_grid: torch.Tensor, agent_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bev_grid: [B, C_in, H, W]
            agent_history: [B, A, T, F_in]
        Returns:
            scene_tokens: [B, N, D]  N = pooled_size**2 + A
        """
        map_tokens = self.bev_cnn(bev_grid) + self.map_type_embed  # [B, P*P, D]
        agent_tokens = self.agent_gru(agent_history) + self.agent_type_embed  # [B, A, D]

        tokens = torch.cat([map_tokens, agent_tokens], dim=1)  # [B, N, D]
        scene_tokens = self.fusion_transformer(tokens)  # [B, N, D]
        return scene_tokens
