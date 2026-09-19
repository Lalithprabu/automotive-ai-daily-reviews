"""
BeamTransFuser -- reconstruction of the core architecture from:

  "BeamTransFuser: Robust Beam Prediction for V2X Networks with Multi-Modal
  Sensing" (Chen Shang, Dinh Thai Hoang, Diep N. Nguyen, Jiadong Yu.
  arXiv:2609.10200, to appear IEEE GLOBECOM 2026)

A hierarchical Transformer-based architecture that progressively fuses
camera, LiDAR, radar, and GPS observations at a roadside unit (RSU) for
robust mmWave beam prediction, with a generative ModalityImputer that
reconstructs missing-modality tokens when a sensor drops out entirely.

IMPORTANT BUG FIX BAKED IN (do not reintroduce):
  An early draft of CrossModalFusionBlock built a `key_padding_mask` from
  each modality's per-sample presence flag, masking out EVERY token
  belonging to a fully-absent modality. But ModalityImputer already writes
  a valid synthesized token into that exact slot upstream. The mask then
  blanked that synthesized token back out -- for any sample where a
  modality was fully absent, its attention row had ZERO valid keys left,
  softmax produced NaN, and training loss went to `nan` on step 1.
  THE FIX: CrossModalFusionBlock (and every attention call inside this
  file) NEVER masks by modality presence. We rely entirely on the imputer
  having already produced a usable token for absent modalities. See
  `tests/test_model.py::test_full_modality_dropout_is_finite_no_nan`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITY_ORDER = ("camera", "lidar", "radar", "gps")


# --------------------------------------------------------------------------
# Per-modality encoders: each maps its raw sensor tensor into a fixed-length
# sequence of tokens in the shared d_model space.
# --------------------------------------------------------------------------
class ConvTokenEncoder(nn.Module):
    """Small conv stem for image-like modalities (camera / LiDAR BEV / radar
    range-doppler map). Reduces spatial resolution with strided convs, then
    adaptive-pools to a fixed `token_grid x token_grid` grid of tokens, each
    projected to `d_model`.
    """

    def __init__(self, in_channels: int, d_model: int, n_tokens: int, base_ch: int = 32):
        super().__init__()
        # n_tokens must be a perfect square (we lay tokens out on a grid)
        self.token_grid = int(round(math.sqrt(n_tokens)))
        assert self.token_grid * self.token_grid == n_tokens, \
            f"n_tokens={n_tokens} must be a perfect square for ConvTokenEncoder"

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.GELU(),
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((self.token_grid, self.token_grid))
        self.proj = nn.Linear(base_ch * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        feat = self.stem(x)                        # (B, 2*base_ch, H/2, W/2)
        feat = self.pool(feat)                      # (B, 2*base_ch, grid, grid)
        b, c, gh, gw = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)     # (B, grid*grid, 2*base_ch)
        tokens = self.proj(tokens)                   # (B, n_tokens, d_model)
        return tokens


class GPSTokenEncoder(nn.Module):
    """MLP encoder for the low-dimensional GPS/kinematics vector
    [x, y, speed, heading]. Produces `n_tokens` tokens via a small MLP that
    expands to n_tokens * d_model and reshapes (n_tokens is typically 1)."""

    def __init__(self, in_dim: int, d_model: int, n_tokens: int, hidden: int = 64):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_tokens * d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        b = x.shape[0]
        out = self.net(x)                              # (B, n_tokens * d_model)
        return out.view(b, self.n_tokens, self.d_model)  # (B, n_tokens, d_model)


# --------------------------------------------------------------------------
# ModalityImputer: generative, cross-attention-based substitution for a
# fully-absent modality, conditioned on whichever modalities ARE present.
# --------------------------------------------------------------------------
class ModalityImputer(nn.Module):
    """For each modality, holds a bank of learnable "query" placeholder
    tokens. When a modality is fully absent for a given sample, its
    placeholder queries cross-attend over the token sequences of the OTHER
    modalities (processed so far) to synthesize a replacement token
    sequence, which is blended in via the per-sample presence mask.

    Crucially this module ALWAYS produces a fully valid (non-NaN) output
    for every sample, and the fusion blocks downstream never re-mask by
    presence -- see the module docstring above for why that matters.
    """

    def __init__(self, d_model: int, n_heads: int, tokens_per_modality: dict, dropout: float = 0.1):
        super().__init__()
        self.tokens_per_modality = tokens_per_modality
        self.queries = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1, n, d_model) * 0.02)
            for m, n in tokens_per_modality.items()
        })
        self.cross_attn = nn.ModuleDict({
            m: nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for m in tokens_per_modality
        })
        self.norm_out = nn.ModuleDict({
            m: nn.LayerNorm(d_model) for m in tokens_per_modality
        })

    def forward(self, tokens: dict, presence: dict) -> dict:
        """
        tokens: {modality: (B, L_m, d_model)}
        presence: {modality: (B,) bool, True = sensor present for that sample}
        returns: {modality: (B, L_m, d_model)} with absent-modality rows
                 replaced by synthesized tokens.
        """
        out = dict(tokens)
        for m in MODALITY_ORDER:
            if m not in tokens:
                continue
            pres = presence[m]                       # (B,)
            if bool(pres.all()):
                # nothing absent for this modality in this batch -- skip the
                # (relatively expensive) cross-attention pass entirely
                continue

            b = tokens[m].shape[0]
            # context = concatenation of every OTHER modality's current
            # tokens (already-imputed ones included, since we process
            # modalities in a fixed order and always write into `out`)
            context_parts = [out[om] for om in MODALITY_ORDER if om != m and om in out]
            context = torch.cat(context_parts, dim=1)   # (B, sum_L_other, d_model)

            query = self.queries[m].expand(b, -1, -1)    # (B, L_m, d_model)
            synth, _ = self.cross_attn[m](query, context, context, need_weights=False)
            synth = self.norm_out[m](synth)               # (B, L_m, d_model)

            mask = pres.view(b, 1, 1).to(synth.dtype)      # 1 = present (keep original)
            out[m] = mask * tokens[m] + (1.0 - mask) * synth
        return out


# --------------------------------------------------------------------------
# CrossModalFusionBlock: progressive cross-attention -> self-attention -> FFN
# --------------------------------------------------------------------------
class CrossModalFusionBlock(nn.Module):
    """One stage of progressive multi-modal fusion over the FULL concatenated
    token sequence (all modalities together).

    Stage 1 "cross-attention": tokens attend across the whole sequence, so
        every modality's tokens gather information from every other
        modality's tokens (and their own).
    Stage 2 "self-attention": a second, independently-weighted attention
        pass that further consolidates the now cross-informed sequence.
    Stage 3 FFN: position-wise feed-forward refinement.

    Each stage is a residual + pre-LayerNorm sublayer, matching a standard
    Transformer block. NO key_padding_mask is ever built or applied here --
    see the module docstring at the top of this file for why.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_total, d_model) -- all modalities' tokens concatenated.
        # NOTE: no attn_mask / key_padding_mask is passed to either
        # attention call, by design (see module docstring).
        h = self.norm1(x)
        attn_out, _ = self.cross_attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)

        h = self.norm2(x)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)

        h = self.norm3(x)
        x = x + self.dropout(self.ffn(h))
        return x


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------
class BeamTransFuser(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_fusion_blocks: int = 4,
        ffn_hidden: int = 256,
        dropout: float = 0.1,
        num_beams: int = 64,
        tokens_per_modality: dict | None = None,
        camera_channels: int = 3,
        lidar_channels: int = 1,
        radar_channels: int = 1,
        gps_dim: int = 4,
    ):
        super().__init__()
        tokens_per_modality = tokens_per_modality or {
            "camera": 8, "lidar": 8, "radar": 4, "gps": 1,
        }
        self.tokens_per_modality = tokens_per_modality
        self.d_model = d_model

        self.encoders = nn.ModuleDict({
            "camera": ConvTokenEncoder(camera_channels, d_model, tokens_per_modality["camera"]),
            "lidar": ConvTokenEncoder(lidar_channels, d_model, tokens_per_modality["lidar"]),
            "radar": ConvTokenEncoder(radar_channels, d_model, tokens_per_modality["radar"]),
            "gps": GPSTokenEncoder(gps_dim, d_model, tokens_per_modality["gps"]),
        })

        # learnable modality-type embeddings (added to every token of that
        # modality so the fusion blocks can tell modalities apart) and
        # per-position embeddings within each modality's token block.
        self.modality_embed = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1, 1, d_model) * 0.02) for m in MODALITY_ORDER
        })
        self.position_embed = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1, n, d_model) * 0.02)
            for m, n in tokens_per_modality.items()
        })

        self.imputer = ModalityImputer(d_model, n_heads, tokens_per_modality, dropout=dropout)

        self.fusion_blocks = nn.ModuleList([
            CrossModalFusionBlock(d_model, n_heads, ffn_hidden, dropout=dropout)
            for _ in range(n_fusion_blocks)
        ])

        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_beams),
        )

    def encode_modalities(self, camera, lidar, radar, gps) -> dict:
        return {
            "camera": self.encoders["camera"](camera),
            "lidar": self.encoders["lidar"](lidar),
            "radar": self.encoders["radar"](radar),
            "gps": self.encoders["gps"](gps),
        }

    def forward(self, camera, lidar, radar, gps, presence: dict, return_tokens: bool = False):
        """
        camera: (B, 3, 32, 32)   lidar: (B, 1, 32, 32)   radar: (B, 1, 16, 16)
        gps:    (B, 4)
        presence: {modality: (B,) bool}
        returns: logits (B, num_beams)  [and, optionally, intermediate tokens]
        """
        tokens = self.encode_modalities(camera, lidar, radar, gps)

        # generative imputation for fully-absent modalities
        tokens = self.imputer(tokens, presence)

        # add modality-type + within-modality positional embeddings, then
        # concatenate all modalities into one token sequence
        embedded = []
        for m in MODALITY_ORDER:
            t = tokens[m] + self.modality_embed[m] + self.position_embed[m]
            embedded.append(t)
        x = torch.cat(embedded, dim=1)   # (B, N_total, d_model)

        for block in self.fusion_blocks:
            x = block(x)

        pooled = self.head_norm(x.mean(dim=1))   # (B, d_model)
        logits = self.head(pooled)               # (B, num_beams)

        if return_tokens:
            return logits, x
        return logits


def build_model_from_config(cfg: dict) -> BeamTransFuser:
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    return BeamTransFuser(
        d_model=mcfg["d_model"],
        n_heads=mcfg["n_heads"],
        n_fusion_blocks=mcfg["n_fusion_blocks"],
        ffn_hidden=mcfg["ffn_hidden"],
        dropout=mcfg["dropout"],
        num_beams=mcfg["num_beams"],
        tokens_per_modality=mcfg["tokens_per_modality"],
        camera_channels=dcfg["camera_shape"][0],
        lidar_channels=dcfg["lidar_shape"][0],
        radar_channels=dcfg["radar_shape"][0],
        gps_dim=dcfg["gps_dim"],
    )
