"""Flash Primitive Decoding (reconstruction from the abstract of arXiv 2611.08449).

Idea: an autoregressive (AR) VLM emits a trajectory/box one coordinate token at a
time (L sequential passes). A *drafter* instead predicts ALL coordinates of one
visual primitive (point / box / trajectory) in a single forward pass, with an
output layout chosen by primitive type, dimensionality and length.

NOTE: the paper's drafter internals are not public in the sources I could read;
this is an independent, minimal implementation of the stated idea.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str      # "point" | "box" | "traj2d" | "traj3d"
    dim: int       # coordinates per element (2, 3, 4)
    max_len: int   # max elements (1 for point/box, T for trajectories)


DEFAULT_SPECS = {
    "point": PrimitiveSpec("point", 2, 1),
    "box": PrimitiveSpec("box", 4, 1),
    "traj2d": PrimitiveSpec("traj2d", 2, 16),
    "traj3d": PrimitiveSpec("traj3d", 3, 16),
}


class VFlashDrafter(nn.Module):
    """One-pass primitive drafter.

    ctx:  [B, N, D] context tokens from a (frozen) vision-language backbone.
    out:  [B, L, dim] coordinates (L = requested length <= spec.max_len), plus
          a validity-aware parametrisation so boxes are always well-formed.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 specs: Optional[Dict[str, PrimitiveSpec]] = None):
        super().__init__()
        self.specs = specs or DEFAULT_SPECS
        self.types = list(self.specs)
        self.type_emb = nn.Embedding(len(self.types), d_model)
        self.max_len = max(s.max_len for s in self.specs.values())
        # Learned slot queries: slot i will become element i of the primitive.
        self.slot_q = nn.Parameter(torch.randn(1, self.max_len, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        # One lightweight head per primitive type -> layout adapts to type/dim.
        self.heads = nn.ModuleDict({n: nn.Linear(d_model, s.dim) for n, s in self.specs.items()})

    def forward(self, ctx: torch.Tensor, ptype: str, length: Optional[int] = None) -> torch.Tensor:
        spec = self.specs[ptype]
        L = min(length or spec.max_len, spec.max_len)
        b = ctx.shape[0]
        t = torch.tensor([self.types.index(ptype)], device=ctx.device)
        q = self.slot_q[:, :L].expand(b, -1, -1) + self.type_emb(t)[:, None]   # [B,L,D]
        # Slots attend to each other (bidirectional => joint prediction) and to ctx.
        h = self.decoder(q, ctx)                                               # [B,L,D]
        y = self.heads[ptype](h)                                               # [B,L,dim]
        if ptype == "box":
            # (cx, cy, w, h): width/height forced positive -> structurally valid.
            y = torch.cat([y[..., :2], nn.functional.softplus(y[..., 2:])], dim=-1)
        elif ptype.startswith("traj"):
            # Predict per-step displacements, integrate -> smooth, continuous path.
            y = torch.cumsum(y, dim=1)
        return y


class ARBaseline(nn.Module):
    """Autoregressive coordinate decoder: L sequential passes (the 'old way')."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 2, dim: int = 2, max_len: int = 16):
        super().__init__()
        self.inp = nn.Linear(dim, d_model)
        self.start = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.out = nn.Linear(d_model, dim)
        self.pos = nn.Parameter(torch.randn(1, max_len + 1, d_model) * 0.02)
        self.dim, self.max_len = dim, max_len

    def forward(self, ctx, target=None, length=None):
        """Teacher-forced if target given ([B,L,dim]); else sequential rollout."""
        L = length or self.max_len
        b = ctx.shape[0]
        if target is not None:
            x = torch.cat([self.start.expand(b, -1, -1), self.inp(target[:, :-1])], 1)
            x = x + self.pos[:, :x.shape[1]]
            mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(ctx.device)
            return self.out(self.decoder(x, ctx, tgt_mask=mask))
        seq = self.start.expand(b, -1, -1)
        outs = []
        for i in range(L):                      # L sequential forward passes
            x = seq + self.pos[:, :seq.shape[1]]
            mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(ctx.device)
            step = self.out(self.decoder(x, ctx, tgt_mask=mask)[:, -1:])
            outs.append(step)
            seq = torch.cat([seq, self.inp(step)], 1)
        return torch.cat(outs, 1)
