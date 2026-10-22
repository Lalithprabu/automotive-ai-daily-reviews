"""LoRA building blocks (the core of the Day 48 review).

LoRA freezes a pretrained weight W0 (out x in) and learns a low-rank update:

    y = W0 x + b + (alpha / r) * B (A x)          A: (r x in), B: (out x r)

B is zero-initialised, so at round 0 the adapted model is *exactly* the
pretrained model.  Only A and B are trained and -- in the federated setting --
only A and B ever leave the vehicle.  Payload per linear layer drops from
in*out to r*(in+out) floats.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen nn.Linear + trainable low-rank adapter."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.r = r
        self.scaling = alpha / r
        # A: Kaiming-uniform (as in the LoRA paper's reference code), B: zeros
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in)  ->  (..., out)
        low_rank = (x @ self.lora_A.t()) @ self.lora_B.t()  # (..., r) -> (..., out)
        return self.base(x) + self.scaling * low_rank

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """W0 + (alpha/r) B A  -- what you would deploy after training."""
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)


def apply_lora(model: nn.Module, r: int = 8, alpha: float = 16.0) -> nn.Module:
    """Wrap EVERY nn.Linear in `model` with LoRA, freeze everything else.

    (The paper applies LoRA to all linear layers of Chronos-2, including the
    output head; this mirrors that.)
    """
    def _wrap(parent: nn.Module):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, r, alpha))
            else:
                _wrap(child)

    _wrap(model)
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
    return model


def trainable_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """The exact tensors a vehicle uploads: every trainable parameter.

    LoRA model -> only the A/B matrices.  Full-FT model -> all parameters.
    """
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in state.items():
            params[n].copy_(t)


def payload_numel(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def payload_bytes(model: nn.Module, bytes_per_param: int = 4) -> int:
    """Uplink bytes per vehicle per communication round (fp32 by default)."""
    return payload_numel(model) * bytes_per_param
