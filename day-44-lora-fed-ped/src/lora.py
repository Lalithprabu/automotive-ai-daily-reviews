"""LoRA: low-rank adaptation of frozen linear layers.

y = x W^T + b + (alpha/r) * x A^T B^T      W:(out,in) frozen, A:(r,in), B:(out,r) trainable
B is zero-initialised so the adapted model equals the base model at step 0.
"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):  # x: (..., in)
        return self.base(x) + self.scale * (x @ self.A.t()) @ self.B.t()  # (..., out)


def inject_lora(model: nn.Module, targets, rank: int, alpha: float) -> int:
    """Replace every nn.Linear whose attribute name is in `targets` by a LoRALinear."""
    n = 0
    for _, mod in list(model.named_modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(mod, name, LoRALinear(child, rank, alpha))
                n += 1
    return n


def freeze_all_but_lora(model: nn.Module, keep=("head",)):
    """Freeze everything except LoRA A/B and (small) modules named in `keep`."""
    for n, p in model.named_parameters():
        p.requires_grad = n.endswith(".A") or n.endswith(".B") or any(k in n for k in keep)


def trainable_state(model: nn.Module) -> dict:
    """The only tensors a vehicle must transmit."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_state(model: nn.Module, state: dict):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            params[n].copy_(v)


def n_params(state: dict) -> int:
    return sum(v.numel() for v in state.values())
