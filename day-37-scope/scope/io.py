import numpy as np, torch
from .backbone import Denoiser, sample
from .precision import PrecisionHead
from .data import make_data


def load(path="checkpoint.pt"):
    ck = torch.load(path, weights_only=False)
    cfg = ck["cfg"]
    b, h = cfg["backbone"], cfg["head"]
    model = Denoiser(b["hidden"], b["depth"], b["sigma_data"]); model.load_state_dict(ck["model"]); model.eval()
    head = PrecisionHead(h["sigma0"], h["hidden"], h["bandwidth"], h["rank"]); head.load_state_dict(ck["head"]); head.eval()
    for p in list(model.parameters()) + list(head.parameters()): p.requires_grad_(False)
    return cfg, model, head, ck


def nominal(model, cfg, cond, mode):
    s = cfg["sampler"]
    return sample(model, cond, mode, None, s["steps"], s["sigma_max"], s["sigma_min"])
