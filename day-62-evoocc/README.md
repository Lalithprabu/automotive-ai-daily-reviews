# Day 62 - EvoOcc: Fast-Slow Evolutionary Occupancy Prediction (toy re-implementation)

Paper: *Fast-Slow Evolutionary Occupancy Prediction via Controlled Dynamics* - arXiv [2611.02499](https://arxiv.org/abs/2611.02499) (submitted 2026-11-02, NeurIPS 2026; Yang, Zhou, Yan, Yang, Pan, Tang, He, Han, Xu, Pu). **No official code found; this is an independent re-implementation from the abstract + a third-party summary. Layer sizes, Euler form and control-signal details are this repo's choices.**

## Folder structure
```
day-62-evoocc/
  src/warp.py        # ego-motion affine warp (ego state evolution)
  src/evoocc.py      # AnchorEncoder, DynamicsField, EvoOcc (core model)
  src/data.py        # synthetic streaming occupancy world + fast/slow predictor stand-ins
  train.py           # trains control model + time-only ablation, evaluates per delay step
  simulate.py        # renders the 6-panel simulation (GIF/MP4, or --live window)
  tests/test_model.py  # 8 tests
  config.yaml  requirements.txt  checkpoints/  assets/
```
Run: `pip install -r requirements.txt && pytest -q && python train.py --steps 800 && python simulate.py`

## What the paper reports (Occ3D-nuScenes; numbers from a third-party summary of the paper)
mIoU 40.3 (EvoOcc) vs 33.4 (fast) vs 38.3 (slow reference); +4.3-6.9 mIoU over fast systems; 2.0 s delay: 33.7 vs warped-delayed-slow 20.0 (mIoU_D); latency 101.8 ms vs 216.6 ms (slow alone); EvoOcc adds 23.2 ms / 0.061 TFLOPs.

## What this repo measures (synthetic world, NOT nuScenes, 1 seed, 300 held-out scenes, 125,859 params)
mIoU over static+dynamic classes by delay step (avg step ~0.5 s):

| step | fast | warped slow | EvoOcc | EvoOcc time-only |
|---|---|---|---|---|
| 1 | 0.433 | 0.614 | 0.917 | 0.914 |
| 2 | 0.435 | 0.491 | 0.909 | 0.898 |
| 4 (~2 s) | 0.432 | 0.361 | 0.893 | 0.877 |
| 6 | 0.431 | 0.269 | 0.871 | 0.851 |
| mean 0-6 | 0.432 | 0.495 | 0.910 | 0.899 |

Dynamic-class IoU at step 4: fast 0.228, warped slow 0.104, EvoOcc 0.837. **Caveat:** the toy FAST predictor is i.i.d. noise + blur, so much of the gap is denoising via the slow anchor; absolute gaps are far larger than the paper's. The qualitative trends (warp-only collapses with delay, control beats time-only by a small margin) match the paper; magnitudes are not comparable.

## Core code (src/evoocc.py)
```python
"""EvoOcc-style fast-slow occupancy refinement (re-implementation, NOT the authors' code).

Paper (arXiv 2611.02499, NeurIPS 2026): treat streaming occupancy prediction as a
continuous-time *controlled dynamical system*.  A frozen FAST predictor emits logits every
frame; a frozen SLOW (accurate) predictor emits logits rarely and with a delay.  EvoOcc
carries the slow system's detail forward in a latent 'state anchor' and evolves it:

    1. ego state evolution   : deterministic coordinate shift  -> warp(state, ego motion)
    2. scene state evolution : real geometric/semantic change  -> Euler step of
                               ds/dt = F(s, u),   u = f_fast(t) - warp(f_fast(t-1))   (control)

Everything below the paper's abstract-level description (layer sizes, the Euler form,
the exact control signal, GroupNorm/FiLM choices) is THIS REPO's engineering choice.
Tensor shapes: B batch, K classes, H,W grid, C hidden channels.
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .warp import warp_bev


def conv_block(cin, cout, k=3, d=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=d * (k // 2), dilation=d),
        nn.GroupNorm(4, cout), nn.GELU())


class FastEncoder(nn.Module):
    """Embeds fast logits [B,K,H,W] -> fast feature f [B,C,H,W]."""
    def __init__(self, k: int, c: int):
        super().__init__()
        self.net = nn.Sequential(conv_block(k, c), conv_block(c, c))

    def forward(self, logits):
        return self.net(logits)


class AnchorEncoder(nn.Module):
    """Slow+fast logits at the SAME timestamp -> state anchor s0 [B,C,H,W].
    The slow-minus-fast difference isolates 'what the fast system is missing'."""
    def __init__(self, k: int, c: int):
        super().__init__()
        self.fast_enc = FastEncoder(k, c)
        self.diff_enc = nn.Sequential(conv_block(2 * k, c), conv_block(c, c))
        self.fuse = conv_block(2 * c, c)

    def forward(self, slow_logits, fast_logits):
        diff = torch.cat([slow_logits - fast_logits, slow_logits], dim=1)   # [B,2K,H,W]
        return self.fuse(torch.cat([self.fast_enc(fast_logits), self.diff_enc(diff)], 1))


class DynamicsField(nn.Module):
    """F(s, u, dt): the vector field of the controlled system, ds/dt = F(s, u).
    Dilated convolutions give a ~19-cell receptive field so objects that moved several
    cells between frames can still be re-associated.  dt enters through FiLM."""
    def __init__(self, c: int):
        super().__init__()
        self.inp = conv_block(2 * c, c)
        self.mid = nn.Sequential(conv_block(c, c, 3, 1), conv_block(c, c, 3, 2),
                                 conv_block(c, c, 3, 4))
        self.film = nn.Linear(1, 2 * c)
        self.out = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # start as identity flow

    def forward(self, s, u, dt):
        h = self.inp(torch.cat([s, u], dim=1))
        g, b = self.film(dt.view(-1, 1)).chunk(2, dim=-1)               # [B,C] each
        h = h * (1 + g[:, :, None, None]) + b[:, :, None, None]
        return self.out(self.mid(h))


class EvoOcc(nn.Module):
    def __init__(self, num_classes=3, hidden=32, extent_m=48.0, use_control=True):
        super().__init__()
        self.k, self.c, self.extent_m, self.use_control = num_classes, hidden, extent_m, use_control
        self.fast_enc = FastEncoder(num_classes, hidden)
        self.anchor_enc = AnchorEncoder(num_classes, hidden)
        self.dyn = DynamicsField(hidden)
        self.time_ctrl = nn.Linear(1, hidden)        # used when use_control=False (time-only ablation)
        self.decoder = nn.Sequential(conv_block(2 * hidden, hidden), conv_block(hidden, hidden),
                                     nn.Conv2d(hidden, num_classes, 1))
        nn.init.zeros_(self.decoder[-1].weight); nn.init.zeros_(self.decoder[-1].bias)

    # -- one step of the controlled system ---------------------------------------------
    def step(self, s_prev, f_prev, fast_logits, ego_delta, dt):
        """Advance the latent state from frame k-1 to k.
        s_prev,f_prev: [B,C,H,W]; fast_logits: [B,K,H,W]; ego_delta: [B,3]; dt: [B]."""
        f_cur = self.fast_enc(fast_logits)                              # [B,C,H,W]
        s_w = warp_bev(s_prev, ego_delta, self.extent_m)               # (1) ego evolution
        f_w = warp_bev(f_prev, ego_delta, self.extent_m)
        if self.use_control:
            u = f_cur - f_w                                             # control signal
        else:                                                           # ablation: elapsed time only
            u = self.time_ctrl(dt.view(-1, 1))[:, :, None, None].expand_as(f_cur)
        s_new = s_w + dt.view(-1, 1, 1, 1) * self.dyn(s_w, u, dt)       # (2) Euler step of ds/dt
        return s_new, f_cur

    def decode(self, s, f, fast_logits):
        """Residual correction on top of the fast logits."""
        return fast_logits + self.decoder(torch.cat([s, f], dim=1))

    def forward(self, slow0, fast_seq, ego_deltas, dts):
        """Streaming rollout.
        slow0:      [B,K,H,W]  slow logits for frame 0 (arrive late, at frame T)
        fast_seq:   [B,T+1,K,H,W] fast logits for frames 0..T
        ego_deltas: [B,T,3]   pose change frame k-1 -> k
        dts:        [B,T]     seconds between frames (irregular allowed)
        returns     [B,T+1,K,H,W] refined logits (frame 0 = anchor-corrected).
        """
        s = self.anchor_enc(slow0, fast_seq[:, 0])
        f = self.fast_enc(fast_seq[:, 0])
        outs = [self.decode(s, f, fast_seq[:, 0])]
        for k in range(1, fast_seq.shape[1]):
            s, f = self.step(s, f, fast_seq[:, k], ego_deltas[:, k - 1], dts[:, k - 1])
            outs.append(self.decode(s, f, fast_seq[:, k]))
        return torch.stack(outs, dim=1)
```
