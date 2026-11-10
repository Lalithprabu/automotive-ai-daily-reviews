"""CoMET-style Common-Mode Error Tracking (re-implementation from the abstract).

A roadside unit (RSU) message carries N boxes that were all produced with the SAME pose, so a
pose error shifts every box by the SAME 2-D translation (a "common mode" error).  We:

  1. match the RSU boxes with the ego detections of the objects both sensors see
     (mutual nearest neighbours, gated by the current error estimate),
  2. turn each matched pair into a measurement of the shared offset,
  3. fuse the measurement into a *per-sender* Kalman filter (random-walk offset),
  4. subtract the posterior offset from EVERY RSU box -- including boxes the ego car cannot see.

When no pair is matched the filter simply carries its estimate forward (and its uncertainty
grows).  The estimator has NO learned parameters and needs no ground truth at inference.

State per sender s:   x_s in R^2 (offset),  P_s scalar variance (isotropic).
    predict : P <- P + q * dt
    measure : z = coordinate-wise median_k (road_j - ego_i)   (robust to a bad pair)
              R = (pi/2) * r0 / n_pairs                         (variance of a median of n)
    update  : K = P / (P + R);  x <- x + K (z - x);  P <- (1 - K) P
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .matching import mutual_nn_pairs


@dataclass
class CoMETConfig:
    num_senders: int = 1
    q: float = 0.05          # process noise: offset random-walk variance [m^2 / s]
    r0: float = 0.20         # variance of ONE matched-pair offset measurement [m^2]
    p0: float = 4.0          # prior variance of the offset [m^2] (~2 m std, unknown at start)
    gate: float = 2.5        # base matching radius [m]
    gate_sigma_k: float = 1.5  # extra gate = k * sqrt(P) while the offset is still uncertain
    use_filter: bool = True  # False -> "current message only" ablation (no temporal memory)


class CoMETTracker(nn.Module):
    """Batched per-sender Kalman tracker of the shared roadside pose offset.

    All state lives in *buffers* (not parameters): ``sum(p.numel() for p in parameters()) == 0``.
    """

    def __init__(self, cfg: CoMETConfig | None = None):
        super().__init__()
        self.cfg = cfg or CoMETConfig()
        S = self.cfg.num_senders
        self.register_buffer("x", torch.zeros(S, 2))                      # (S, 2) offset estimate
        self.register_buffer("P", torch.full((S,), self.cfg.p0))          # (S,)   its variance
        self.register_buffer("n_matched", torch.zeros(S, dtype=torch.long))  # last pair count

    # ------------------------------------------------------------------ state handling
    def reset(self) -> None:
        self.x.zero_()
        self.P.fill_(self.cfg.p0)
        self.n_matched.zero_()

    # ------------------------------------------------------------------ one message step
    @torch.no_grad()
    def step(
        self,
        ego_xy: torch.Tensor,       # (S, N, 2)
        road_xy: torch.Tensor,      # (S, M, 2) raw (uncorrected) roadside boxes in ego frame
        ego_mask: torch.Tensor,     # (S, N) bool
        road_mask: torch.Tensor,    # (S, M) bool
        dt: float | torch.Tensor = 0.1,
    ) -> dict[str, torch.Tensor]:
        c = self.cfg
        S = ego_xy.shape[0]
        dt_t = torch.as_tensor(dt, dtype=ego_xy.dtype, device=ego_xy.device).expand(S)

        # ---- predict: the offset is a random walk, so only the variance moves -------------
        if c.use_filter:
            P_pred = self.P + c.q * dt_t                                    # (S,)
            x_pred = self.x
        else:                                                               # memoryless ablation
            P_pred = torch.full_like(self.P, c.p0)
            x_pred = torch.zeros_like(self.x)

        # ---- match using the *predicted* correction so the gate can stay tight -------------
        gate = c.gate + c.gate_sigma_k * P_pred.clamp_min(0).sqrt()         # (S,)
        road_pred = road_xy - x_pred[:, None, :]                            # (S, M, 2)
        pairs = mutual_nn_pairs(ego_xy, road_pred, ego_mask, road_mask, gate)  # (S, N, M)
        n = pairs.sum(dim=(1, 2))                                           # (S,)

        # ---- measurement: median of per-pair offsets (road - ego) -------------------------
        resid = road_xy[:, None, :, :] - ego_xy[:, :, None, :]              # (S, N, M, 2)
        resid = torch.where(pairs[..., None], resid, torch.full_like(resid, float("nan")))
        if resid.shape[1] * resid.shape[2] == 0:                            # no boxes on one side
            z = x_pred.clone()
        else:
            z = torch.nanmedian(resid.reshape(S, -1, 2), dim=1).values      # (S, 2); nan if n == 0
        has = n > 0
        z = torch.where(has[:, None], z, x_pred)
        R = (math.pi / 2.0) * c.r0 / n.clamp_min(1).to(ego_xy.dtype)        # (S,)

        # ---- Kalman update (skipped, i.e. carried forward, when there is no pair) ---------
        K = torch.where(has, P_pred / (P_pred + R), torch.zeros_like(P_pred))  # (S,)
        x_new = x_pred + K[:, None] * (z - x_pred)
        P_new = (1.0 - K) * P_pred

        self.x.copy_(x_new)
        self.P.copy_(P_new)
        self.n_matched.copy_(n)

        return {
            "offset": x_new.clone(),                                        # (S, 2)
            "variance": P_new.clone(),                                      # (S,)
            "n_pairs": n,                                                   # (S,)
            "gain": K,                                                      # (S,)
            "corrected_road_xy": road_xy - x_new[:, None, :],               # (S, M, 2)
        }

    def forward(self, *args, **kwargs):  # nn.Module convention
        return self.step(*args, **kwargs)
