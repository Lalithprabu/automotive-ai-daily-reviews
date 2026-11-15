"""Downstream use: an ego MERGE planner that treats the predicted occupancy set as hard constraints.

The ego is in a lane next to the target and must merge into the target's lane within 3 s (ahead of it or
behind it).  Candidates = {merge start 0 / 0.5 / 1.0 s} x {accel -3, -1.5, 0, +1, +2} plus "stay in lane"
fallbacks {+1, 0, -2, -4, -6}.  A candidate is FEASIBLE if no point of the (margin-inflated) ego footprint
falls inside the predicted union at any step:   for all t, i, p:  max_h n_h.(p - c_i(t)) - s_ih(t) > margin.
That is exactly the hyperplane collision-avoidance constraint the paper feeds into trajectory optimisation;
here it filters a candidate set instead of an optimiser (my simplification).

Pick: the feasible merge with the most progress; else the feasible stay with the most progress; else -6.
Truth: collision if the ego footprint and the TRUE target footprint overlap at any step; "interaction
completed" = merged AND no collision (the paper's 'interaction completion rate', in toy form).
"""
from __future__ import annotations

import torch

from . import geometry as G
from . import world as W
from .model import traj_heading

LC_START = (0.0, 0.5, 1.0)
LC_DUR = 2.0
MERGE_ACC = (-3.0, -1.5, 0.0, 1.0, 2.0)
STAY_ACC = (1.0, 0.0, -2.0, -4.0, -6.0)


def candidates(ego: torch.Tensor, lane_y: torch.Tensor):
    """ego (B, 3) = [side, x0, v0]; lane_y (B,) target-lane centre.  -> traj (B, C, H, 2), is_merge (C,)."""
    t = torch.tensor(W.T_FUT, dtype=torch.float32)
    side, x0, v0 = ego[:, 0], ego[:, 1], ego[:, 2]
    y_ego = lane_y + side * W.LANE_W
    trajs, merge = [], []
    for ts in LC_START:
        for a in MERGE_ACC:
            tt = torch.minimum(t[None], (v0 / max(-a, 1e-6))[:, None]) if a < 0 else t[None].expand(len(v0), -1)
            x = x0[:, None] + v0[:, None] * tt + 0.5 * a * tt ** 2
            u = ((t - ts) / LC_DUR).clamp(0, 1)
            y = y_ego[:, None] + (lane_y - y_ego)[:, None] * (u * u * (3 - 2 * u))
            trajs.append(torch.stack([x, y], -1)); merge.append(True)
    for a in STAY_ACC:
        tt = torch.minimum(t[None], (v0 / max(-a, 1e-6))[:, None]) if a < 0 else t[None].expand(len(v0), -1)
        x = x0[:, None] + v0[:, None] * tt + 0.5 * a * tt ** 2
        trajs.append(torch.stack([x, y_ego[:, None].expand_as(x)], -1)); merge.append(False)
    return torch.stack(trajs, 1), torch.tensor(merge)


def ego_heading(traj):
    """Heading along the ego path (finite differences, first step uses the second)."""
    d = traj[..., 1:, :] - traj[..., :-1, :]
    h = torch.atan2(d[..., 1], d[..., 0].clamp(min=1e-2))
    return torch.cat([h[..., :1], h], -1)


def feasible(traj, c, s, margin=0.25, chunk=128):
    """traj (B, C, H, 2) vs predicted union c (B, M, H, 2), s (B, M, H, N_H) -> (B, C) bool."""
    out = []
    for i in range(0, traj.shape[0], chunk):
        tr, cc, ss = traj[i:i + chunk], c[i:i + chunk], s[i:i + chunk]
        p = G.footprint_points(tr, ego_heading(tr), 5, 3, W.EGO_L, W.EGO_W)          # (b, C, H, 15, 2)
        v = G.violation(p[:, :, None], cc[:, None], ss[:, None])                     # (b, C, M, H, 15)
        out.append(v.amin(-1).amin(-1).amin(-1) > margin)
    return torch.cat(out)


def choose(traj, is_merge, feas):
    """Index of the chosen candidate per scene.  -> (B,) long."""
    prog = traj[:, :, -1, 0]                                                          # (B, C) final x
    big = 1e6
    score = torch.where(feas & is_merge[None], prog + big, torch.where(feas, prog, prog - big))
    k = score.argmax(-1)
    none = ~feas.any(-1)                                                              # nothing feasible:
    return torch.where(none, torch.full_like(k, traj.shape[1] - 1), k)                # brake hardest in lane


def collides_true(traj, fut, head):
    """Ego path (B, H, 2) vs true target footprint (B, H, 2)/(B, H) -> (B,) bool (two-way point test)."""
    pe = G.footprint_points(traj, ego_heading(traj), 9, 5, W.EGO_L, W.EGO_W)
    pt = G.footprint_points(fut, head, 9, 5)
    in_t = G.violation(pe, fut, G.rect_support(head)) <= 0                           # ego points in target
    in_e = G.violation(pt, traj, G.rect_support(ego_heading(traj), W.EGO_L, W.EGO_W)) <= 0
    return (in_t.any(-1) | in_e.any(-1)).any(-1)


def evaluate_planner(c, s, d):
    traj, is_merge = candidates(d["ego"], d["lane_y"])
    feas = feasible(traj, c, s)
    k = choose(traj, is_merge, feas)
    sel = traj[torch.arange(len(k)), k]
    coll = collides_true(sel, d["fut"], d["head"])
    merged = is_merge[k]
    return dict(completion=(merged & ~coll).float(), collision=coll.float(), merged=merged.float(),
                choice=k, traj=sel, n_feasible_merge=(feas & is_merge[None]).sum(-1).float())
