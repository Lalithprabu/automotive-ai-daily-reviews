"""PDM-style trajectory scoring on the synthetic 3-lane road (road axis == x, so Frenet (s,d) == (x,y)).

All functions are pure tensor code so they serve as (a) training targets for the evidence head,
(b) the evaluation metric, and (c) the ground-truth feasibility oracle used by the data generator.
Shapes: traj (...,T,2) metres in the ego frame at t=0; agents_fut (...,N,T,2); valid (...,N).
"""
import torch

DT = 0.5                    # seconds per future step
TF = 12                     # future steps (6 s)
EGO_HALF_LEN = 2.25         # ego & agents are 4.5 m x 2.0 m boxes
COLL_DX = 2 * EGO_HALF_LEN + 0.3
COLL_DY = 2.0 + 0.1
ROAD_HALF = 4.0             # |y| above this => leaves the drivable corridor (road edge 5.25 m minus half vehicle width)
V_REF = 13.0                # progress reference speed (m/s)
ACC_MAX = 4.0               # comfort limit (m/s^2)
MANEUVERS = ["KEEP", "LEFT", "RIGHT", "BRAKE", "FOLLOW"]


def factors(traj, agents_fut, valid):
    """Returns dict of (...,) tensors: nc, dac, ep, comf (all in [0,1])."""
    dx = (traj[..., None, :, 0] - agents_fut[..., 0]).abs()            # (...,N,T)
    dy = (traj[..., None, :, 1] - agents_fut[..., 1]).abs()
    hit = (dx < COLL_DX) & (dy < COLL_DY) & valid[..., None].bool()
    nc = (~hit.flatten(-2).any(-1)).float()
    dac = (traj[..., 1].abs() <= ROAD_HALF).all(-1).float()
    ep = (traj[..., -1, 0] / (V_REF * TF * DT)).clamp(0, 1)
    pts = torch.cat([torch.zeros_like(traj[..., :1, :]), traj], -2)     # prepend origin
    vel = (pts[..., 1:, :] - pts[..., :-1, :]) / DT
    acc = (vel[..., 1:, :] - vel[..., :-1, :]) / DT
    comf = (acc.abs().amax((-1, -2)) <= ACC_MAX).float()
    return dict(nc=nc, dac=dac, ep=ep, comf=comf)


def pdms(f):
    return f["nc"] * f["dac"] * (0.8 * f["ep"] + 0.2 * f["comf"])


def classify_maneuver(traj):
    """Heuristic maneuver label of a trajectory (for coverage metrics). (...,T,2) -> (...,) long."""
    yT = traj[..., -1, 1]
    xT = traj[..., -1, 0]
    prog = xT / (V_REF * TF * DT)
    lab = torch.zeros_like(yT, dtype=torch.long)                        # KEEP
    lab = torch.where(prog < 0.55, torch.full_like(lab, 4), lab)        # FOLLOW (slower than ~0.55 of ref)
    # BRAKE = effectively stopping: last-second speed < 1.5 m/s
    v_end = (traj[..., -1, 0] - traj[..., -3, 0]) / (2 * DT)
    lab = torch.where(v_end < 1.5, torch.full_like(lab, 3), lab)
    lab = torch.where(yT > 1.75, torch.full_like(lab, 1), lab)
    lab = torch.where(yT < -1.75, torch.full_like(lab, 2), lab)
    return lab
