"""Closed-loop cut-in simulation (1-D longitudinal + bell-shaped lateral intrusion).

Three ego strategies on the same cut-in:
  reactive   - brakes only once the cutter's centre is inside the ego lane (+ actuator delay)
  predictive - brakes as soon as the trained KG-Bayes posterior P(cutter enters ego lane) > trigger_prob
  (a lateral-evasion strategy exists in the paper; NOT reproduced here -- see SOURCING.md)
Braking law (both): a = -clip(v_rel^2 / (2*(gap - d0)), 0, a_max) once a 'lead' is declared.
Predictive is additionally capped to comfort decel until the cutter is physically in the lane.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np
import torch
from .data import features_for, LK, cutter_features, cutter_lateral_velocity, discretise, LANE_W, LLC, RLC, H0

D0 = 4.0          # standstill buffer (m)
VEH_W = 1.9


@dataclass
class Trace:
    t: list = field(default_factory=list); gap: list = field(default_factory=list)
    v_ego: list = field(default_factory=list); a_ego: list = field(default_factory=list)
    y_cut: list = field(default_factory=list); s_ego: list = field(default_factory=list)
    s_cut: list = field(default_factory=list); p_enter: list = field(default_factory=list)
    post: list = field(default_factory=list); collided: bool = False; declared: bool = False; t_collision: float | None = None

    def stats(self):
        a = np.array(self.a_ego)
        return dict(collided=self.collided, min_gap=float(min(self.gap)), peak_decel=float(-a.min()),
                    peak_decel_before_impact=float(-a[:self._impact_idx()].min()) if len(a) else 0.0)

    def _impact_idx(self):
        if self.collided:
            return max(1, int(np.searchsorted(self.t, self.t_collision)))
        return len(self.a_ego)


def run_cutin(model, strategy: str, cfg: dict, seed: int, log_prior=None, direction=1.0,
              t_start_cross=5.25, t_end=11.0, lane_keep=False) -> Trace:
    """Simulate one cut-in. The cutter's lateral centre starts at +/-LANE_W (adjacent lane) and ends in ego lane (y=0)."""
    rng = np.random.default_rng(seed)
    dt = cfg["dt"]; v0 = cfg["ego_speed"]; vc = cfg["cutter_speed"]
    t_c = t_start_cross
    gap0 = cfg["crossing_gap_m"] + (v0 - vc) * t_c
    s_ego, s_cut, v_ego = 0.0, gap0 + 4.5, v0       # s_cut = rear bumper-to-front gap handled below
    y = LANE_W * direction
    delay_steps = int(round(cfg["reaction_delay_s"] / dt))
    cmd_buf = [0.0] * delay_steps
    tr = Trace(); declared = False; a_cmd = 0.0; streak = 0
    for k in range(int(t_end / dt)):
        t = k * dt
        # --- cutter lateral motion: integrate bell velocity toward the ego lane
        if not lane_keep:
            y -= direction * cutter_lateral_velocity(t, t_c) * dt
            y = max(min(y, LANE_W), 0.0) if direction > 0 else min(max(y, -LANE_W), 0.0)
        # --- perception: noisy features at current time-to-crossing -> posterior
        if lane_keep:      # false-alarm scenario: neighbour just keeps its lane
            feats = features_for(np.array([LK]), np.array([3.0]), rng)[0]
        else:
            feats = cutter_features(t_c - t, direction, rng)
        ents = torch.from_numpy(discretise(feats[None, :]))
        with torch.no_grad():
            post = model.log_posterior(ents, log_prior).exp()[0].numpy()
        p_enter = float(post[LLC] if direction > 0 else post[RLC])
        in_lane = abs(y) < (LANE_W / 2)                       # centre crossed lane line
        gap = (s_cut - s_ego) - 4.5                           # bumper gap (veh len 4.5)
        v_rel = max(v_ego - vc, 0.0)
        if strategy == "reactive":
            declared = declared or in_lane
        else:
            streak = streak + 1 if p_enter > cfg["trigger_prob"] else 0
            declared = declared or streak >= cfg.get("trigger_frames", 1)   # debounce: N consecutive frames
        if declared and gap > -50:
            b_req = v_rel ** 2 / (2 * max(gap - D0, 0.3))
            cap = cfg["max_decel"] if (strategy == "reactive" or in_lane) else cfg["comfort_decel"]
            a_cmd = -float(np.clip(b_req, 0, cap))
        else:
            a_cmd = 0.0
        tr.declared = tr.declared or declared
        cmd_buf.append(a_cmd); a = cmd_buf.pop(0)             # actuator / reaction delay
        v_ego = max(v_ego + a * dt, 0.0); s_ego += v_ego * dt; s_cut += vc * dt
        gap = (s_cut - s_ego) - 4.5
        lat_overlap = abs(y) < VEH_W
        tr.t.append(t); tr.gap.append(gap); tr.v_ego.append(v_ego); tr.a_ego.append(a)
        tr.y_cut.append(y); tr.s_ego.append(s_ego); tr.s_cut.append(s_cut)
        tr.p_enter.append(p_enter); tr.post.append(post)
        if lat_overlap and gap <= 0 and not tr.collided:
            tr.collided = True; tr.t_collision = t
            break
    return tr


def lc_log_prior(cfg):
    """Prior over (LLC, LK, RLC). Real highway traffic: lane changes are rare, so classifier-time balanced
    priors would over-trigger. cfg['prior_lane_change'] is P(LLC)=P(RLC)."""
    p = cfg.get("prior_lane_change", 1 / 3)
    return torch.log(torch.tensor([p, 1 - 2 * p, p]))


def false_alarm_rate(model, cfg, log_prior=None, n=None, base_seed=5000):
    """Neighbour keeps its lane: how often does the predictive ego brake anyway, and how hard?"""
    n = n or cfg["n_monte_carlo"]; S = []
    for i in range(n):
        tr = run_cutin(model, "predictive", cfg, base_seed + i, log_prior, direction=1.0 if i % 2 == 0 else -1.0, lane_keep=True)
        S.append((tr.declared, -min(tr.a_ego)))
    return dict(false_brake_rate=float(np.mean([d for d, _ in S])), mean_peak_decel=float(np.mean([a for _, a in S])), n=n)


def monte_carlo(model, cfg, log_prior=None, n=None, base_seed=1000):
    n = n or cfg["n_monte_carlo"]; res = {}
    for strat in ("reactive", "predictive"):
        S = [run_cutin(model, strat, cfg, base_seed + i, log_prior,
                       direction=1.0 if i % 2 == 0 else -1.0).stats() for i in range(n)]
        res[strat] = dict(collision_rate=float(np.mean([s["collided"] for s in S])),
                          mean_peak_decel=float(np.mean([s["peak_decel_before_impact"] for s in S])),
                          mean_min_gap=float(np.mean([s["min_gap"] for s in S])), n=n)
    return res
