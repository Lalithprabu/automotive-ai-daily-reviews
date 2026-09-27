"""
Synthetic ego-centric driving scene generator for the ForeDrive reconstruction.

DISCLOSURE (see SOURCING.md): the ForeDrive paper trains on real NAVSIM /
nuScenes-derived data with real camera images. We have none of that here.
This module generates fully synthetic scenes so the repo is runnable and
self-contained, and — critically — so the JEPA forecasting task and the
imitation-learning task both have *real, learnable signal* rather than being
degenerate (a standing lesson from earlier entries in this series: a
no-signal synthetic task makes every downstream metric meaningless without a
sanity check, see tests/test_sanity_signal.py for the check that would catch
that failure mode).

What each scene contains
-------------------------
- An ego vehicle driving forward along a straight 3-lane road at roughly
  constant longitudinal position advancement, governed by a small REACTIVE
  controller (not a random walk): it brakes when an agent ahead in its lane
  is too close, and takes a small evasive lateral shift when an agent gets
  very close. This gives the imitation-learning target (the "expert"
  trajectory) genuine, non-trivial structure — a straight-line / constant
  velocity baseline should NOT track it well whenever braking/evasion
  triggers, which is exactly what tests/test_sanity_signal.py checks for.
- 1-3 other agents moving with simple kinematics: constant velocity, plus
  occasionally a scripted lane change or a scripted stop event partway
  through the scene. This gives the JEPA future-prediction target genuine
  dynamics: the world several frames ahead is systematically different from
  the current frame (not simply the current frame plus i.i.d. noise), so a
  "predict no change" baseline is beatable by a model that actually learns
  the dynamics.

Rendering (IMPORTANT — simplified proxy, NOT real camera imagery)
-------------------------------------------------------------------
Real ForeDrive consumes a real front-view camera image. We do not have
camera imagery, so we render a small multi-channel *ego-centric bird's-eye
occupancy raster* and use it as a stand-in for "front view": the window
follows the ego vehicle longitudinally (ego is always at x_rel = 0, looking
forward) and keeps lateral position in absolute lane coordinates so lane
position/drift is visible. This is a deliberate simplification chosen so
the whole pipeline (encoder -> world model -> planner) is runnable on CPU
in a couple of minutes; it is explicitly NOT a perspective camera image and
should not be read as one.

Channels of the raster (C=3):
  0: lane geometry (dashed lane-boundary lines, scrolling with ego motion)
  1: ego footprint (small box near the bottom of the window)
  2: other-agent occupancy (boxes for every other agent inside the window)
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# World / rendering geometry constants
# ---------------------------------------------------------------------------
X_RANGE_M = 40.0      # forward window extent in meters (ego always at x_rel=0)
Y_RANGE_M = 12.0       # lateral window extent in meters, centered at y=0 (i.e. y in [-6, 6])
LANE_CENTERS = (-4.0, 0.0, 4.0)   # 3 lane centers, 4m lane width
LANE_BOUNDARIES = (-6.0, -2.0, 2.0, 6.0)
VEHICLE_LENGTH_M = 4.0
VEHICLE_WIDTH_M = 2.0
# NOTE (real bug found + fixed during development): the dash period must NOT
# evenly divide the ego's typical per-timestep travel distance
# (EGO_CRUISE_V * dt = 8.0 * 0.5 = 4.0m). An earlier version used
# DASH_PERIOD_M = 4.0, which made the scrolling lane-dash channel exactly
# alias to a static pattern for any cruise-speed scene (4.0m shift mod 4.0m
# period = 0every step), silently erasing the one visual cue that a frame
# had advanced in time whenever nothing else in the scene was moving nearby.
# 5.0 is coprime-ish with the 4.0m cruise step over the horizons we predict
# (max 4 steps => cumulative shifts 4,8,12,16m => phases 4,3,2,1 mod 5, all
# distinct), so channel 0 reliably carries a time signal even in the
# common "cruising with no nearby agent" case.
DASH_PERIOD_M = 5.0
DASH_ON_M = 2.5

# Ego reactive-controller constants
EGO_CRUISE_V = 8.0        # m/s target cruising speed
EGO_V_MAX = 12.0
EGO_MAX_ACCEL = 2.5        # m/s^2
EGO_MAX_DECEL = 4.5         # m/s^2
SAFE_GAP_M = 14.0            # start braking when gap to lead agent < this
CRITICAL_GAP_M = 6.0           # trigger evasive lateral shift when gap < this
LATERAL_RESPONSE_GAIN = 0.35   # per-step first-order response toward lateral target
SAME_LANE_TOL_M = 2.0            # |dy| below this counts as "same lane" for gap-finding


@dataclasses.dataclass
class AgentScript:
    """Scripted kinematic behaviour for one non-ego agent over a scene."""
    x0: float
    y_lane: float          # starting lane center
    vx: float
    lane_change_t: Optional[int]     # timestep index at which a lane change begins, or None
    target_lane: float               # lane center to move to if lane_change_t is set
    stop_t: Optional[int]            # timestep index at which the agent begins stopping, or None


@dataclasses.dataclass
class Scene:
    """A fully simulated synthetic scene: per-timestep world state for ego + agents."""
    scene_len: int
    dt: float
    ego_x: np.ndarray   # (T,) world-frame forward position (m)
    ego_y: np.ndarray   # (T,) world-frame lateral position (m)
    ego_vx: np.ndarray  # (T,) forward speed (m/s)
    ego_vy: np.ndarray  # (T,) lateral speed (m/s), finite-differenced
    agents_x: np.ndarray  # (num_agents, T)
    agents_y: np.ndarray  # (num_agents, T)
    num_agents: int


def _simulate_agent(script: AgentScript, scene_len: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Roll out one scripted agent's (x, y) trajectory over the scene."""
    xs = np.zeros(scene_len, dtype=np.float32)
    ys = np.zeros(scene_len, dtype=np.float32)
    x, y = script.x0, script.y_lane
    vx = script.vx
    for t in range(scene_len):
        # Scripted stop event: agent decelerates to a halt shortly after stop_t.
        if script.stop_t is not None and t >= script.stop_t:
            vx = max(0.0, vx - 3.0 * dt)
        # Scripted lane change: smoothly interpolate lateral position toward target_lane.
        if script.lane_change_t is not None and t >= script.lane_change_t:
            y += (script.target_lane - y) * 0.30
        xs[t] = x
        ys[t] = y
        x += vx * dt
    return xs, ys


def generate_scene(rng: np.random.Generator, scene_len: int, dt: float,
                    num_agents_min: int, num_agents_max: int) -> Scene:
    """Simulate one synthetic scene: agents follow scripted kinematics, ego follows
    a reactive gap-keeping + evasive-lateral-shift controller reacting to them."""
    num_agents = int(rng.integers(num_agents_min, num_agents_max + 1))

    # NOTE: these ranges are deliberately tuned (see synthetic_dataset.py dev
    # notes / SOURCING.md) so that a lead agent triggers the ego's reactive
    # controller (braking / evasive shift) in a LARGE fraction of scenes, not
    # just a rare minority. An earlier version placed agents much farther
    # ahead and moving faster on average, which meant ~83% of scenes never
    # triggered any reaction at all -- the "expert" trajectory was then
    # almost always indistinguishable from a trivial constant-velocity
    # extrapolation, making the imitation-learning task nearly signal-free in
    # aggregate (the sanity check in tests/test_sanity_signal.py is what
    # caught this). Agents now start closer to the ego and are generally
    # slower than its cruise speed, so braking/evasion is the common case.
    agent_scripts: List[AgentScript] = []
    for _ in range(num_agents):
        x0 = float(rng.uniform(8.0, 22.0))            # start much closer ahead of ego
        lane = float(rng.choice(LANE_CENTERS))
        vx = float(rng.uniform(2.0, 7.0))               # reliably slower than ego cruise -> forces braking
        has_lane_change = rng.random() < 0.35
        lane_change_t = int(rng.integers(3, scene_len - 3)) if has_lane_change else None
        target_lane = float(rng.choice(LANE_CENTERS)) if has_lane_change else lane
        has_stop = rng.random() < 0.35
        stop_t = int(rng.integers(4, scene_len - 4)) if has_stop else None
        agent_scripts.append(AgentScript(x0, lane, vx, lane_change_t, target_lane, stop_t))

    agents_x = np.zeros((max(num_agents, 1), scene_len), dtype=np.float32)
    agents_y = np.zeros((max(num_agents, 1), scene_len), dtype=np.float32)
    for i, script in enumerate(agent_scripts):
        xs, ys = _simulate_agent(script, scene_len, dt)
        agents_x[i] = xs
        agents_y[i] = ys

    # --- Ego reactive controller -------------------------------------------------
    ego_x = np.zeros(scene_len, dtype=np.float32)
    ego_y = np.zeros(scene_len, dtype=np.float32)
    ego_vx_arr = np.zeros(scene_len, dtype=np.float32)
    x, y, vx = 0.0, 0.0, EGO_CRUISE_V * 0.6   # start a bit below cruise speed
    lateral_target = 0.0
    for t in range(scene_len):
        ego_x[t] = x
        ego_y[t] = y
        ego_vx_arr[t] = vx

        # Find the closest agent ahead, roughly in the ego's current lane.
        gap = np.inf
        lead_vx = EGO_CRUISE_V
        if num_agents > 0:
            ax = agents_x[:, t]
            ay = agents_y[:, t]
            ahead_mask = ax > x
            same_lane_mask = np.abs(ay - y) < SAME_LANE_TOL_M
            candidate_mask = ahead_mask & same_lane_mask
            if np.any(candidate_mask):
                candidate_gaps = ax[candidate_mask] - x
                idx = int(np.argmin(candidate_gaps))
                gap = float(candidate_gaps[idx])
                lead_idx = np.where(candidate_mask)[0][idx]
                # finite-difference the lead agent's own speed for a smoother reaction
                if t > 0:
                    lead_vx = float((agents_x[lead_idx, t] - agents_x[lead_idx, t - 1]) / dt)
                else:
                    lead_vx = float(agents_x[lead_idx, min(t + 1, scene_len - 1)] - agents_x[lead_idx, t]) / dt

        # Longitudinal control: gap-keeping proportional controller.
        if gap < SAFE_GAP_M:
            target_v = max(0.0, lead_vx * max(0.0, (gap - 1.5) / SAFE_GAP_M))
            accel = (target_v - vx) / dt
            accel = float(np.clip(accel, -EGO_MAX_DECEL, EGO_MAX_ACCEL))
        else:
            accel = float(np.clip((EGO_CRUISE_V - vx) / dt, -EGO_MAX_DECEL, EGO_MAX_ACCEL))
        vx = float(np.clip(vx + accel * dt, 0.0, EGO_V_MAX))

        # Lateral control: evasive shift toward the nearest open lane if an agent is
        # critically close ahead in-lane; otherwise relax back toward lane center 0.
        if gap < CRITICAL_GAP_M:
            # shift toward whichever adjacent lane keeps us farther from lane 0 traffic
            lateral_target = -4.0 if y >= 0 else 4.0
        else:
            lateral_target = 0.0
        y += (lateral_target - y) * LATERAL_RESPONSE_GAIN

        x += vx * dt

    ego_vy = np.gradient(ego_y, dt).astype(np.float32)

    return Scene(
        scene_len=scene_len, dt=dt,
        ego_x=ego_x, ego_y=ego_y, ego_vx=ego_vx_arr, ego_vy=ego_vy,
        agents_x=agents_x, agents_y=agents_y, num_agents=num_agents,
    )


def _draw_box(raster_channel: np.ndarray, cx: float, cy: float, length: float, width: float,
              raster_size: int) -> None:
    """Rasterize a filled axis-aligned box (world meters) into one occupancy channel.
    World -> pixel mapping: x_rel in [0, X_RANGE_M] -> col in [0, raster_size),
                             y     in [-Y_RANGE_M/2, Y_RANGE_M/2] -> row in [0, raster_size)
    (row 0 = far left / -Y, increasing row = +Y; this is an arbitrary but fixed convention)."""
    px_per_m_x = raster_size / X_RANGE_M
    px_per_m_y = raster_size / Y_RANGE_M

    x_lo, x_hi = cx - length / 2.0, cx + length / 2.0
    y_lo, y_hi = cy - width / 2.0, cy + width / 2.0

    col_lo = int(np.floor(x_lo * px_per_m_x))
    col_hi = int(np.ceil(x_hi * px_per_m_x))
    row_lo = int(np.floor((y_lo + Y_RANGE_M / 2.0) * px_per_m_y))
    row_hi = int(np.ceil((y_hi + Y_RANGE_M / 2.0) * px_per_m_y))

    col_lo, col_hi = np.clip([col_lo, col_hi], 0, raster_size)
    row_lo, row_hi = np.clip([row_lo, row_hi], 0, raster_size)
    if col_hi > col_lo and row_hi > row_lo:
        raster_channel[row_lo:row_hi, col_lo:col_hi] = 1.0


def render_frame(scene: Scene, t: int, raster_size: int) -> np.ndarray:
    """Render the ego-centric occupancy raster at timestep t. Returns (3, H, W) float32
    in [0, 1]. See module docstring for the channel layout and the "not real camera
    imagery" disclosure."""
    raster = np.zeros((3, raster_size, raster_size), dtype=np.float32)
    ego_x_t = scene.ego_x[t]
    ego_y_t = scene.ego_y[t]

    # --- Channel 0: dashed lane boundaries, scrolling with ego's world x position ---
    px_per_m_x = raster_size / X_RANGE_M
    px_per_m_y = raster_size / Y_RANGE_M
    for boundary_y in LANE_BOUNDARIES:
        # NOTE: lane boundaries are drawn in the WINDOW's absolute-y convention (not
        # ego-relative), matching the ego/agent channels below, so lane structure and
        # ego lateral drift are both directly comparable in the same raster.
        row = int(np.clip((boundary_y + Y_RANGE_M / 2.0) * px_per_m_y, 0, raster_size - 1))
        row_band = raster[0, max(row - 1, 0):row + 1, :]
        # Dash pattern is a function of world-frame x (ego_x_t) so it visibly scrolls
        # from frame to frame as the ego advances -- this is what gives channel 0
        # genuine temporal structure instead of being a static background.
        for col in range(raster_size):
            x_rel = col / px_per_m_x
            world_x = ego_x_t + x_rel
            phase = world_x % DASH_PERIOD_M
            if phase < DASH_ON_M:
                row_band[:, col] = 1.0

    # --- Channel 1: ego footprint, near the bottom (x_rel ~ 0) of the window ---
    _draw_box(raster[1], cx=VEHICLE_LENGTH_M / 2.0 + 0.5, cy=ego_y_t,
              length=VEHICLE_LENGTH_M, width=VEHICLE_WIDTH_M, raster_size=raster_size)

    # --- Channel 2: other agents, in window-relative x (agent_x - ego_x), absolute y ---
    for i in range(scene.num_agents):
        ax = scene.agents_x[i, t] - ego_x_t
        ay = scene.agents_y[i, t]
        if 0.0 <= ax <= X_RANGE_M:
            _draw_box(raster[2], cx=ax, cy=ay, length=VEHICLE_LENGTH_M, width=VEHICLE_WIDTH_M,
                      raster_size=raster_size)

    return raster


class SyntheticDrivingDataset(Dataset):
    """Each item is one synthetic scene, sampled at a fixed "current" timestep t:

      current_frame     : (C, H, W)                current ego-centric raster
      future_frames      : (num_horizons, C, H, W)   rasters at t + h for h in `horizons`
      ego_context         : (4,)                        [vx, vy, accel_proxy, speed] at t
      expert_waypoints     : (num_waypoints, 2)           GT relative (dx, dy) ego displacement
                                                            from position at t, for the next
                                                            num_waypoints timesteps
      cv_baseline_waypoints: (num_waypoints, 2)            constant-velocity extrapolation
                                                            baseline, for sanity-test comparisons
    """

    def __init__(self, num_scenes: int, scene_len: int, dt: float, raster_size: int,
                 horizons: List[int], num_waypoints: int, context_t: int,
                 num_agents_min: int, num_agents_max: int, seed: int):
        assert context_t + max(horizons) < scene_len, "horizons run past scene_len"
        assert context_t + num_waypoints < scene_len, "waypoints run past scene_len"
        self.num_scenes = num_scenes
        self.scene_len = scene_len
        self.dt = dt
        self.raster_size = raster_size
        self.horizons = horizons
        self.num_waypoints = num_waypoints
        self.context_t = context_t
        self.num_agents_min = num_agents_min
        self.num_agents_max = num_agents_max
        # Each scene gets its own decorrelated seed derived from a base RNG so the
        # dataset is fully deterministic and reproducible given `seed`.
        base_rng = np.random.default_rng(seed)
        self._scene_seeds = base_rng.integers(0, 2**31 - 1, size=num_scenes)

    def __len__(self) -> int:
        return self.num_scenes

    def get_scene(self, idx: int) -> Scene:
        rng = np.random.default_rng(int(self._scene_seeds[idx]))
        return generate_scene(rng, self.scene_len, self.dt, self.num_agents_min, self.num_agents_max)

    def __getitem__(self, idx: int):
        scene = self.get_scene(idx)
        t = self.context_t

        current_frame = render_frame(scene, t, self.raster_size)
        future_frames = np.stack(
            [render_frame(scene, t + h, self.raster_size) for h in self.horizons], axis=0
        )

        vx = scene.ego_vx[t]
        vy = scene.ego_vy[t]
        accel_proxy = (scene.ego_vx[min(t + 1, self.scene_len - 1)] - scene.ego_vx[t]) / scene.dt
        speed = float(np.hypot(vx, vy))
        ego_context = np.array([vx, vy, accel_proxy, speed], dtype=np.float32)

        ex0, ey0 = scene.ego_x[t], scene.ego_y[t]
        expert_waypoints = np.stack([
            [scene.ego_x[t + k] - ex0, scene.ego_y[t + k] - ey0]
            for k in range(1, self.num_waypoints + 1)
        ], axis=0).astype(np.float32)

        # Constant-velocity baseline: extrapolate using ego's velocity AT t only.
        cv_baseline = np.stack([
            [vx * scene.dt * k, vy * scene.dt * k] for k in range(1, self.num_waypoints + 1)
        ], axis=0).astype(np.float32)

        return {
            "current_frame": torch.from_numpy(current_frame),
            "future_frames": torch.from_numpy(future_frames),
            "ego_context": torch.from_numpy(ego_context),
            "expert_waypoints": torch.from_numpy(expert_waypoints),
            "cv_baseline_waypoints": torch.from_numpy(cv_baseline),
        }
