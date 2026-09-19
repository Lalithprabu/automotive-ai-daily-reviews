"""Synthetic BEV scene generator + classical fixed-shape potential-field
baseline.

No NAVSIM access was available for this reconstruction, so we build a
minimal, from-scratch synthetic stand-in scene: an ego vehicle following
a straight in-lane reference path, a lead vehicle driving away in the
same lane, and a pedestrian/cyclist crossing the lane at some point in
time. This is enough to exercise every shape/behavior the model needs to
learn: a moving "safe to follow" agent and a moving "conflict" agent
whose risk should be sharply localized in space AND time.

Also implemented here: `classical_potential_field`, the decades-old
fixed-shape "safety bubble" baseline READ replaces -- an isotropic
Gaussian bump of constant radius centered on each agent's position,
independent of that agent's velocity, heading, or the ego's intent. This
is used purely as the "OLD WAY" comparison panel in simulate.py.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

# Scene extent, in meters, centered at the origin. The ego starts at
# roughly (0, 0) and drives in +x.
SCENE_HALF_EXTENT = 30.0
GRID_SIZE = 48  # BEV raster resolution (GRID_SIZE x GRID_SIZE)
NUM_HISTORY_STEPS = 5
CLASSICAL_RISK_RADIUS = 4.0  # meters, fixed regardless of agent motion


@dataclass
class SyntheticScene:
    """A single synthetic driving scene.

    Agents are indexed [0: lead vehicle, 1: crossing pedestrian/cyclist].
    All positions in meters, ego-centric, +x = forward, +y = left.
    """

    num_timesteps: int
    dt: float
    lead_positions: np.ndarray  # [T, 2]
    crossing_positions: np.ndarray  # [T, 2]
    ego_reference_path: np.ndarray  # [T, 2]
    lead_history: np.ndarray = field(default=None)  # [T, NUM_HISTORY_STEPS, 4]
    crossing_history: np.ndarray = field(default=None)  # [T, NUM_HISTORY_STEPS, 4]


def _heading_speed(positions: np.ndarray, dt: float) -> np.ndarray:
    """Compute per-step (heading, speed) from a position sequence.

    Args:
        positions: [T, 2]
    Returns:
        [T, 2] (heading in radians, speed in m/s)
    """
    vel = np.gradient(positions, dt, axis=0)  # [T, 2]
    heading = np.arctan2(vel[:, 1], vel[:, 0])
    speed = np.linalg.norm(vel, axis=1)
    return np.stack([heading, speed], axis=1)


def _build_history(positions: np.ndarray, dt: float, num_history: int) -> np.ndarray:
    """Build a sliding window of (x, y, heading, speed) history ending at
    each timestep (padded with the first frame at the start).

    Returns:
        [T, num_history, 4]
    """
    T = positions.shape[0]
    hs = _heading_speed(positions, dt)  # [T, 2]
    full = np.concatenate([positions, hs], axis=1)  # [T, 4]

    padded = np.concatenate([np.repeat(full[:1], num_history - 1, axis=0), full], axis=0)
    windows = np.stack([padded[i:i + num_history] for i in range(T)], axis=0)  # [T, H, 4]
    return windows


def generate_synthetic_scene(num_timesteps: int = 30, dt: float = 0.2, seed: int = None) -> SyntheticScene:
    """Generates one synthetic scene: lead vehicle driving away in-lane,
    plus a pedestrian/cyclist crossing the lane partway through.

    Args:
        num_timesteps: number of frames T.
        dt: seconds per frame.
        seed: optional RNG seed for reproducibility.
    Returns:
        a SyntheticScene with per-timestep agent positions and histories.
    """
    rng = np.random.default_rng(seed)

    t = np.arange(num_timesteps) * dt

    # Ego drives straight down the lane (+x) at ~8 m/s.
    ego_speed = 8.0
    ego_reference_path = np.stack([ego_speed * t, np.zeros_like(t)], axis=1)

    # Lead vehicle starts ~12m ahead, drives away faster than ego.
    lead_start_x = 12.0 + rng.uniform(-1.0, 1.0)
    lead_speed = 10.0 + rng.uniform(-0.5, 0.5)
    lead_positions = np.stack([lead_start_x + lead_speed * t, np.zeros_like(t)], axis=1)

    # Crossing agent walks/cycles across the lane, reaching the lane
    # centerline (y=0) at roughly the scene's midpoint in time, at an
    # ego x-position ego reaches a bit later -- i.e. a genuine conflict.
    cross_speed = 1.8 + rng.uniform(-0.3, 0.3)  # pedestrian-ish
    conflict_x = ego_speed * t[num_timesteps // 2] + rng.uniform(-1.5, 1.5)
    cross_start_y = -6.0 - rng.uniform(0.0, 1.5)
    crossing_positions = np.stack([
        np.full(num_timesteps, conflict_x),
        cross_start_y + cross_speed * t,
    ], axis=1)

    lead_history = _build_history(lead_positions, dt, NUM_HISTORY_STEPS)
    crossing_history = _build_history(crossing_positions, dt, NUM_HISTORY_STEPS)

    return SyntheticScene(
        num_timesteps=num_timesteps,
        dt=dt,
        lead_positions=lead_positions,
        crossing_positions=crossing_positions,
        ego_reference_path=ego_reference_path,
        lead_history=lead_history,
        crossing_history=crossing_history,
    )


def rasterize_bev(scene: SyntheticScene, timestep: int, grid_size: int = GRID_SIZE,
                   half_extent: float = SCENE_HALF_EXTENT) -> np.ndarray:
    """Rasterizes a simple 3-channel BEV grid at a given timestep:
      channel 0: drivable lane corridor (constant band around y=0)
      channel 1: lead vehicle occupancy (small blob at its position)
      channel 2: crossing agent occupancy (small blob at its position)

    Returns:
        [3, grid_size, grid_size] float32 array.
    """
    xs = np.linspace(-half_extent, half_extent, grid_size)
    ys = np.linspace(-half_extent, half_extent, grid_size)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")

    lane_channel = (np.abs(gy) < 2.0).astype(np.float32)

    lead_pos = scene.lead_positions[timestep]
    cross_pos = scene.crossing_positions[timestep]

    lead_channel = np.exp(-((gx - lead_pos[0]) ** 2 + (gy - lead_pos[1]) ** 2) / (2 * 1.5 ** 2)).astype(np.float32)
    cross_channel = np.exp(-((gx - cross_pos[0]) ** 2 + (gy - cross_pos[1]) ** 2) / (2 * 1.0 ** 2)).astype(np.float32)

    return np.stack([lane_channel, lead_channel, cross_channel], axis=0)


def classical_potential_field(query_xy: np.ndarray, agent_positions: np.ndarray,
                               radius: float = CLASSICAL_RISK_RADIUS) -> np.ndarray:
    """The decades-old fixed-shape "safety bubble" baseline: an isotropic
    Gaussian risk bump of constant radius around each agent, independent
    of the agent's velocity/heading or of time. This is what READ's
    learned field replaces.

    Args:
        query_xy: [..., 2] query points.
        agent_positions: [A, 2] agent positions at the query's timestep.
        radius: fixed Gaussian std, identical for every agent every time.
    Returns:
        [...] risk in [0, 1]: the max over agents of each agent's bump.
    """
    query_xy = np.asarray(query_xy)
    agent_positions = np.asarray(agent_positions)

    # Broadcast: [..., 1, 2] - [A, 2] -> [..., A, 2]
    diff = query_xy[..., None, :] - agent_positions
    dist_sq = np.sum(diff ** 2, axis=-1)  # [..., A]
    bumps = np.exp(-dist_sq / (2 * radius ** 2))  # [..., A]
    risk = np.max(bumps, axis=-1)
    return risk


def sample_agent_proximal_probes(scene: SyntheticScene, timestep: int, num_probes: int = 16,
                                  jitter_std: float = 1.5, seed: int = None) -> np.ndarray:
    """Samples (x, y, t) probe points near the agents at a given
    timestep, for the risk-ranking loss's "should be high risk" side.

    Returns:
        [num_probes, 3] (x, y, t) points, t given in the same units as
        the scene's dt-scaled timeline.
    """
    rng = np.random.default_rng(seed)
    agent_pos = np.stack([scene.lead_positions[timestep], scene.crossing_positions[timestep]], axis=0)  # [2, 2]

    picks = rng.integers(0, agent_pos.shape[0], size=num_probes)
    centers = agent_pos[picks]  # [num_probes, 2]
    jitter = rng.normal(0.0, jitter_std, size=centers.shape)
    xy = centers + jitter

    t_val = timestep * scene.dt
    t_col = np.full((num_probes, 1), t_val, dtype=np.float32)
    return np.concatenate([xy.astype(np.float32), t_col], axis=1)


def sample_background_probes(num_probes: int = 32, half_extent: float = SCENE_HALF_EXTENT,
                              t_max: float = 6.0, seed: int = None) -> np.ndarray:
    """Samples uniformly-random background (x, y, t) points, used by the
    background-risk regularizer to pull risk -> 0 away from any agent.

    Returns:
        [num_probes, 3] (x, y, t) points.
    """
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-half_extent, half_extent, size=(num_probes, 2))
    t_col = rng.uniform(0.0, t_max, size=(num_probes, 1))
    return np.concatenate([xy, t_col], axis=1).astype(np.float32)


def scene_to_tensors(scene: SyntheticScene, timestep: int, device="cpu"):
    """Convenience: build the (bev_grid, agent_history) tensors a
    SceneEncoder expects for a single scene at a given timestep, with a
    leading batch dim of 1.

    Returns:
        bev_grid: [1, 3, GRID_SIZE, GRID_SIZE]
        agent_history: [1, 2, NUM_HISTORY_STEPS, 4]
    """
    bev = rasterize_bev(scene, timestep)
    bev_grid = torch.from_numpy(bev).unsqueeze(0).float().to(device)

    hist = np.stack([scene.lead_history[timestep], scene.crossing_history[timestep]], axis=0)  # [2, H, 4]
    agent_history = torch.from_numpy(hist).unsqueeze(0).float().to(device)

    return bev_grid, agent_history
