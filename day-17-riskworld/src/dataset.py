"""Synthetic BEV driving-scenario generator.

This is entirely this repo's own synthetic data -- not derived from nuScenes
or any paper artifact. It is built specifically to contain a scenario family
the paper's mechanism is meant to catch: a cross-traffic agent that is close
to the ego's future path but, at the last-observed frame, has not yet entered
the road corridor a naive "hold everything static" (persistence) forecast
would predict. A persistence baseline therefore predicts empty road ahead,
while the ground truth (and, if trained well, RiskWorld's flow-warped
forecast) shows the agent arriving in the ego's lane a couple of steps later.

Coordinate convention used throughout this repo: grid arrays are indexed
[..., row, col] where row == y (grid_size direction 0) and col == x
(grid_size direction 1); position tuples are always written (x, y), matching
the (col, row) split and matching `grid_sample`'s (x, y) axis convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


def _soft_raster(grid_size: int, positions_xy: list[tuple[float, float]], radius: float = 1.3) -> np.ndarray:
    """Rasterizes a list of (x, y) cell-coordinate points as soft disks
    (linear falloff to 0 at `radius` cells), taking the per-cell max across
    points so overlapping agents don't double-saturate past 1.0."""
    ys, xs = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")  # (H, W) each
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for (x, y) in positions_xy:
        d = np.sqrt((xs - x) ** 2 + (ys - y) ** 2)
        val = np.clip(1.0 - d / radius, 0.0, 1.0)
        grid = np.maximum(grid, val)
    return grid


def _line_mask(grid_size: int, band: tuple[int, int], axis: str) -> np.ndarray:
    """Boolean road-band mask: `axis='row'` fills full-width rows in [band],
    `axis='col'` fills full-height columns in [band]."""
    mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    lo, hi = band
    if axis == "row":
        mask[lo:hi, :] = 1.0
    else:
        mask[:, lo:hi] = 1.0
    return mask


@dataclass
class SceneConfig:
    grid_size: int
    history_len: int
    horizon: int
    max_agents: int
    cross_traffic_prob: float


class Agent:
    """A constant-velocity dynamic agent: position(t) = pos0 + vel * t, with
    t=0 the last-observed frame."""

    def __init__(self, x0: float, y0: float, vx: float, vy: float):
        self.x0, self.y0, self.vx, self.vy = x0, y0, vx, vy

    def pos(self, t: int) -> tuple[float, float]:
        return (self.x0 + self.vx * t, self.y0 + self.vy * t)


ROAD_ROW_BAND = None  # set per-instance below via generate_scene closures


def generate_scene(cfg: SceneConfig, rng: np.random.Generator) -> dict:
    """Generates one synthetic scene. Returns a dict of numpy arrays (caller
    converts to torch tensors -- kept as numpy here so it's easy to unit test
    without a torch dependency in this function)."""
    G = cfg.grid_size
    T = cfg.history_len
    Hz = cfg.horizon
    A = cfg.max_agents

    road_row_band = (G // 2 - 2, G // 2 + 2)   # horizontal road, 4 rows wide
    road_col_band = (G // 2 - 2, G // 2 + 2)   # vertical crossroad, 4 cols wide

    road_mask = np.maximum(
        _line_mask(G, road_row_band, "row"), _line_mask(G, road_col_band, "col")
    )
    lane_mask = np.zeros((G, G), dtype=np.float32)
    lane_mask[(road_row_band[0] + road_row_band[1]) // 2, :] = 1.0
    lane_mask[:, (road_col_band[0] + road_col_band[1]) // 2] = 1.0

    # A few static obstacles (curbs / parked-car proxies), kept off the road.
    static_obstacle_mask = np.zeros((G, G), dtype=np.float32)
    n_obs = rng.integers(2, 5)
    tries = 0
    placed = 0
    while placed < n_obs and tries < 50:
        tries += 1
        oy, ox = rng.integers(1, G - 1, size=2)
        if road_mask[oy, ox] > 0:
            continue
        static_obstacle_mask[oy, ox] = 1.0
        placed += 1

    # --- Ego trajectory: constant-velocity drive along the horizontal road. ---
    ego_row = (road_row_band[0] + road_row_band[1]) // 2 - 1  # a specific lane within the band
    ego_vx = 4.0
    ego_x0 = float(rng.integers(4, 9))
    ego = Agent(x0=ego_x0, y0=float(ego_row), vx=ego_vx, vy=0.0)

    # --- Dynamic agents: lane traffic (family A) + optional cross-traffic hazard (family B). ---
    agents: list[Agent] = []
    n_lane = int(rng.integers(1, 4))
    for _ in range(n_lane):
        row = rng.choice([road_row_band[0], road_row_band[1] - 1])  # opposite lane row
        x0 = float(rng.integers(0, G))
        vx = float(rng.choice([-3.0, -2.0, 2.0, 3.0]))
        agents.append(Agent(x0=x0, y0=float(row), vx=vx, vy=0.0))

    has_hazard = bool(rng.random() < cfg.cross_traffic_prob)
    if has_hazard:
        cross_col = (road_col_band[0] + road_col_band[1]) // 2
        vy = float(rng.choice([2.5, 3.0, 3.5]))
        # Choose y0 so the agent sits well above the road band at t=0 (not yet
        # a visible threat) but arrives inside the ego's row band within the
        # forecast horizon -- i.e. "enters the ego's path" only after the
        # last-observed frame, which is exactly what a persistence baseline
        # (static at t=0's position) will fail to predict.
        target_t = float(rng.integers(1, Hz + 1))
        y0 = ego_row - vy * target_t + rng.normal(0, 0.4)
        agents.append(Agent(x0=float(cross_col), y0=float(y0), vx=0.0, vy=vy))

    n_real_agents = len(agents)
    n_real_agents = min(n_real_agents, A)
    agents = agents[:A]

    # --- Build per-agent history/current/future positions. ---
    agent_history = np.zeros((A, T, 4), dtype=np.float32)  # (x, y, vx, vy), t = -(T-1)..0
    agent_last_pos = np.zeros((A, 2), dtype=np.float32)
    agent_mask = np.zeros((A,), dtype=np.float32)

    for i, ag in enumerate(agents):
        for ti, t in enumerate(range(-(T - 1), 1)):
            x, y = ag.pos(t)
            agent_history[i, ti] = [x / G, y / G, ag.vx / 5.0, ag.vy / 5.0]
        agent_last_pos[i] = ag.pos(0)
        agent_mask[i] = 1.0

    # --- Occupancy ground truth: soft-rasterized footprint of all dynamic
    # agents (not ego) at each timestep t = 0 (last observed) .. horizon. ---
    occ_all = np.zeros((Hz + 1, G, G), dtype=np.float32)
    for t in range(0, Hz + 1):
        pts = [ag.pos(t) for ag in agents]
        occ_all[t] = _soft_raster(G, pts, radius=1.4)

    prev_occupancy = occ_all[0]                    # (G, G) -- last observed frame
    future_occupancy_gt = occ_all[1:]                # (horizon, G, G)

    # BEV current-frame agent footprint channel (t=0 snapshot only, ego excluded).
    agent_footprint_t0 = _soft_raster(G, [ag.pos(0) for ag in agents], radius=1.2)

    bev_grid = np.stack([road_mask, lane_mask, static_obstacle_mask, agent_footprint_t0], axis=0)  # (4, G, G)

    # Risk supervision target: cells that will be occupied at any future step
    # (a simple, easy-to-supervise proxy for "risk" -- see README reconstruction note).
    risk_target = future_occupancy_gt.max(axis=0)  # (G, G)

    ego_nominal_traj = np.array([ego.pos(t) for t in range(1, Hz + 1)], dtype=np.float32)  # (horizon, 2)
    ego_pos0 = np.array(ego.pos(0), dtype=np.float32)

    return {
        "bev_grid": bev_grid,
        "agent_history": agent_history,
        "agent_last_pos": agent_last_pos,
        "agent_mask": agent_mask,
        "prev_occupancy": prev_occupancy[None, ...],          # (1, G, G)
        "future_occupancy_gt": future_occupancy_gt[:, None, ...],  # (horizon, 1, G, G)
        "risk_target": risk_target[None, ...],                 # (1, G, G)
        "ego_nominal_traj": ego_nominal_traj,                     # (horizon, 2)
        "ego_pos0": ego_pos0,                                       # (2,)
        "has_hazard": np.array(has_hazard, dtype=np.bool_),
    }


class RiskWorldSyntheticDataset(Dataset):
    """Deterministic synthetic dataset: scene `idx` is generated from a fixed
    seed derived from (base_seed, idx), so repeated epochs see the same data
    (useful for the overfit-one-batch sanity test) while train/val splits use
    disjoint seed ranges."""

    def __init__(self, cfg: dict, num_scenes: int, base_seed: int):
        data_cfg = cfg["data"]
        self.scene_cfg = SceneConfig(
            grid_size=data_cfg["grid_size"],
            history_len=data_cfg["history_len"],
            horizon=data_cfg["horizon"],
            max_agents=data_cfg["max_agents"],
            cross_traffic_prob=data_cfg["cross_traffic_prob"],
        )
        self.num_scenes = num_scenes
        self.base_seed = base_seed

    def __len__(self) -> int:
        return self.num_scenes

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.base_seed * 100_003 + idx)
        scene = generate_scene(self.scene_cfg, rng)
        return {k: torch.from_numpy(v) for k, v in scene.items()}
