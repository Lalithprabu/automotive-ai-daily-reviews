"""
Synthetic BEV scene generator for DRiF reconstruction.

Scenario (matches the spec): ego holding lane, a lead vehicle ahead, and a
cyclist crossing the ego's path. Everything is generated in an ego-centered
metric frame: x_forward in [-BEV_RANGE_M/2, +BEV_RANGE_M/2] (ahead of ego is
positive), y_lateral in the same range (left/right of ego's heading).

For every scene we produce:
  - a rasterized multi-channel BEV grid (encoder input)
  - a static map ground truth (drivable / lane-boundary / background)
  - a dense continuous "risk signature" oracle (used only to *derive*
    pairwise labels + for visualization -- never regressed against directly,
    per the paper's pairwise-ranking-only supervision)
  - PAIRWISE risk labels: sampled point indices (idx_i, idx_j) + y_ij in
    {+1, 0, -1}, derived from a 3-tier priority order: overlap > corridor >
    occupancy-risk (with background lowest of all)
  - a classical fixed-shape potential-field baseline (constant-sigma
    Gaussian bump per dynamic agent) -- the "OLD WAY" reconstruction
  - a planning ground-truth trajectory that lane-keeps but nudges laterally
    away from the ego-cyclist conflict zone
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.models.risk_head import BEV_RANGE_M

# Priority tiers (paper: overlap > corridor > occupancy-risk)
TIER_BACKGROUND = 0
TIER_OCCUPANCY = 1
TIER_CORRIDOR = 2
TIER_OVERLAP = 3

LANE_HALF_WIDTH_M = 3.5      # ego lane corridor half-width
AGENT_FOOTPRINT_M = 2.0       # rough vehicle/cyclist half-length used for occupancy radius
CONFLICT_RADIUS_M = 4.5        # radius around the cyclist that counts as "overlap" w/ ego corridor
CLASSICAL_SIGMA_M = 3.0        # fixed-shape baseline Gaussian sigma (paper's naive baseline)


def _agent_positions(t: int, total_steps: int = 12) -> Dict[str, np.ndarray]:
    """Deterministic scripted motion for the 3 agents at integer timestep t,
    in ego-centered meters (x_forward, y_lateral). Ego is fixed at the origin
    of its own frame; lead vehicle and cyclist move relative to it.
    """
    frac = t / max(total_steps - 1, 1)

    ego = np.array([0.0, 0.0])
    # Lead vehicle: holds ~15m ahead, slow closing drift.
    lead = np.array([16.0 - 3.0 * frac, 0.0])
    # Cyclist: crosses the ego's forward corridor, ~9m ahead, sweeping
    # laterally from one side to the other.
    cyclist = np.array([9.0, -9.0 + 18.0 * frac])

    return {"ego": ego, "lead": lead, "cyclist": cyclist}


def _history(t: int, key: str, total_steps: int = 12, n_hist: int = 4) -> np.ndarray:
    """Last `n_hist` positions (including current) for trailing dotted paths."""
    pts = []
    for dt in range(n_hist - 1, -1, -1):
        tt = max(t - dt, 0)
        pts.append(_agent_positions(tt, total_steps)[key])
    return np.stack(pts, axis=0)  # [n_hist, 2]


def _grid_coords(grid_size: int, range_m: float) -> np.ndarray:
    """[G, G, 2] array of (x_forward, y_lateral) meters at each cell center."""
    lin = (np.arange(grid_size) + 0.5) / grid_size * range_m - range_m / 2.0
    xs, ys = np.meshgrid(lin, lin, indexing="ij")  # xs varies along forward axis
    return np.stack([xs, ys], axis=-1)


def _gaussian_bump(coords: np.ndarray, center: np.ndarray, sigma: float) -> np.ndarray:
    d2 = ((coords - center) ** 2).sum(axis=-1)
    return np.exp(-d2 / (2.0 * sigma ** 2))


@dataclass
class SceneConfig:
    grid_size: int = 64
    range_m: float = BEV_RANGE_M
    in_channels: int = 5
    num_risk_points: int = 96
    num_pairs: int = 64
    planning_horizon: int = 8
    total_steps: int = 12
    seed: int = 0


class SyntheticSceneGenerator:
    """Generates ego/lead-vehicle/cyclist BEV scenes with pairwise risk labels."""

    def __init__(self, cfg: SceneConfig):
        self.cfg = cfg
        self.coords = _grid_coords(cfg.grid_size, cfg.range_m)  # [G, G, 2]

    # ------------------------------------------------------------------
    def _tier_at_points(self, points_xy: np.ndarray, agents: Dict[str, np.ndarray]) -> np.ndarray:
        """3-tier (+background) priority label for each sampled point."""
        tiers = np.full(points_xy.shape[0], TIER_BACKGROUND, dtype=np.int64)

        # Occupancy-risk: near any agent's footprint.
        for key in ("lead", "cyclist"):
            d = np.linalg.norm(points_xy - agents[key], axis=-1)
            tiers = np.where(d < AGENT_FOOTPRINT_M * 2.5, np.maximum(tiers, TIER_OCCUPANCY), tiers)

        # Corridor: inside ego's forward lane corridor (|y| < half width, x > 0).
        in_corridor = (np.abs(points_xy[:, 1]) < LANE_HALF_WIDTH_M) & (points_xy[:, 0] > 0)
        tiers = np.where(in_corridor, np.maximum(tiers, TIER_CORRIDOR), tiers)

        # Overlap: inside the corridor AND close to the cyclist's current conflict zone.
        d_cyc = np.linalg.norm(points_xy - agents["cyclist"], axis=-1)
        overlap = in_corridor & (d_cyc < CONFLICT_RADIUS_M)
        tiers = np.where(overlap, TIER_OVERLAP, tiers)

        return tiers

    def _risk_signature(self, agents: Dict[str, np.ndarray]) -> np.ndarray:
        """Dense continuous oracle risk field [G, G], consistent with the
        discrete tier ordering (used only to derive labels / for plotting).
        """
        coords = self.coords.reshape(-1, 2)
        tiers = self._tier_at_points(coords, agents).astype(np.float32)

        # Smooth continuous component so the oracle isn't piecewise-constant:
        # occupancy bumps around agents + a mild corridor gradient.
        cont = np.zeros(coords.shape[0], dtype=np.float32)
        cont += 0.6 * _gaussian_bump(coords, agents["lead"], AGENT_FOOTPRINT_M * 1.5)
        cont += 0.8 * _gaussian_bump(coords, agents["cyclist"], AGENT_FOOTPRINT_M * 1.5)

        field = tiers + cont
        field = field / (TIER_OVERLAP + 1.0)  # normalize roughly into [0, ~1.3]
        return field.reshape(self.cfg.grid_size, self.cfg.grid_size)

    def _classical_baseline(self, agents: Dict[str, np.ndarray]) -> np.ndarray:
        """Fixed-radius Gaussian bump per dynamic agent -- constant shape
        regardless of motion/geometry. This is the 'OLD WAY' baseline.
        """
        coords = self.coords.reshape(-1, 2)
        field = np.zeros(coords.shape[0], dtype=np.float32)
        for key in ("lead", "cyclist"):
            field += _gaussian_bump(coords, agents[key], CLASSICAL_SIGMA_M)
        return field.reshape(self.cfg.grid_size, self.cfg.grid_size)

    def _static_map_gt(self) -> np.ndarray:
        """0 = background, 1 = drivable, 2 = lane-boundary."""
        coords = self.coords
        ax = np.abs(coords[..., 1])
        gt = np.zeros(coords.shape[:2], dtype=np.int64)
        gt[ax < LANE_HALF_WIDTH_M] = 1
        boundary = (ax >= LANE_HALF_WIDTH_M) & (ax < LANE_HALF_WIDTH_M + 1.0)
        gt[boundary] = 2
        return gt

    def _bev_grid(self, agents: Dict[str, np.ndarray], hist: Dict[str, np.ndarray]) -> np.ndarray:
        """Rasterize occupancy / lane-corridor / ego-history-x / ego-history-y
        / velocity-magnitude channels onto the grid. Shape: [C_in, G, G].
        """
        G = self.cfg.grid_size
        coords_flat = self.coords.reshape(-1, 2)

        occ = np.zeros(coords_flat.shape[0], dtype=np.float32)
        for key in ("lead", "cyclist"):
            occ += _gaussian_bump(coords_flat, agents[key], AGENT_FOOTPRINT_M)
        occ = np.clip(occ, 0, 1).reshape(G, G)

        lane = (np.abs(self.coords[..., 1]) < LANE_HALF_WIDTH_M).astype(np.float32)

        ego_hist = hist["ego"]  # [n_hist, 2]
        hist_x = np.zeros((G, G), dtype=np.float32)
        hist_y = np.zeros((G, G), dtype=np.float32)
        for p in ego_hist:
            bump = _gaussian_bump(coords_flat, p, 1.0).reshape(G, G)
            hist_x += bump * (p[0] / (self.cfg.range_m / 2))
            hist_y += bump * (p[1] / (self.cfg.range_m / 2))

        vel = np.zeros((G, G), dtype=np.float32)
        cyc_speed = np.linalg.norm(hist["cyclist"][-1] - hist["cyclist"][0])
        vel += _gaussian_bump(coords_flat, agents["cyclist"], AGENT_FOOTPRINT_M * 2).reshape(G, G) * cyc_speed / 10.0

        grid = np.stack([occ, lane, hist_x, hist_y, vel], axis=0)
        assert grid.shape[0] == self.cfg.in_channels
        return grid.astype(np.float32)

    def _planning_gt(self, agents: Dict[str, np.ndarray]) -> np.ndarray:
        """Lane-keeping trajectory that nudges laterally away from the
        ego-cyclist conflict zone (classical risk-avoidance target).
        """
        horizon = self.cfg.planning_horizon
        xs = np.linspace(2.0, self.cfg.range_m / 2 - 4.0, horizon)
        ys = np.zeros(horizon, dtype=np.float32)

        cyc = agents["cyclist"]
        for i, x in enumerate(xs):
            d = math.hypot(x - cyc[0], 0.0 - cyc[1])
            if d < CONFLICT_RADIUS_M * 1.5:
                push = -np.sign(cyc[1]) if cyc[1] != 0 else 1.0
                strength = max(0.0, 1.0 - d / (CONFLICT_RADIUS_M * 1.5))
                ys[i] = push * strength * 1.5
        return np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=-1)

    # ------------------------------------------------------------------
    def generate(self, t: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        """Generate one full scene at timestep t (t in [0, total_steps))."""
        agents = _agent_positions(t, self.cfg.total_steps)
        hist = {k: _history(t, k, self.cfg.total_steps) for k in agents}

        bev_grid = self._bev_grid(agents, hist)
        map_gt = self._static_map_gt()
        risk_signature = self._risk_signature(agents)
        classical_map = self._classical_baseline(agents)
        planning_gt = self._planning_gt(agents)

        # Sample N points uniformly in the BEV window for pairwise supervision.
        half = self.cfg.range_m / 2
        points_xy = rng.uniform(-half, half, size=(self.cfg.num_risk_points, 2)).astype(np.float32)
        tiers = self._tier_at_points(points_xy, agents)

        # Sample P pairs; label by tier ordering (paper's 3-tier priority).
        P = self.cfg.num_pairs
        idx_i = rng.integers(0, self.cfg.num_risk_points, size=P)
        idx_j = rng.integers(0, self.cfg.num_risk_points, size=P)
        tier_diff = tiers[idx_i] - tiers[idx_j]
        y_ij = np.sign(tier_diff).astype(np.float32)

        return {
            "bev_grid": bev_grid,                          # [C_in, G, G]
            "map_gt": map_gt,                                 # [G, G] int64
            "risk_signature": risk_signature,                  # [G, G] float32
            "classical_map": classical_map,                     # [G, G] float32
            "points_xy": points_xy,                               # [N, 2] float32
            "point_tiers": tiers.astype(np.int64),                 # [N]
            "idx_i": idx_i.astype(np.int64),                        # [P]
            "idx_j": idx_j.astype(np.int64),                        # [P]
            "y_ij": y_ij,                                              # [P] in {-1,0,1}
            "planning_gt": planning_gt,                                 # [horizon, 2]
            "agents": {k: v.astype(np.float32) for k, v in agents.items()},
            "history": {k: v.astype(np.float32) for k, v in hist.items()},
            "t": t,
        }

    def generate_batch(self, batch_size: int, rng: np.random.Generator, t: int | None = None) -> List[Dict[str, np.ndarray]]:
        """Generate a batch of independent scenes. If `t` is None, each scene
        gets a random timestep (used for training); a fixed `t` is used to
        render a synced animation frame across a batch (used for simulate.py).
        """
        scenes = []
        for _ in range(batch_size):
            tt = t if t is not None else int(rng.integers(0, self.cfg.total_steps))
            scenes.append(self.generate(tt, rng))
        return scenes
