"""Synthetic asynchronous vehicle-infrastructure world (NOT V2X-Seq).

Ego frame, 10 Hz.  K vehicles drive along gently curving paths.  Two sensors:
  * ego vehicle  : sees objects within ``ego_range`` of the origin (noise sigma_ego)
  * roadside unit: sees objects within ``rsu_range`` of its mount; every box in a message
                   is shifted by the SAME pose error e(t) (random walk + rare jumps)
                   and made stale by ``delay_ms`` (extrapolation noise grows with delay)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WorldConfig:
    steps: int = 120
    dt: float = 0.1
    n_objects: int = 10
    ego_range: float = 15.0
    rsu_pos: tuple = (22.0, 6.0)
    rsu_range: float = 26.0
    sigma_ego: float = 0.25
    sigma_rsu: float = 0.25
    miss_prob: float = 0.08
    false_pos_rate: float = 0.15           # expected false boxes per sensor per frame
    err_walk_std: float = 0.05             # pose error random-walk step std [m / sqrt(step)]
    err_jump_prob: float = 0.01            # chance per step of a jump
    err_jump_std: float = 0.8              # jump size [m]
    err_init_std: float = 1.5              # initial |error| scale [m]
    vel_err_std: float = 1.0               # per-object velocity error [m/s] -> stale extrapolation
    delay_ms: float = 200.0


@dataclass
class Frame:
    t: int
    gt_xy: np.ndarray                      # (K, 2)
    gt_vis_ego: np.ndarray                 # (K,) bool
    gt_vis_rsu: np.ndarray                 # (K,) bool
    ego_xy: np.ndarray                     # (Ne, 2)
    road_xy: np.ndarray                    # (Nr, 2) raw (shifted) roadside boxes
    true_err: np.ndarray                   # (2,) shared pose error actually applied


def _paths(rng, cfg: WorldConfig):
    K = cfg.n_objects
    lane_y = rng.uniform(-18, 22, K)
    start_x = rng.uniform(-10, 45, K)
    speed = rng.uniform(3.0, 9.0, K) * rng.choice([-1, 1], K)
    curve = rng.uniform(-0.08, 0.08, K)
    return lane_y, start_x, speed, curve


def generate(seed: int, cfg: WorldConfig | None = None) -> list[Frame]:
    cfg = cfg or WorldConfig()
    rng = np.random.default_rng(seed)
    lane_y, start_x, speed, curve = _paths(rng, cfg)
    rsu = np.array(cfg.rsu_pos)
    err = rng.normal(0, cfg.err_init_std, 2)
    frames: list[Frame] = []
    for t in range(cfg.steps):
        tt = t * cfg.dt
        x = start_x + speed * tt
        x = 5 + (x - 5) % 60 - 10                                           # wrap into [-5, 55)
        y = lane_y + 3.0 * np.sin(curve * x)
        gt = np.stack([x, y], 1)

        err = err + rng.normal(0, cfg.err_walk_std, 2)
        if rng.random() < cfg.err_jump_prob:
            err = err + rng.normal(0, cfg.err_jump_std, 2)

        vis_e = np.linalg.norm(gt, axis=1) < cfg.ego_range
        vis_r = np.linalg.norm(gt - rsu, axis=1) < cfg.rsu_range

        det_e = gt[vis_e & (rng.random(len(gt)) > cfg.miss_prob)]
        det_e = det_e + rng.normal(0, cfg.sigma_ego, det_e.shape)
        det_r = gt[vis_r & (rng.random(len(gt)) > cfg.miss_prob)]
        stale = rng.normal(0, cfg.vel_err_std, det_r.shape) * (cfg.delay_ms / 1000.0)
        det_r = det_r + rng.normal(0, cfg.sigma_rsu, det_r.shape) + stale + err

        for _ in range(rng.poisson(cfg.false_pos_rate)):                    # clutter
            det_e = np.vstack([det_e, rng.uniform(-cfg.ego_range, cfg.ego_range, (1, 2))])
        for _ in range(rng.poisson(cfg.false_pos_rate)):
            det_r = np.vstack([det_r, rsu + rng.uniform(-cfg.rsu_range, cfg.rsu_range, (1, 2)) + err])

        frames.append(Frame(t, gt, vis_e, vis_r, det_e, det_r, err.copy()))
    return frames
