"""
Synthetic pedestrian-crossing scene generator.

Real TrajFusionNet+ training reads PIE/JAAD dashcam footage with a semantic
segmentation model upstream to extract the scene graph. Neither dataset nor
segmentation model is bundled here (out of scope for a from-scratch repo
package), so this module generates a physically-plausible SYNTHETIC scene
instead: a pedestrian approaching a curb near a marked crosswalk with zero,
one, or two nearby vehicles, used for (a) `train.py`'s smoke-test training
loop and (b) `simulate.py`'s rendered overlay animation. Every downstream
model input (past trajectory, frame overlays, scene graph) is derived
consistently from the same underlying synthetic scenario.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


PEDESTRIAN, VEHICLE, CROSSWALK, OTHER_STATIC = 0, 1, 2, 3


@dataclass
class SyntheticScene:
    past_traj: np.ndarray          # [past_len, 5]  (x1,y1,x2,y2,speed), normalized image coords
    future_traj: np.ndarray        # [future_len, 5]
    will_cross: int                # 0 / 1 ground truth label
    node_positions: np.ndarray     # [N, 2]
    node_classes: np.ndarray       # [N]
    node_areas: np.ndarray         # [N]
    node_valid: np.ndarray         # [N]
    frame_size: int = 128


def generate_scene(rng: np.random.Generator, past_len: int = 15, future_len: int = 60,
                    max_nodes: int = 16) -> SyntheticScene:
    """Simulates a pedestrian walking toward (or lingering near) a curb,
    with a crosswalk and 0-3 vehicles scattered in the scene. Crossing intent
    is made to (loosely, realistically) depend on proximity to the crosswalk
    and to the nearest approaching vehicle, so the task isn't purely random."""
    total_len = past_len + future_len

    start_x = rng.uniform(0.1, 0.9)
    start_y = rng.uniform(0.55, 0.75)  # pedestrians occupy the lower half of the frame (near-camera)
    crosswalk_x = rng.uniform(0.3, 0.7)
    curb_y = rng.uniform(0.35, 0.5)

    heading_toward_curb = rng.uniform(0.6, 1.0)  # how directly the pedestrian walks toward the curb
    speed = rng.uniform(0.005, 0.02)  # normalized units / frame

    xs, ys = [start_x], [start_y]
    for _ in range(total_len - 1):
        dx = (crosswalk_x - xs[-1]) * 0.02 * heading_toward_curb + rng.normal(0, 0.003)
        dy = -abs(speed) + rng.normal(0, 0.002)
        xs.append(float(np.clip(xs[-1] + dx, 0.02, 0.98)))
        ys.append(float(np.clip(ys[-1] + dy, 0.05, 0.95)))
    xs, ys = np.array(xs), np.array(ys)

    box_w = rng.uniform(0.03, 0.06)
    box_h = box_w * rng.uniform(1.8, 2.4)  # pedestrian bounding boxes are taller than wide
    speeds = np.gradient(ys) * -1.0  # proxy for approach speed toward the camera/curb

    x1 = xs - box_w / 2
    y1 = ys - box_h / 2
    x2 = xs + box_w / 2
    y2 = ys + box_h / 2
    traj = np.stack([x1, y1, x2, y2, speeds], axis=-1).astype(np.float32)  # [total_len, 5]

    past_traj = traj[:past_len]
    future_traj = traj[past_len:]

    dist_to_crosswalk_at_end = abs(xs[past_len - 1] - crosswalk_x)
    near_curb = ys[past_len - 1] < curb_y + 0.1

    n_vehicles = int(rng.integers(0, 3))
    node_positions, node_classes, node_areas = [np.array([xs[past_len - 1], ys[past_len - 1]])], [PEDESTRIAN], [box_w * box_h]
    nearest_vehicle_dist = 1.0
    for _ in range(n_vehicles):
        vx, vy = rng.uniform(0.1, 0.9), rng.uniform(0.15, 0.5)
        node_positions.append(np.array([vx, vy]))
        node_classes.append(VEHICLE)
        node_areas.append(rng.uniform(0.02, 0.08))
        d = math.dist((vx, vy), (xs[past_len - 1], ys[past_len - 1]))
        nearest_vehicle_dist = min(nearest_vehicle_dist, d)

    node_positions.append(np.array([crosswalk_x, curb_y]))
    node_classes.append(CROSSWALK)
    node_areas.append(rng.uniform(0.05, 0.12))

    if rng.random() < 0.4:
        node_positions.append(np.array([rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.3)]))
        node_classes.append(OTHER_STATIC)
        node_areas.append(rng.uniform(0.01, 0.05))

    n_nodes = len(node_positions)
    node_positions = np.stack(node_positions).astype(np.float32)
    node_classes = np.array(node_classes, dtype=np.int64)
    node_areas = np.array(node_areas, dtype=np.float32)

    # Ground-truth crossing rule: likely to cross if near the crosswalk, close to the
    # curb, and no vehicle is dangerously close (a rough, deliberately-simplified proxy).
    cross_score = (1.0 - dist_to_crosswalk_at_end) * 0.5 + float(near_curb) * 0.3 + (nearest_vehicle_dist) * 0.2
    will_cross = int(cross_score > 0.55)

    pad = max_nodes - n_nodes
    node_valid = np.ones(n_nodes, dtype=np.float32)
    if pad > 0:
        node_positions = np.concatenate([node_positions, np.zeros((pad, 2), dtype=np.float32)])
        node_classes = np.concatenate([node_classes, np.zeros(pad, dtype=np.int64)])
        node_areas = np.concatenate([node_areas, np.zeros(pad, dtype=np.float32)])
        node_valid = np.concatenate([node_valid, np.zeros(pad, dtype=np.float32)])
    else:
        node_positions, node_classes = node_positions[:max_nodes], node_classes[:max_nodes]
        node_areas, node_valid = node_areas[:max_nodes], node_valid[:max_nodes]

    return SyntheticScene(past_traj, future_traj.astype(np.float32), will_cross,
                           node_positions, node_classes, node_areas, node_valid)


def scene_batch_to_tensors(scenes: list[SyntheticScene], frame_size: int = 128):
    """Stacks a list of SyntheticScene into batched tensors ready for
    TrajFusionNetPlus.forward(). Frame overlays are rendered as simple
    filled-rectangle bounding-box masks (RGB) rather than real dashcam
    imagery, so VAM has a real (if synthetic) image to convolve over."""
    B = len(scenes)
    past_traj = torch.tensor(np.stack([s.past_traj for s in scenes]))
    future_traj = torch.tensor(np.stack([s.future_traj for s in scenes]))
    will_cross = torch.tensor([s.will_cross for s in scenes], dtype=torch.long)
    node_positions = torch.tensor(np.stack([s.node_positions for s in scenes]))
    node_classes = torch.tensor(np.stack([s.node_classes for s in scenes]))
    node_areas = torch.tensor(np.stack([s.node_areas for s in scenes]))
    node_valid = torch.tensor(np.stack([s.node_valid for s in scenes]))

    def render_bbox_frame(box_xyxy: np.ndarray) -> np.ndarray:
        img = np.full((3, frame_size, frame_size), 0.15, dtype=np.float32)  # dark "road" background
        x1, y1, x2, y2 = (box_xyxy[:4] * frame_size).astype(int)
        x1, x2 = sorted((np.clip(x1, 0, frame_size - 1), np.clip(x2, 0, frame_size - 1)))
        y1, y2 = sorted((np.clip(y1, 0, frame_size - 1), np.clip(y2, 0, frame_size - 1)))
        img[0, y1:y2 + 1, x1:x2 + 1] = 0.9  # highlight the bbox region in the red channel
        return img

    observed_frame = torch.tensor(np.stack([render_bbox_frame(s.past_traj[0]) for s in scenes]))
    predicted_frame = torch.tensor(np.stack([render_bbox_frame(s.past_traj[-1]) for s in scenes]))

    return {
        "past_traj": past_traj, "future_traj": future_traj, "will_cross": will_cross,
        "node_positions": node_positions, "node_classes": node_classes,
        "node_areas": node_areas, "node_valid": node_valid,
        "observed_frame": observed_frame, "predicted_frame": predicted_frame,
    }
