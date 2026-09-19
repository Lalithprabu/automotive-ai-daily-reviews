"""
Procedural synthetic V2X drive-by scene generator for BeamTransFuser.

No public dataset could be recovered for arXiv:2609.10200 (arXiv was
rate-limiting full-text fetches during reconstruction). This module
synthesizes a DeepSense6G-style scenario instead: a vehicle drives past a
roadside unit (RSU) along a straight lane, and at each sampled timestep we
render correlated camera / LiDAR / radar observations plus a GPS trace, and
compute a ground-truth "best beam" label from the true vehicle-to-RSU angle.

This is a reconstruction stand-in for research/benchmarking purposes only --
it does not reproduce any real sensing data or the paper's actual dataset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

MODALITIES = ("camera", "lidar", "radar", "gps")

# Scene geometry (arbitrary but fixed units, "meters")
LANE_OFFSET_M = 10.0      # perpendicular distance of the lane from the RSU
X_RANGE_M = 50.0          # vehicle travels from -X_RANGE_M .. +X_RANGE_M
FOV_MIN_RAD = 0.0         # RSU field of view: 0 .. pi (a full forward half-plane)
FOV_MAX_RAD = math.pi


@dataclass
class SceneBatch:
    camera: torch.Tensor   # (B, 3, 32, 32)
    lidar: torch.Tensor    # (B, 1, 32, 32)
    radar: torch.Tensor    # (B, 1, 16, 16)
    gps: torch.Tensor      # (B, 4)
    presence: dict         # modality -> (B,) bool tensor, True = sensor present
    beam_label: torch.Tensor  # (B,) long, ground-truth best beam index

    def to(self, device):
        self.camera = self.camera.to(device)
        self.lidar = self.lidar.to(device)
        self.radar = self.radar.to(device)
        self.gps = self.gps.to(device)
        self.presence = {k: v.to(device) for k, v in self.presence.items()}
        self.beam_label = self.beam_label.to(device)
        return self


def _angle_to_beam(angle_rad: np.ndarray, num_beams: int) -> np.ndarray:
    """Discretize the RSU-to-vehicle angle (0..pi) into a beam codebook index."""
    clipped = np.clip(angle_rad, FOV_MIN_RAD, FOV_MAX_RAD - 1e-6)
    frac = (clipped - FOV_MIN_RAD) / (FOV_MAX_RAD - FOV_MIN_RAD)
    return np.floor(frac * num_beams).astype(np.int64)


def _render_camera(x, y, dist, angle_frac, rng, shape=(3, 32, 32), noise=0.06):
    c, h, w = shape
    img = rng.normal(0.0, noise, size=shape).astype(np.float32)
    col = angle_frac * (w - 1)
    row = h / 2.0
    # closer vehicles -> bigger, brighter blob
    radius = max(1.5, 6.0 - 0.08 * dist)
    brightness = np.clip(1.4 - dist / 70.0, 0.25, 1.4)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    blob = brightness * np.exp(-(((xx - col) ** 2 + (yy - row) ** 2) / (2 * radius ** 2)))
    for ch in range(c):
        tint = 1.0 - 0.15 * ch
        img[ch] += blob * tint
    return img.astype(np.float32)


def _render_lidar(x, y, rng, shape=(1, 32, 32), noise=0.03, x_range=X_RANGE_M):
    _, h, w = shape
    grid = rng.normal(0.0, noise, size=shape).astype(np.float32)
    gx = int((x + x_range) / (2 * x_range) * (w - 1))
    gy = int(np.clip(y / (LANE_OFFSET_M * 2), 0, 1) * (h - 1))
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    sigma = 1.8
    blob = np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sigma ** 2)))
    grid[0] += blob
    return grid.astype(np.float32)


def _render_radar(dist, speed, rng, shape=(1, 16, 16), noise=0.03,
                   max_range=80.0, max_speed=20.0):
    _, h, w = shape
    rdm = rng.normal(0.0, noise, size=shape).astype(np.float32)
    range_bin = int(np.clip(dist / max_range, 0, 1) * (w - 1))
    doppler_bin = int(np.clip((speed + max_speed) / (2 * max_speed), 0, 1) * (h - 1))
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    sigma = 1.1
    blob = 1.2 * np.exp(-(((xx - range_bin) ** 2 + (yy - doppler_bin) ** 2) / (2 * sigma ** 2)))
    rdm[0] += blob
    return rdm.astype(np.float32)


def generate_batch(
    batch_size: int,
    num_beams: int = 64,
    modality_dropout_prob: float = 0.15,
    force_drop: str | None = None,
    force_drop_all: bool = False,
    rng: np.random.Generator | None = None,
) -> SceneBatch:
    """Generate one synthetic drive-by batch.

    Args:
        force_drop: if set to a modality name, that modality is fully
            dropped (zeroed, presence=False) for every sample in the batch.
            Used by tests / the "sensor drop" simulation segment.
        force_drop_all: if True, ALL samples get force_drop applied and no
            other random dropout is layered in (deterministic drop test).
    """
    if rng is None:
        rng = np.random.default_rng()

    cameras, lidars, radars, gpss = [], [], [], []
    presence = {m: np.ones(batch_size, dtype=bool) for m in MODALITIES}
    beam_labels = np.zeros(batch_size, dtype=np.int64)

    for i in range(batch_size):
        t = rng.uniform(0.0, 1.0)
        x = -X_RANGE_M + 2 * X_RANGE_M * t
        y = LANE_OFFSET_M
        dist = math.hypot(x, y)
        # atan2(y, x) sweeps ~0..pi as x goes from -X_RANGE..+X_RANGE with
        # y > 0 fixed, giving a smooth left->right angular sweep past the RSU.
        angle = math.atan2(y, x)
        angle_frac = np.clip(angle / math.pi, 0.0, 1.0)

        speed = float(np.clip(rng.normal(8.0, 2.0), 0.5, 20.0))
        heading = float(rng.normal(0.0, 0.15))

        cam = _render_camera(x, y, dist, angle_frac, rng)
        lid = _render_lidar(x, y, rng)
        rad = _render_radar(dist, speed, rng)
        gps_vec = np.array(
            [x / X_RANGE_M, (y - LANE_OFFSET_M) / LANE_OFFSET_M, speed / 20.0, heading / math.pi],
            dtype=np.float32,
        )

        label = int(_angle_to_beam(np.array([angle]), num_beams)[0])
        # small label noise to emulate real-world beam-selection jitter
        if rng.uniform() < 0.03:
            label = int(np.clip(label + rng.integers(-1, 2), 0, num_beams - 1))

        cameras.append(cam)
        lidars.append(lid)
        radars.append(rad)
        gpss.append(gps_vec)
        beam_labels[i] = label

        for m in MODALITIES:
            drop = False
            if force_drop is not None and (force_drop_all or rng.uniform() < 0.5) and m == force_drop:
                drop = True
            elif force_drop is None and rng.uniform() < modality_dropout_prob:
                drop = True
            presence[m][i] = not drop

    camera_t = torch.from_numpy(np.stack(cameras))
    lidar_t = torch.from_numpy(np.stack(lidars))
    radar_t = torch.from_numpy(np.stack(radars))
    gps_t = torch.from_numpy(np.stack(gpss))

    presence_t = {}
    for m, arr in presence.items():
        mask = torch.from_numpy(arr)
        presence_t[m] = mask

    # zero out the raw tensors for dropped samples (the encoder still runs,
    # the imputer is responsible for producing something usable downstream)
    camera_t[~presence_t["camera"]] = 0.0
    lidar_t[~presence_t["lidar"]] = 0.0
    radar_t[~presence_t["radar"]] = 0.0
    gps_t[~presence_t["gps"]] = 0.0

    return SceneBatch(
        camera=camera_t,
        lidar=lidar_t,
        radar=radar_t,
        gps=gps_t,
        presence=presence_t,
        beam_label=torch.from_numpy(beam_labels),
    )


def generate_drive_sequence(n_frames: int, num_beams: int = 64, seed: int = 0):
    """Generate a single continuous drive-by sequence (one 'vehicle') for
    the simulation script: a smooth sweep of the vehicle across the RSU's
    field of view, frame by frame, with no random modality dropout applied
    here (the simulation script controls drops explicitly per-frame)."""
    rng = np.random.default_rng(seed)
    frames = []
    for f in range(n_frames):
        t = f / max(n_frames - 1, 1)
        x = -X_RANGE_M + 2 * X_RANGE_M * t
        y = LANE_OFFSET_M
        dist = math.hypot(x, y)
        angle = math.atan2(y, x)
        angle_frac = np.clip(angle / math.pi, 0.0, 1.0)
        speed = float(np.clip(8.0 + 1.5 * math.sin(t * 6.0), 0.5, 20.0))
        heading = 0.05 * math.sin(t * 4.0)

        cam = _render_camera(x, y, dist, angle_frac, rng)
        lid = _render_lidar(x, y, rng)
        rad = _render_radar(dist, speed, rng)
        gps_vec = np.array(
            [x / X_RANGE_M, (y - LANE_OFFSET_M) / LANE_OFFSET_M, speed / 20.0, heading / math.pi],
            dtype=np.float32,
        )
        label = int(_angle_to_beam(np.array([angle]), num_beams)[0])
        frames.append(dict(x=x, y=y, dist=dist, angle=angle, speed=speed,
                            camera=cam, lidar=lid, radar=rad, gps=gps_vec, label=label))
    return frames
