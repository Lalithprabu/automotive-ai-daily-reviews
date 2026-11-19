"""Kinematic / naive forecasting baselines for the last n_future patches of a window, plus forecast metrics.

    vehicle: constant-velocity in the road frame -> lateral offset continues its least-squares slope over the last
             1 s, speed held, steering held, acceleration 0
    gaze:    hold the mean gaze of the last 0.3 s (a fixation persists)
"""
from __future__ import annotations

import numpy as np

DT = 0.1


def kinematic_forecast(gaze: np.ndarray, veh: np.ndarray, n_steps: int):
    """gaze (B,T,2), veh (B,T,4) with the last n_steps unknown -> copies with the future filled in."""
    g, v = gaze.copy(), veh.copy()
    T = g.shape[1]
    t0 = T - n_steps                                                   # first future index
    g[:, t0:] = g[:, t0 - 3:t0].mean(1, keepdims=True)
    tt = np.arange(-10, 0) * DT                                        # last 1 s
    y = v[:, t0 - 10:t0, 0]
    A = np.stack([np.ones_like(tt), tt], -1)
    coef = np.linalg.lstsq(A, y.T, rcond=None)[0]                      # (2, B)
    fut_t = np.arange(0, n_steps) * DT                                 # t0 is 0.1 s after the last sample (-0.1 s)
    v[:, t0:, 0] = coef[0][:, None] + coef[1][:, None] * fut_t[None]   # extrapolated lateral offset
    v[:, t0:, 1] = v[:, t0 - 1:t0, 1]                                  # speed held
    v[:, t0:, 2] = v[:, t0 - 1:t0, 2]                                  # steering held
    v[:, t0:, 3] = 0.0                                                 # no acceleration
    return g, v


def road_frame_xy(veh: np.ndarray, t0: int) -> np.ndarray:
    """Future positions (B, T-t0, 2) in the road frame relative to the car at t0-1: x = integral of speed, y = lateral."""
    speed = veh[:, t0:, 1] * 5.0 + 25.0
    x = np.cumsum(speed * DT, 1)
    return np.stack([x, veh[:, t0:, 0]], -1)


def forecast_metrics(pred_g, pred_v, gaze, veh, n_steps: int) -> dict:
    """ADE / FDE of the road-frame position (m), lateral ADE (m), speed RMSE (m/s), gaze angular error (deg)."""
    t0 = gaze.shape[1] - n_steps
    pe = np.linalg.norm(road_frame_xy(pred_v, t0) - road_frame_xy(veh, t0), axis=-1)    # (B, n)
    ge = np.linalg.norm((pred_g[:, t0:] - gaze[:, t0:]) * 30.0, axis=-1)                 # degrees
    return dict(pos_ade=pe.mean(1), pos_fde=pe[:, -1],
                lat_ade=np.abs(pred_v[:, t0:, 0] - veh[:, t0:, 0]).mean(1),
                speed_rmse=np.sqrt((((pred_v[:, t0:, 1] - veh[:, t0:, 1]) * 5.0) ** 2).mean(1)),
                gaze_deg=ge.mean(1))
