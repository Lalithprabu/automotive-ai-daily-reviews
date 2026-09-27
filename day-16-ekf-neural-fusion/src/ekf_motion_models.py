"""
Classical Extended Kalman Filter (EKF) motion models used as the "physically
feasible trajectory candidates" side of the fusion framework described in the
abstract of:

    Kim & Kong, "Vehicle Trajectory Prediction via Neural Fusion of Multiple
    EKF-Based Trajectory Candidates", arXiv:2609.19813 (2026).

The abstract states the paper fuses a neural predictor (Trajectron++) with
"extended Kalman filter (EKF)-based multiple trajectory candidates" -- it does
not specify which motion models are used. CV/CA/CTRV/CTRA is the standard,
textbook set of motion models used across the vehicle-tracking / trajectory
prediction literature for exactly this purpose (e.g. Schubert et al. 2008,
and it is the same family used in nuScenes-style baselines), so we use it here
as a faithful, well-established reconstruction choice -- NOT a value taken
from the paper's (unavailable) full text.

Pipeline per model, all pure NumPy (no torch):
    1. Initialize a state vector from the observed position history using
       finite differences (position -> velocity -> heading -> yaw rate/accel).
    2. Run the EKF *predict + update* loop over the history window, using the
       observed positions as noisy measurements. This lets each filter
       converge to a de-noised estimate of the agent's current kinematic
       state and reduces the influence of the arbitrary finite-difference
       initialization.
    3. Roll the state forward for `future_len` steps using EKF *predict only*
       (no measurement is available for the future) to produce the mean
       future trajectory, propagating the covariance forward at each step
       (P' = F P F^T + Q) so uncertainty grows over the horizon.

Shapes:
    history_xy : np.ndarray [T_hist, 2]   observed (x, y) positions
    mean_traj  : np.ndarray [T_fut, 2]    predicted (x, y) positions
    cov_trace  : np.ndarray [T_fut]       trace of the 2x2 position block of P
                                           at each future step (a scalar
                                           uncertainty proxy per step)
"""

from __future__ import annotations

import numpy as np


def _finite_diff_kinematics(history_xy: np.ndarray, dt: float):
    """Estimate position/velocity/heading/yaw-rate/acceleration from a short
    history window via finite differences. Used only to *initialize* each
    EKF's state vector before the predict/update loop takes over.

    history_xy: [T, 2]
    returns: dict of scalars/arrays (x, y, vx, vy, v, theta, omega, a)
    """
    T = history_xy.shape[0]
    assert T >= 3, "need at least 3 history points for finite differences"

    # velocity via central-ish differences on the last few steps: [T-1, 2]
    vel = (history_xy[1:] - history_xy[:-1]) / dt  # [T-1, 2]
    vx, vy = vel[-1, 0], vel[-1, 1]
    v = float(np.hypot(vx, vy))
    theta = float(np.arctan2(vy, vx)) if v > 1e-6 else 0.0

    # heading sequence to estimate yaw rate
    headings = np.arctan2(vel[:, 1], vel[:, 0])
    # unwrap to avoid +-pi discontinuities before differencing
    headings = np.unwrap(headings)
    if len(headings) >= 2:
        omega = float((headings[-1] - headings[-2]) / dt)
    else:
        omega = 0.0

    # speed sequence to estimate longitudinal acceleration
    speeds = np.hypot(vel[:, 0], vel[:, 1])
    if len(speeds) >= 2:
        a = float((speeds[-1] - speeds[-2]) / dt)
    else:
        a = 0.0

    return dict(
        x=float(history_xy[-1, 0]),
        y=float(history_xy[-1, 1]),
        vx=float(vx),
        vy=float(vy),
        v=v,
        theta=theta,
        omega=omega,
        a=a,
    )


class _BaseEKF:
    """Shared EKF machinery: predict/update loop + numerical Jacobian.

    Subclasses implement:
        state_dim         -> int
        init_state(kin)   -> np.ndarray [state_dim]
        state_transition(state, dt) -> np.ndarray [state_dim]  (nonlinear f)
        process_noise(dt) -> np.ndarray [state_dim, state_dim] (Q)
        H                 -> np.ndarray [2, state_dim] (measurement matrix,
                                                          linear: extracts x,y)
    """

    state_dim: int

    def __init__(self, process_noise_scale: dict, meas_noise_std: float = 0.1):
        self.process_noise_scale = process_noise_scale
        # R: measurement noise covariance for a 2D position observation.
        self.R = np.eye(2) * (meas_noise_std ** 2)

    # ---- to be implemented by subclasses -------------------------------
    def init_state(self, kin: dict) -> np.ndarray:
        raise NotImplementedError

    def state_transition(self, state: np.ndarray, dt: float) -> np.ndarray:
        raise NotImplementedError

    def process_noise(self, dt: float) -> np.ndarray:
        raise NotImplementedError

    @property
    def H(self) -> np.ndarray:
        # Measurement model is linear: z = H x = (x_pos, y_pos).
        H = np.zeros((2, self.state_dim))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        return H

    # ---- shared EKF math -------------------------------------------------
    def _numerical_jacobian(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Numerically differentiate `state_transition` w.r.t. `state` to get
        the Jacobian F = df/dx at the current operating point. Using a
        numerical Jacobian (central differences) avoids hand-derived-algebra
        bugs for the nonlinear CTRV/CTRA models while remaining a faithful
        EKF linearization (the same object an analytic Jacobian would give,
        to O(eps^2) accuracy).
        """
        n = self.state_dim
        F = np.zeros((n, n))
        eps = 1e-5
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            f_plus = self.state_transition(state + dx, dt)
            f_minus = self.state_transition(state - dx, dt)
            F[:, i] = (f_plus - f_minus) / (2 * eps)
        return F

    def predict(self, state: np.ndarray, P: np.ndarray, dt: float):
        """EKF predict step.
            state' = f(state)                (nonlinear propagation of mean)
            P'     = F P F^T + Q             (covariance propagation)
        """
        F = self._numerical_jacobian(state, dt)
        new_state = self.state_transition(state, dt)
        Q = self.process_noise(dt)
        new_P = F @ P @ F.T + Q
        return new_state, new_P

    def update(self, state: np.ndarray, P: np.ndarray, z: np.ndarray):
        """EKF update (correction) step given a position measurement z=[x,y].
            y = z - H x                      (innovation)
            S = H P H^T + R                  (innovation covariance)
            K = P H^T S^-1                   (Kalman gain)
            x' = x + K y
            P' = (I - K H) P
        """
        H = self.H
        y = z - H @ state
        S = H @ P @ H.T + self.R
        K = P @ H.T @ np.linalg.inv(S)
        new_state = state + K @ y
        new_P = (np.eye(self.state_dim) - K @ H) @ P
        return new_state, new_P

    def predict_from_history(self, history_xy: np.ndarray, dt: float, future_len: int):
        """Run predict+update over the history window to obtain a filtered
        current state, then roll forward `future_len` pure-predict steps.

        history_xy: [T_hist, 2]
        returns: mean_traj [future_len, 2], cov_trace [future_len]
        """
        kin = _finite_diff_kinematics(history_xy, dt)
        state = self.init_state(kin)
        P = np.eye(self.state_dim) * 1.0  # initial uncertainty

        # --- filtering pass over history (predict + update at each obs) ---
        # Start from the 3rd point since finite differences need 2 prior
        # points; use points [2:] as sequential measurements.
        for t in range(2, history_xy.shape[0]):
            state, P = self.predict(state, P, dt)
            z = history_xy[t]
            state, P = self.update(state, P, z)

        # --- forward rollout (predict only, no measurements) ---
        mean_traj = np.zeros((future_len, 2))
        cov_trace = np.zeros(future_len)
        for t in range(future_len):
            state, P = self.predict(state, P, dt)
            mean_traj[t] = state[:2]
            pos_cov = P[:2, :2]
            cov_trace[t] = float(np.trace(pos_cov))

        return mean_traj, cov_trace


class ConstantVelocityEKF(_BaseEKF):
    """CV model. State = [x, y, vx, vy] (4,). Linear motion model."""

    state_dim = 4

    def init_state(self, kin):
        return np.array([kin["x"], kin["y"], kin["vx"], kin["vy"]])

    def state_transition(self, state, dt):
        x, y, vx, vy = state
        return np.array([x + vx * dt, y + vy * dt, vx, vy])

    def process_noise(self, dt):
        s = self.process_noise_scale
        q_pos, q_vel = s["pos"], s["vel"]
        return np.diag([q_pos, q_pos, q_vel, q_vel]) * dt


class ConstantAccelerationEKF(_BaseEKF):
    """CA model. State = [x, y, vx, vy, ax, ay] (6,). Linear motion model."""

    state_dim = 6

    def init_state(self, kin):
        # split scalar accel `a` (along heading) into x/y components
        ax = kin["a"] * np.cos(kin["theta"])
        ay = kin["a"] * np.sin(kin["theta"])
        return np.array([kin["x"], kin["y"], kin["vx"], kin["vy"], ax, ay])

    def state_transition(self, state, dt):
        x, y, vx, vy, ax, ay = state
        return np.array([
            x + vx * dt + 0.5 * ax * dt ** 2,
            y + vy * dt + 0.5 * ay * dt ** 2,
            vx + ax * dt,
            vy + ay * dt,
            ax,
            ay,
        ])

    def process_noise(self, dt):
        s = self.process_noise_scale
        q_pos, q_vel, q_acc = s["pos"], s["vel"], s["acc"]
        return np.diag([q_pos, q_pos, q_vel, q_vel, q_acc, q_acc]) * dt


class ConstantTurnRateVelocityEKF(_BaseEKF):
    """CTRV model. State = [x, y, v, theta, omega] (5,). Nonlinear (constant
    speed, constant yaw rate) -- the standard model for turning maneuvers.
    """

    state_dim = 5

    def init_state(self, kin):
        return np.array([kin["x"], kin["y"], kin["v"], kin["theta"], kin["omega"]])

    def state_transition(self, state, dt):
        x, y, v, theta, omega = state
        if abs(omega) > 1e-4:
            new_x = x + (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
            new_y = y + (v / omega) * (-np.cos(theta + omega * dt) + np.cos(theta))
        else:
            # degenerate case: omega ~ 0 -> falls back to straight-line motion
            new_x = x + v * np.cos(theta) * dt
            new_y = y + v * np.sin(theta) * dt
        new_theta = theta + omega * dt
        return np.array([new_x, new_y, v, new_theta, omega])

    def process_noise(self, dt):
        s = self.process_noise_scale
        q_pos, q_vel, q_yaw = s["pos"], s["vel"], s["yaw"]
        return np.diag([q_pos, q_pos, q_vel, q_yaw, q_yaw]) * dt


class ConstantTurnRateAccelerationEKF(_BaseEKF):
    """CTRA model. State = [x, y, v, theta, omega, a] (6,). Nonlinear
    (constant yaw rate AND constant longitudinal acceleration) -- captures
    the combined accelerating/turning regime (e.g. accelerating out of a
    turn at an intersection).
    """

    state_dim = 6

    def init_state(self, kin):
        return np.array([kin["x"], kin["y"], kin["v"], kin["theta"], kin["omega"], kin["a"]])

    def state_transition(self, state, dt):
        x, y, v, theta, omega, a = state
        new_v = v + a * dt
        if abs(omega) > 1e-4:
            # closed-form CTRA integral (Schubert et al. 2008 style)
            new_theta = theta + omega * dt
            new_x = x + (1.0 / (omega ** 2)) * (
                (new_v * omega * np.sin(new_theta) + a * np.cos(new_theta))
                - (v * omega * np.sin(theta) + a * np.cos(theta))
            )
            new_y = y + (1.0 / (omega ** 2)) * (
                (-new_v * omega * np.cos(new_theta) + a * np.sin(new_theta))
                - (-v * omega * np.cos(theta) + a * np.sin(theta))
            )
        else:
            new_theta = theta
            new_x = x + v * np.cos(theta) * dt + 0.5 * a * np.cos(theta) * dt ** 2
            new_y = y + v * np.sin(theta) * dt + 0.5 * a * np.sin(theta) * dt ** 2
        return np.array([new_x, new_y, new_v, new_theta, omega, a])

    def process_noise(self, dt):
        s = self.process_noise_scale
        q_pos, q_vel, q_yaw, q_acc = s["pos"], s["vel"], s["yaw"], s["acc"]
        return np.diag([q_pos, q_pos, q_vel, q_yaw, q_yaw, q_acc]) * dt


def build_ekf_bank(cfg: dict):
    """Construct the 4 EKF motion models from a config dict (see config.yaml
    `ekf:` section). Returns an ordered list [CV, CA, CTRV, CTRA].
    """
    noise = dict(
        pos=cfg.get("process_noise_pos", 0.05),
        vel=cfg.get("process_noise_vel", 0.1),
        yaw=cfg.get("process_noise_yaw", 0.05),
        acc=cfg.get("process_noise_acc", 0.1),
    )
    return [
        ConstantVelocityEKF(noise),
        ConstantAccelerationEKF(noise),
        ConstantTurnRateVelocityEKF(noise),
        ConstantTurnRateAccelerationEKF(noise),
    ]


def run_ekf_bank(models, history_xy: np.ndarray, dt: float, future_len: int):
    """Run every model in `models` on the same history and stack results.

    Returns:
        candidates : np.ndarray [K, future_len, 2]
        uncertainty: np.ndarray [K, future_len]   (cov trace per step)
    """
    K = len(models)
    candidates = np.zeros((K, future_len, 2))
    uncertainty = np.zeros((K, future_len))
    for k, model in enumerate(models):
        mean_traj, cov_trace = model.predict_from_history(history_xy, dt, future_len)
        candidates[k] = mean_traj
        uncertainty[k] = cov_trace
    return candidates, uncertainty
