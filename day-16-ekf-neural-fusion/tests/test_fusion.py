"""Real (non-smoke) tests for the Day 16 EKF + neural late fusion repo."""

import numpy as np
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ekf_motion_models import (
    ConstantVelocityEKF, ConstantAccelerationEKF,
    ConstantTurnRateVelocityEKF, ConstantTurnRateAccelerationEKF,
    build_ekf_bank, run_ekf_bank,
)
from src.fusion_network import LateFusionNetwork
from src.neural_predictor import NeuralTrajectoryPredictor
from src.utils import ade, fde, set_seed

HISTORY_LEN = 8
FUTURE_LEN = 12
DT = 0.5


def _straight_history():
    """A simple straight-line history: moving at constant velocity (1, 0.5) m/s."""
    t = np.arange(HISTORY_LEN) * DT
    x = 1.0 * t
    y = 0.5 * t
    return np.stack([x, y], axis=1)


def _turning_history():
    """A history with nonzero constant yaw rate, to exercise CTRV/CTRA."""
    omega = 0.3
    v = 5.0
    theta = 0.0
    x, y = 0.0, 0.0
    pts = []
    for _ in range(HISTORY_LEN):
        pts.append([x, y])
        new_theta = theta + omega * DT
        dx = (v / omega) * (np.sin(new_theta) - np.sin(theta))
        dy = (v / omega) * (-np.cos(new_theta) + np.cos(theta))
        x += dx
        y += dy
        theta = new_theta
    return np.array(pts)


# ---------------------------------------------------------------------------
# 1. EKF motion models: finite, shape-correct output
# ---------------------------------------------------------------------------

class TestEKFModels:
    def test_cv_output_shape_and_finite(self):
        model = ConstantVelocityEKF({"pos": 0.05, "vel": 0.1, "yaw": 0.05, "acc": 0.1})
        mean_traj, cov_trace = model.predict_from_history(_straight_history(), DT, FUTURE_LEN)
        assert mean_traj.shape == (FUTURE_LEN, 2)
        assert cov_trace.shape == (FUTURE_LEN,)
        assert np.all(np.isfinite(mean_traj))
        assert np.all(np.isfinite(cov_trace))
        assert np.all(cov_trace > 0), "covariance trace should be strictly positive"

    def test_ca_output_shape_and_finite(self):
        model = ConstantAccelerationEKF({"pos": 0.05, "vel": 0.1, "yaw": 0.05, "acc": 0.1})
        mean_traj, cov_trace = model.predict_from_history(_straight_history(), DT, FUTURE_LEN)
        assert mean_traj.shape == (FUTURE_LEN, 2)
        assert np.all(np.isfinite(mean_traj))
        assert np.all(np.isfinite(cov_trace))

    def test_ctrv_output_shape_and_finite_on_turn(self):
        model = ConstantTurnRateVelocityEKF({"pos": 0.05, "vel": 0.1, "yaw": 0.05, "acc": 0.1})
        mean_traj, cov_trace = model.predict_from_history(_turning_history(), DT, FUTURE_LEN)
        assert mean_traj.shape == (FUTURE_LEN, 2)
        assert np.all(np.isfinite(mean_traj))
        assert np.all(np.isfinite(cov_trace))

    def test_ctra_output_shape_and_finite_on_turn(self):
        model = ConstantTurnRateAccelerationEKF({"pos": 0.05, "vel": 0.1, "yaw": 0.05, "acc": 0.1})
        mean_traj, cov_trace = model.predict_from_history(_turning_history(), DT, FUTURE_LEN)
        assert mean_traj.shape == (FUTURE_LEN, 2)
        assert np.all(np.isfinite(mean_traj))
        assert np.all(np.isfinite(cov_trace))

    def test_covariance_grows_over_horizon(self):
        """Uncertainty should generally grow further into the future (no
        measurement updates during the pure-predict rollout)."""
        model = ConstantVelocityEKF({"pos": 0.05, "vel": 0.1, "yaw": 0.05, "acc": 0.1})
        _, cov_trace = model.predict_from_history(_straight_history(), DT, FUTURE_LEN)
        assert cov_trace[-1] > cov_trace[0]

    def test_cv_matches_straight_line_reasonably(self):
        """On a clean constant-velocity history, the CV filter's prediction
        should closely track the true straight-line continuation."""
        model = ConstantVelocityEKF({"pos": 0.01, "vel": 0.01, "yaw": 0.01, "acc": 0.01})
        history = _straight_history()
        mean_traj, _ = model.predict_from_history(history, DT, FUTURE_LEN)
        t_future = (HISTORY_LEN + np.arange(FUTURE_LEN)) * DT
        expected = np.stack([1.0 * t_future, 0.5 * t_future], axis=1)
        err = np.linalg.norm(mean_traj - expected, axis=-1).mean()
        assert err < 0.5, f"CV prediction should track a clean straight line, got mean err={err}"

    def test_ekf_bank_shapes(self):
        models = build_ekf_bank({"process_noise_pos": 0.05, "process_noise_vel": 0.1,
                                  "process_noise_yaw": 0.05, "process_noise_acc": 0.1})
        assert len(models) == 4
        cand, unc = run_ekf_bank(models, _turning_history(), DT, FUTURE_LEN)
        assert cand.shape == (4, FUTURE_LEN, 2)
        assert unc.shape == (4, FUTURE_LEN)
        assert np.all(np.isfinite(cand))
        assert np.all(np.isfinite(unc))


# ---------------------------------------------------------------------------
# 2 & 3. Fusion network: forward shapes + attention weights sum to 1
# ---------------------------------------------------------------------------

class TestFusionNetwork:
    def _make_inputs(self, B=4):
        neural_traj = torch.randn(B, FUTURE_LEN, 2)
        ekf_candidates = torch.randn(B, 4, FUTURE_LEN, 2)
        ekf_uncertainty = torch.rand(B, 4, FUTURE_LEN)
        return neural_traj, ekf_candidates, ekf_uncertainty

    def test_forward_shapes(self):
        fusion = LateFusionNetwork(future_len=FUTURE_LEN, num_ekf=4, embed_dim=16, attn_hidden=16, residual_hidden=16)
        neural_traj, ekf_candidates, ekf_uncertainty = self._make_inputs(B=5)
        fused, weights = fusion(neural_traj, ekf_candidates, ekf_uncertainty)
        assert fused.shape == (5, FUTURE_LEN, 2)
        assert weights.shape == (5, 5, FUTURE_LEN)  # 5 candidates (1 neural + 4 EKF)
        assert torch.all(torch.isfinite(fused))

    def test_attention_weights_sum_to_one_per_timestep(self):
        fusion = LateFusionNetwork(future_len=FUTURE_LEN, num_ekf=4, embed_dim=16, attn_hidden=16, residual_hidden=16)
        neural_traj, ekf_candidates, ekf_uncertainty = self._make_inputs(B=3)
        _, weights = fusion(neural_traj, ekf_candidates, ekf_uncertainty)
        sums = weights.sum(dim=1)  # sum over candidate axis -> [B, T]
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_weights_are_nonnegative(self):
        fusion = LateFusionNetwork(future_len=FUTURE_LEN, num_ekf=4, embed_dim=16, attn_hidden=16, residual_hidden=16)
        neural_traj, ekf_candidates, ekf_uncertainty = self._make_inputs(B=2)
        _, weights = fusion(neural_traj, ekf_candidates, ekf_uncertainty)
        assert torch.all(weights >= 0)

    def test_overfit_one_batch_loss_decreases(self):
        """Sanity test: the fusion network must be able to overfit a single
        small batch, with loss decreasing meaningfully -- this catches
        gradient-flow bugs (e.g. a detached tensor, or the residual head
        being disconnected from the loss)."""
        set_seed(0)
        fusion = LateFusionNetwork(future_len=FUTURE_LEN, num_ekf=4, embed_dim=16, attn_hidden=16, residual_hidden=16)
        neural_traj, ekf_candidates, ekf_uncertainty = self._make_inputs(B=8)
        target = torch.randn(8, FUTURE_LEN, 2)

        optimizer = torch.optim.Adam(fusion.parameters(), lr=5e-3)
        loss_fn = nn.SmoothL1Loss()

        losses = []
        for _ in range(150):
            optimizer.zero_grad()
            fused, _ = fusion(neural_traj, ekf_candidates, ekf_uncertainty)
            loss = loss_fn(fused, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.2, (
            f"expected substantial loss decrease when overfitting one batch, "
            f"got {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Neural predictor: basic shape/finiteness sanity (supports the fusion tests)
# ---------------------------------------------------------------------------

class TestNeuralPredictor:
    def test_forward_train_and_inference_shapes(self):
        model = NeuralTrajectoryPredictor(hidden_size=16, latent_size=8)
        history = torch.randn(4, HISTORY_LEN, 2)
        future = torch.randn(4, FUTURE_LEN, 2)

        pred_train, kl = model(history, future=future, future_len=FUTURE_LEN)
        assert pred_train.shape == (4, FUTURE_LEN, 2)
        assert kl is not None and kl.item() >= 0.0  # KL divergence is non-negative

        pred_infer, kl_none = model(history, future=None, future_len=FUTURE_LEN, sample_z=False)
        assert pred_infer.shape == (4, FUTURE_LEN, 2)
        assert kl_none is None
        assert torch.all(torch.isfinite(pred_infer))


# ---------------------------------------------------------------------------
# 5. ADE/FDE metrics: hand-computed toy example
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_ade_fde_hand_computed(self):
        # Two timesteps, single sample. pred vs gt differ by (3, 4) at t=0
        # (distance 5) and by (0, 0) at t=1 (distance 0).
        pred = np.array([[[1.0, 1.0], [5.0, 5.0]]])  # [1, 2, 2]
        gt = np.array([[[-2.0, -3.0], [5.0, 5.0]]])
        expected_ade = (5.0 + 0.0) / 2.0  # = 2.5
        expected_fde = 0.0
        assert abs(ade(pred, gt) - expected_ade) < 1e-6
        assert abs(fde(pred, gt) - expected_fde) < 1e-6

    def test_fde_uses_only_last_timestep(self):
        pred = np.array([[[0.0, 0.0], [3.0, 4.0]]])
        gt = np.array([[[100.0, 100.0], [0.0, 0.0]]])  # huge error at t=0, zero-ish setup at t=1
        # FDE should only look at the last timestep: distance between (3,4) and (0,0) = 5
        assert abs(fde(pred, gt) - 5.0) < 1e-6

    def test_ade_zero_for_identical_trajectories(self):
        traj = np.random.randn(3, 12, 2)
        assert ade(traj, traj) == 0.0
        assert fde(traj, traj) == 0.0
