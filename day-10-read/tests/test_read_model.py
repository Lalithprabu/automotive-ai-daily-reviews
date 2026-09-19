"""10-test verification suite for the READ reconstruction.

Covers: positional encoding, scene encoder, risk field range/shape,
trajectory-risk integration, differentiable trajectory refinement, full
forward pass, training-loss descent, planning-cost shape, synthetic-scene
shapes, and the classical-baseline field's range.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.positional_encoding import FourierPositionalEncoding
from src.utils.synthetic_scene import (
    generate_synthetic_scene,
    rasterize_bev,
    classical_potential_field,
    sample_agent_proximal_probes,
    sample_background_probes,
    scene_to_tensors,
    GRID_SIZE,
    NUM_HISTORY_STEPS,
)
from src.models.scene_encoder import SceneEncoder
from src.models.risk_field_net import RiskFieldNetwork, refine_trajectory_by_risk_descent
from src.models.read_model import READModel, risk_ranking_loss, trajectory_planning_cost


torch.manual_seed(0)


def test_positional_encoding_shape_and_range():
    pe = FourierPositionalEncoding(num_input_dims=3, num_freqs=8, max_freq=4.0)
    coords = torch.randn(2, 5, 3)
    encoded = pe(coords)
    assert encoded.shape == (2, 5, pe.output_dim)
    assert pe.output_dim == 3 * (1 + 2 * 8)
    # sin/cos features must be bounded in [-1, 1]; raw coords are not,
    # so check just the sin/cos block (skip the first 3 raw dims).
    trig_part = encoded[..., 3:]
    assert torch.all(trig_part >= -1.0001) and torch.all(trig_part <= 1.0001)


def test_scene_encoder_output_shape():
    enc = SceneEncoder(bev_channels=3, agent_features=4, hidden_dim=64, pooled_size=4,
                        num_transformer_layers=2, num_heads=4)
    bev = torch.randn(2, 3, GRID_SIZE, GRID_SIZE)
    agent_hist = torch.randn(2, 2, NUM_HISTORY_STEPS, 4)
    tokens = enc(bev, agent_hist)
    # N = pooled_size**2 (16) + num_agents (2) = 18
    assert tokens.shape == (2, 18, 64)


def test_risk_field_output_range_and_shape():
    field = RiskFieldNetwork(hidden_dim=64, num_freqs=8, num_cross_layers=2, num_heads=4)
    scene_tokens = torch.randn(2, 18, 64)
    query = torch.randn(2, 10, 3)
    risk = field(query, scene_tokens)
    assert risk.shape == (2, 10)
    assert torch.all(risk >= 0.0) and torch.all(risk <= 1.0)


def test_trajectory_risk_integration_shape():
    model = READModel(hidden_dim=32, pooled_size=3, num_scene_layers=1, num_scene_heads=2,
                       num_freqs=4, num_cross_layers=1, num_risk_heads=2)
    bev = torch.randn(1, 3, GRID_SIZE, GRID_SIZE)
    agent_hist = torch.randn(1, 2, NUM_HISTORY_STEPS, 4)
    trajectory_xy = torch.randn(1, 8, 2)
    timestamps = torch.linspace(0, 1, 8).unsqueeze(0)
    cost = trajectory_planning_cost(model, bev, agent_hist, trajectory_xy, timestamps)
    assert cost.shape == (1,)
    assert torch.isfinite(cost).all()


def test_differentiable_trajectory_refinement_changes_trajectory():
    field = RiskFieldNetwork(hidden_dim=32, num_freqs=4, num_cross_layers=1, num_heads=2)
    scene_tokens = torch.randn(1, 6, 32)
    init_traj = torch.zeros(1, 5, 2)
    timestamps = torch.linspace(0, 1, 5).unsqueeze(0)
    refined = refine_trajectory_by_risk_descent(field, scene_tokens, init_traj, timestamps,
                                                 num_steps=5, step_size=0.05)
    assert refined.shape == init_traj.shape
    assert refined.requires_grad is False
    # Gradient descent against a (generically nonzero-gradient) random
    # field should move the trajectory away from its initialization.
    assert not torch.allclose(refined, init_traj)


def test_read_model_full_forward_pass():
    model = READModel(hidden_dim=32, pooled_size=3, num_scene_layers=1, num_scene_heads=2,
                       num_freqs=4, num_cross_layers=1, num_risk_heads=2)
    bev = torch.randn(2, 3, GRID_SIZE, GRID_SIZE)
    agent_hist = torch.randn(2, 2, NUM_HISTORY_STEPS, 4)
    query = torch.randn(2, 7, 3)
    risk = model(bev, agent_hist, query)
    assert risk.shape == (2, 7)
    assert torch.all(risk >= 0.0) and torch.all(risk <= 1.0)


def test_training_loss_descends():
    torch.manual_seed(1)
    model = READModel(hidden_dim=32, pooled_size=3, num_scene_layers=1, num_scene_heads=2,
                       num_freqs=4, num_cross_layers=1, num_risk_heads=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scene = generate_synthetic_scene(num_timesteps=10, dt=0.2, seed=0)
    bev_grid, agent_history = scene_to_tensors(scene, timestep=5)

    agent_probes = sample_agent_proximal_probes(scene, timestep=5, num_probes=8, seed=0)
    bg_probes = sample_background_probes(num_probes=16, seed=0)
    agent_probe_xyt = torch.from_numpy(agent_probes).unsqueeze(0).float()
    bg_probe_xyt = torch.from_numpy(bg_probes).unsqueeze(0).float()

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = risk_ranking_loss(model, bev_grid, agent_history, agent_probe_xyt, bg_probe_xyt)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_planning_cost_shape_batched():
    model = READModel(hidden_dim=32, pooled_size=3, num_scene_layers=1, num_scene_heads=2,
                       num_freqs=4, num_cross_layers=1, num_risk_heads=2)
    bev = torch.randn(3, 3, GRID_SIZE, GRID_SIZE)
    agent_hist = torch.randn(3, 2, NUM_HISTORY_STEPS, 4)
    trajectory_xy = torch.randn(3, 6, 2)
    timestamps = torch.linspace(0, 1, 6).unsqueeze(0).repeat(3, 1)
    cost = trajectory_planning_cost(model, bev, agent_hist, trajectory_xy, timestamps)
    assert cost.shape == (3,)


def test_synthetic_scene_shapes():
    scene = generate_synthetic_scene(num_timesteps=20, dt=0.2, seed=42)
    assert scene.lead_positions.shape == (20, 2)
    assert scene.crossing_positions.shape == (20, 2)
    assert scene.ego_reference_path.shape == (20, 2)
    assert scene.lead_history.shape == (20, NUM_HISTORY_STEPS, 4)
    assert scene.crossing_history.shape == (20, NUM_HISTORY_STEPS, 4)

    bev = rasterize_bev(scene, timestep=10)
    assert bev.shape == (3, GRID_SIZE, GRID_SIZE)

    bev_t, hist_t = scene_to_tensors(scene, timestep=10)
    assert bev_t.shape == (1, 3, GRID_SIZE, GRID_SIZE)
    assert hist_t.shape == (1, 2, NUM_HISTORY_STEPS, 4)


def test_classical_baseline_field_range_and_peak():
    agent_positions = np.array([[5.0, 0.0], [0.0, 5.0]])
    query = np.array([[5.0, 0.0], [50.0, 50.0], [0.0, 5.0]])
    risk = classical_potential_field(query, agent_positions, radius=4.0)
    assert risk.shape == (3,)
    assert np.all(risk >= 0.0) and np.all(risk <= 1.0001)
    # Exactly at an agent's position, risk should be (near) maximal.
    assert risk[0] > 0.99
    assert risk[2] > 0.99
    # Far from every agent, risk should be near zero.
    assert risk[1] < 0.01
