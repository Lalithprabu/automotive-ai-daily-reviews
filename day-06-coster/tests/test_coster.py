"""Shape / composition-math / gradient-flow smoke tests for COSTER."""
import torch

from src.models.coster import (
    COSTERConfig, PolylineMapEncoder, AgentHistoryEncoder, SceneContextEncoder,
    CollisionSnapshotHead, compose_collision_state, COSTERModel, coster_loss,
)


def _cfg():
    return COSTERConfig(map_point_dim=6, agent_state_dim=4, hidden_dim=16, latent_dim=4,
                         state_dim=4, n_history_steps=8, t_rev=6, n_collision_time_bins=5, n_heads=2)


def test_map_encoder_shape():
    cfg = _cfg()
    enc = PolylineMapEncoder(cfg)
    polylines = torch.randn(3, 7, 5, cfg.map_point_dim)
    point_mask = torch.ones(3, 7, 5, dtype=torch.bool)
    out = enc(polylines, point_mask)
    assert out.shape == (3, 7, cfg.hidden_dim)


def test_agent_history_encoder_shape():
    cfg = _cfg()
    enc = AgentHistoryEncoder(cfg)
    hist = torch.randn(3, 4, cfg.n_history_steps, cfg.agent_state_dim)
    out = enc(hist)
    assert out.shape == (3, 4, cfg.hidden_dim)


def test_scene_context_fusion_shape():
    cfg = _cfg()
    scene = SceneContextEncoder(cfg)
    target = torch.randn(3, cfg.hidden_dim)
    agents = torch.randn(3, 4, cfg.hidden_dim)
    maps = torch.randn(3, 7, cfg.hidden_dim)
    out = scene(target, agents, maps)
    assert out.shape == (3, cfg.hidden_dim)


def test_compose_collision_state_math():
    cfg = _cfg()
    b, n_bins = 2, cfg.n_collision_time_bins
    target_future_states = torch.zeros(b, n_bins, cfg.state_dim)
    # bin 2 is a known state
    target_future_states[:, 2] = torch.tensor([1.0, 2.0, 0.0, 5.0])
    logits = torch.full((b, n_bins), -100.0)
    logits[:, 2] = 100.0  # force hard-argmax onto bin 2
    contact_offset = torch.tensor([[0.5, -0.5, 0.1]] * b)

    state = compose_collision_state(target_future_states, logits, contact_offset, hard=True)
    assert torch.allclose(state[:, 0], torch.tensor([1.5, 1.5]), atol=1e-5)
    assert torch.allclose(state[:, 1], torch.tensor([1.5, 1.5]), atol=1e-5)


def test_time_reversed_decoder_anchors_t0():
    cfg = _cfg()
    model = COSTERModel(cfg)
    collision_state = torch.randn(2, cfg.state_dim)
    z = torch.randn(2, cfg.latent_dim)
    scene_context = torch.randn(2, cfg.hidden_dim)
    rollout = model.decoder(collision_state, z, scene_context)
    assert rollout.shape == (2, cfg.t_rev, cfg.state_dim)
    assert torch.allclose(rollout[:, 0], collision_state)  # t=0 must equal the collision instant


def test_full_model_forward_train_and_inference_mode():
    cfg = _cfg()
    model = COSTERModel(cfg)
    B, N, M, P = 3, 4, 7, 5

    polylines = torch.randn(B, M, P, cfg.map_point_dim)
    point_mask = torch.ones(B, M, P, dtype=torch.bool)
    agent_histories = torch.randn(B, N, cfg.n_history_steps, cfg.agent_state_dim)
    target_idx = torch.zeros(B, dtype=torch.long)
    target_future_states = torch.randn(B, cfg.n_collision_time_bins, cfg.state_dim)
    gt_lead_up_traj = torch.randn(B, cfg.t_rev, cfg.state_dim)

    model.train()
    out_train = model(polylines, point_mask, agent_histories, target_idx,
                       target_future_states, gt_lead_up_traj=gt_lead_up_traj)
    assert out_train["rollout"].shape == (B, cfg.t_rev, cfg.state_dim)
    assert out_train["posterior_mu"] is not None

    model.eval()
    with torch.no_grad():
        out_infer = model(polylines, point_mask, agent_histories, target_idx, target_future_states)
    assert out_infer["rollout"].shape == (B, cfg.t_rev, cfg.state_dim)
    assert out_infer["posterior_mu"] is None


def test_backward_pass_and_optimizer_step_changes_params():
    cfg = _cfg()
    model = COSTERModel(cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    B, N, M, P = 2, 3, 4, 5

    polylines = torch.randn(B, M, P, cfg.map_point_dim)
    point_mask = torch.ones(B, M, P, dtype=torch.bool)
    agent_histories = torch.randn(B, N, cfg.n_history_steps, cfg.agent_state_dim)
    target_idx = torch.zeros(B, dtype=torch.long)
    target_future_states = torch.randn(B, cfg.n_collision_time_bins, cfg.state_dim)
    gt_lead_up_traj = torch.randn(B, cfg.t_rev, cfg.state_dim)
    gt_collision_time_bin = torch.randint(0, cfg.n_collision_time_bins, (B,))
    gt_contact_pose_offset = torch.randn(B, 3)

    before = model.decoder.to_delta[0].weight.clone()

    model.train()
    outputs = model(polylines, point_mask, agent_histories, target_idx,
                     target_future_states, gt_lead_up_traj=gt_lead_up_traj)
    loss, logs = coster_loss(outputs, gt_lead_up_traj, gt_collision_time_bin, gt_contact_pose_offset)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = model.decoder.to_delta[0].weight
    assert not torch.allclose(before, after)
    assert torch.isfinite(torch.tensor(list(logs.values()))).all()


def test_param_count_sanity():
    cfg = _cfg()
    model = COSTERModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
