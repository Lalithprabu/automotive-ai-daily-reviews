import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.gam import GAMConfig, PedestrianCentricGAM


def test_build_graph_shapes():
    B, N = 4, 16
    node_positions = torch.rand(B, N, 2)
    node_classes = torch.randint(0, 4, (B, N))
    node_areas = torch.rand(B, N)
    valid_mask = torch.ones(B, N)
    valid_mask[:, 10:] = 0.0  # simulate padding past 10 real nodes

    node_features, edge_bias, adj_mask = PedestrianCentricGAM.build_graph(
        node_positions, node_classes, node_areas, valid_mask
    )
    assert node_features.shape == (B, N, 8)
    assert edge_bias.shape == (B, N, N)
    assert adj_mask.shape == (B, N, N)
    # pedestrian hub (node 0) must connect to every valid node
    assert torch.all(adj_mask[:, 0, :10] > 0)
    # padded nodes must not appear as valid edges
    assert torch.all(adj_mask[:, 0, 10:] == 0)


def test_gam_forward_and_backward():
    cfg = GAMConfig(max_nodes=16, d_model=64, num_heads=4, num_layers=2, proj_dim=40)
    model = PedestrianCentricGAM(cfg)

    B, N = 3, cfg.max_nodes
    node_positions = torch.rand(B, N, 2, requires_grad=False)
    node_classes = torch.randint(0, 4, (B, N))
    node_areas = torch.rand(B, N)
    valid_mask = torch.ones(B, N)
    valid_mask[:, 12:] = 0.0

    node_features, edge_bias, adj_mask = PedestrianCentricGAM.build_graph(
        node_positions, node_classes, node_areas, valid_mask
    )
    out = model(node_features, edge_bias, adj_mask)
    assert out.shape == (B, cfg.proj_dim)
    assert torch.isfinite(out).all()

    loss = out.pow(2).mean()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g == g for g in grad_norms)  # no NaNs
