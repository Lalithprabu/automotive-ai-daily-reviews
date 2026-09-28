import torch

from src.models.anchor_conditioned_predictor import AnchorConditionedPredictor

H, T, STATE_DIM, COND_DIM, FUTURE_DIM, HIDDEN = 8, 10, 4, 2, 2, 16


def make_model():
    return AnchorConditionedPredictor(state_dim=STATE_DIM, cond_dim=COND_DIM,
                                       hidden_dim=HIDDEN, future_len=T, future_dim=FUTURE_DIM)


def make_hists(batch=4):
    torch.manual_seed(0)
    ego_hist = torch.randn(batch, H, STATE_DIM)
    other_hist = torch.randn(batch, H, STATE_DIM)
    return ego_hist, other_hist


def test_output_shape():
    model = make_model()
    ego_hist, other_hist = make_hists(batch=5)
    cond = torch.randn(5, COND_DIM)
    out = model(ego_hist, other_hist, cond)
    assert out.shape == (5, T, FUTURE_DIM)


def test_anchor_conditioning_changes_output():
    """Critical property: the SAME history under DIFFERENT anchors/intents must produce
    genuinely different predictions. If this failed, anchor conditioning would be a no-op
    and querying once-per-anchor vs. once-globally would be pointless."""
    model = make_model()
    model.eval()
    ego_hist, other_hist = make_hists(batch=1)

    cond_a = torch.tensor([[1.0, 0.8]])     # aggressive_merge-like
    cond_b = torch.tensor([[-1.0, -1.0]])   # decelerate_follow-like

    with torch.no_grad():
        out_a = model(ego_hist, other_hist, cond_a)
        out_b = model(ego_hist, other_hist, cond_b)

    diff = torch.norm(out_a - out_b).item()
    assert diff > 1e-3, f"predictor output barely changed across very different anchors (diff={diff})"


def test_conditioning_is_continuous_not_constant():
    """Sweeping cond smoothly should move the prediction, not just flip between two fixed outputs."""
    model = make_model()
    model.eval()
    ego_hist, other_hist = make_hists(batch=1)

    outs = []
    with torch.no_grad():
        for alpha in torch.linspace(-1.0, 1.0, 5):
            cond = torch.tensor([[alpha.item(), alpha.item()]])
            outs.append(model(ego_hist, other_hist, cond))
    # consecutive outputs along the sweep should all differ from each other
    for i in range(len(outs) - 1):
        assert torch.norm(outs[i] - outs[i + 1]).item() > 1e-4


def test_gradient_flows_through_all_branches():
    model = make_model()
    ego_hist, other_hist = make_hists(batch=3)
    cond = torch.randn(3, COND_DIM, requires_grad=False)
    target = torch.randn(3, T, FUTURE_DIM)

    pred = model(ego_hist, other_hist, cond)
    loss = torch.nn.functional.mse_loss(pred, target)
    loss.backward()

    # every parameter should have received a real (nonzero somewhere) gradient
    n_params_with_grad = 0
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached parameter {name}"
        if p.grad.abs().sum().item() > 0:
            n_params_with_grad += 1
    assert n_params_with_grad == len(list(model.parameters()))
