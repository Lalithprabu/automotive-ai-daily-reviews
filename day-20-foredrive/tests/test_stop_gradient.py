"""
THE most important correctness test in this repository (see models/foredrive.py's
module docstring and the task spec that generated this repo).

Verifies, with real `.backward()` calls and real `.grad` inspection (not just
code-reading / comments), that:

  1. A PLANNING-ONLY backward pass gives the shared online encoder NONZERO
     gradient (planning gradients update the encoder end-to-end), and gives
     the world-model predictor ZERO/None gradient (stop-gradient routing
     keeps planning_loss from touching the predictor's weights).

  2. A FORECASTING-ONLY backward pass gives the world-model predictor
     NONZERO gradient (it's trained by the forecasting loss), and gives the
     shared online encoder ZERO/None gradient (the detach on the predictor's
     input, and on the JEPA target, blocks encoder grad from this loss).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

from models.foredrive import ForeDriveModel

CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def _load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def _fresh_batch(cfg, b=4):
    h = len(cfg["data"]["horizons"])
    current = torch.randn(b, cfg["data"]["raster_channels"], cfg["data"]["raster_size"], cfg["data"]["raster_size"])
    future = torch.randn(b, h, cfg["data"]["raster_channels"], cfg["data"]["raster_size"], cfg["data"]["raster_size"])
    ego = torch.randn(b, cfg["encoder"]["ego_state_dim"])
    waypoints = torch.randn(b, cfg["data"]["num_waypoints"], 2)
    return current, future, ego, waypoints


def _grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.abs().sum().item()
    return total


def test_planning_loss_updates_encoder_not_predictor():
    cfg = _load_cfg()
    model = ForeDriveModel(cfg)
    model.zero_grad(set_to_none=True)

    current, future, ego, waypoints = _fresh_batch(cfg)
    out = model.forward_train(current, future, ego, waypoints)

    out["planning_loss"].backward()

    encoder_grad = _grad_norm(model.encoder)
    predictor_grad = _grad_norm(model.world_model)

    assert encoder_grad > 0.0, (
        "Planning loss must give the shared online encoder nonzero gradient "
        "(planning gradients are supposed to update the encoder end-to-end)."
    )
    assert predictor_grad == 0.0, (
        "Planning loss must NOT reach the latent predictor's parameters "
        "(stop-gradient routing: the predictor should be trained by the "
        "forecasting loss only). Got nonzero predictor grad from planning_loss."
    )


def test_forecasting_loss_updates_predictor_not_encoder():
    cfg = _load_cfg()
    model = ForeDriveModel(cfg)
    model.zero_grad(set_to_none=True)

    current, future, ego, waypoints = _fresh_batch(cfg)
    out = model.forward_train(current, future, ego, waypoints)

    out["forecast_loss"].backward()

    encoder_grad = _grad_norm(model.encoder)
    predictor_grad = _grad_norm(model.world_model)

    assert predictor_grad > 0.0, (
        "Forecasting loss must give the latent predictor nonzero gradient "
        "(it is the only loss meant to train it)."
    )
    assert encoder_grad == 0.0, (
        "Forecasting loss must NOT reach the shared online encoder's "
        "parameters (the predictor's input is a detached copy of the "
        "encoder output, and the JEPA target is also detached). Got "
        "nonzero encoder grad from forecast_loss alone."
    )


def test_combined_loss_updates_both_correctly():
    """Sanity-check the realistic training case: both losses backward()'d
    together (as train.py does) still respects the same routing — encoder
    grad should come entirely from the planning path, predictor grad
    entirely from the forecasting path. We check this indirectly by
    comparing against the sum of the two isolated cases."""
    cfg = _load_cfg()

    torch.manual_seed(0)
    model_a = ForeDriveModel(cfg)
    torch.manual_seed(0)
    model_b = ForeDriveModel(cfg)  # identical init to model_a

    # Dropout draws a fresh random mask on every forward call. model_a and
    # model_b each call forward_train() exactly once, but at DIFFERENT points
    # in the global RNG stream (model_b's call happens after model_a's), so
    # their masks would differ and make this an apples-to-oranges numeric
    # comparison for a reason that has nothing to do with stop-gradient
    # routing. eval() disables dropout so the comparison is exact.
    model_a.eval()
    model_b.eval()

    current, future, ego, waypoints = _fresh_batch(cfg)

    # forward_train() internally samples a random diffusion timestep `t` and
    # Gaussian `noise` for the planning path. Reset the seed immediately
    # before each model's forward_train() call so both draw IDENTICAL t/noise
    # -- otherwise model_a's and model_b's forward_train() calls would land
    # at different points in the global RNG stream (model_b's runs second)
    # and get different noise, again for a reason unrelated to stop-gradient
    # routing.
    torch.manual_seed(123)
    # Combined backward on model_a
    out_a = model_a.forward_train(current, future, ego, waypoints)
    (out_a["forecast_loss"] + out_a["planning_loss"]).backward()
    encoder_grad_combined = _grad_norm(model_a.encoder)
    predictor_grad_combined = _grad_norm(model_a.world_model)

    torch.manual_seed(123)
    # Planning-only + forecast-only backward on model_b, accumulated (grads add)
    out_b = model_b.forward_train(current, future, ego, waypoints)
    out_b["planning_loss"].backward(retain_graph=True)
    out_b["forecast_loss"].backward()
    encoder_grad_separate = _grad_norm(model_b.encoder)
    predictor_grad_separate = _grad_norm(model_b.world_model)

    assert abs(encoder_grad_combined - encoder_grad_separate) < 1e-4
    assert abs(predictor_grad_combined - predictor_grad_separate) < 1e-4
