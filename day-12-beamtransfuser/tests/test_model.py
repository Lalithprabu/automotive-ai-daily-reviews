"""
Tests for the BeamTransFuser reconstruction.

`test_full_modality_dropout_is_finite_no_nan` is a REQUIRED regression test:
an early draft applied a key_padding_mask from modality-presence flags
inside CrossModalFusionBlock, which fought against ModalityImputer's
synthesized tokens and produced an all-masked attention row -> NaN softmax
-> loss=nan on step 1 whenever a modality was fully absent. The fix was to
never mask by modality presence in the fusion blocks. This test drops each
modality entirely (for the WHOLE batch) in turn and asserts the model's
output and the training loss stay finite.
"""
import os
import sys

import pytest
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.beam_transfuser import (
    BeamTransFuser,
    ConvTokenEncoder,
    GPSTokenEncoder,
    ModalityImputer,
    CrossModalFusionBlock,
    build_model_from_config,
    MODALITY_ORDER,
)
from src.utils.synthetic_data import generate_batch

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(0)
    return build_model_from_config(cfg)


def test_camera_encoder_shape(cfg):
    d_model = cfg["model"]["d_model"]
    n_tok = cfg["model"]["tokens_per_modality"]["camera"]
    enc = ConvTokenEncoder(3, d_model, n_tok)
    x = torch.randn(5, 3, 32, 32)
    out = enc(x)
    assert out.shape == (5, n_tok, d_model)
    assert torch.isfinite(out).all()


def test_lidar_and_radar_encoder_shapes(cfg):
    d_model = cfg["model"]["d_model"]
    n_lidar = cfg["model"]["tokens_per_modality"]["lidar"]
    n_radar = cfg["model"]["tokens_per_modality"]["radar"]
    lidar_enc = ConvTokenEncoder(1, d_model, n_lidar)
    radar_enc = ConvTokenEncoder(1, d_model, n_radar)
    lidar_out = lidar_enc(torch.randn(6, 1, 32, 32))
    radar_out = radar_enc(torch.randn(6, 1, 16, 16))
    assert lidar_out.shape == (6, n_lidar, d_model)
    assert radar_out.shape == (6, n_radar, d_model)
    assert torch.isfinite(lidar_out).all() and torch.isfinite(radar_out).all()


def test_gps_encoder_shape(cfg):
    d_model = cfg["model"]["d_model"]
    n_tok = cfg["model"]["tokens_per_modality"]["gps"]
    enc = GPSTokenEncoder(4, d_model, n_tok)
    x = torch.randn(7, 4)
    out = enc(x)
    assert out.shape == (7, n_tok, d_model)
    assert torch.isfinite(out).all()


def test_imputer_preserves_present_tokens_and_fills_absent(cfg):
    d_model = cfg["model"]["d_model"]
    tpm = cfg["model"]["tokens_per_modality"]
    imputer = ModalityImputer(d_model, cfg["model"]["n_heads"], tpm)

    b = 4
    tokens = {m: torch.randn(b, n, d_model) for m, n in tpm.items()}
    presence = {m: torch.ones(b, dtype=torch.bool) for m in tpm}
    # drop "lidar" for samples 0 and 2 only
    presence["lidar"][0] = False
    presence["lidar"][2] = False

    out = imputer(tokens, presence)

    # shapes preserved
    for m, n in tpm.items():
        assert out[m].shape == (b, n, d_model)

    # present-modality tokens must be untouched (samples 1, 3 for lidar; all
    # samples for every other modality)
    assert torch.allclose(out["lidar"][1], tokens["lidar"][1])
    assert torch.allclose(out["lidar"][3], tokens["lidar"][3])
    for m in tpm:
        if m == "lidar":
            continue
        assert torch.allclose(out[m], tokens[m])

    # absent-modality tokens must have been REPLACED (not equal to the
    # zeroed input) and must be finite
    assert not torch.allclose(out["lidar"][0], tokens["lidar"][0])
    assert torch.isfinite(out["lidar"]).all()


def test_fusion_block_forward_shape(cfg):
    d_model = cfg["model"]["d_model"]
    block = CrossModalFusionBlock(d_model, cfg["model"]["n_heads"], cfg["model"]["ffn_hidden"])
    x = torch.randn(3, 23, d_model)
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_full_model_forward_shape_and_finite(model, cfg):
    batch = generate_batch(6, num_beams=cfg["model"]["num_beams"], modality_dropout_prob=0.15)
    logits = model(batch.camera, batch.lidar, batch.radar, batch.gps, batch.presence)
    assert logits.shape == (6, cfg["model"]["num_beams"])
    assert torch.isfinite(logits).all()


def test_gradients_flow_through_all_encoders(cfg):
    torch.manual_seed(1)
    model = build_model_from_config(cfg)
    batch = generate_batch(4, num_beams=cfg["model"]["num_beams"], modality_dropout_prob=0.2)
    logits = model(batch.camera, batch.lidar, batch.radar, batch.gps, batch.presence)
    loss = torch.nn.functional.cross_entropy(logits, batch.beam_label)
    loss.backward()
    for m in MODALITY_ORDER:
        enc = model.encoders[m]
        grads = [p.grad for p in enc.parameters() if p.grad is not None]
        assert len(grads) > 0, f"no gradients reached {m} encoder"
        assert all(torch.isfinite(g).all() for g in grads), f"non-finite grad in {m} encoder"


def test_full_modality_dropout_is_finite_no_nan(model, cfg):
    """REQUIRED regression test for the imputer/mask NaN bug.

    Drop each modality entirely (force_drop_all=True -> every sample in the
    batch is missing that modality) one at a time, run a forward + backward
    pass, and assert the loss and all gradients stay finite. This is exactly
    the scenario that produced `nan` loss on step 1 before the fix: the
    fusion blocks must NOT re-mask the imputer's synthesized tokens.
    """
    for m in MODALITY_ORDER:
        local_model = BeamTransFuser(
            d_model=cfg["model"]["d_model"],
            n_heads=cfg["model"]["n_heads"],
            n_fusion_blocks=cfg["model"]["n_fusion_blocks"],
            ffn_hidden=cfg["model"]["ffn_hidden"],
            dropout=cfg["model"]["dropout"],
            num_beams=cfg["model"]["num_beams"],
            tokens_per_modality=cfg["model"]["tokens_per_modality"],
            camera_channels=cfg["data"]["camera_shape"][0],
            lidar_channels=cfg["data"]["lidar_shape"][0],
            radar_channels=cfg["data"]["radar_shape"][0],
            gps_dim=cfg["data"]["gps_dim"],
        )
        batch = generate_batch(
            16, num_beams=cfg["model"]["num_beams"], force_drop=m, force_drop_all=True,
        )
        assert bool((~batch.presence[m]).all()), f"expected {m} fully absent for whole batch"

        logits = local_model(batch.camera, batch.lidar, batch.radar, batch.gps, batch.presence)
        assert torch.isfinite(logits).all(), f"non-finite logits when '{m}' fully dropped"

        loss = torch.nn.functional.cross_entropy(logits, batch.beam_label)
        assert torch.isfinite(loss).all(), f"loss is not finite when '{m}' fully dropped: {loss.item()}"

        local_model.zero_grad()
        loss.backward()
        for name, p in local_model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"non-finite grad in '{name}' when '{m}' dropped"
