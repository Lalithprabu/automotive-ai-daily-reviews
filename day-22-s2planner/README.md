# Day 22 -- S2Planner: Multi-Scale Semantic Planner for End-to-End Autonomous Driving

Reconstruction of **arXiv:2609.29813** (Lu, Zhou, Guo, Yu, Knoll -- TU Munich / Huaibei Normal University, submitted 2026-09-24), built for the Automotive AI Daily Tech Reviews series.

> **Read `SOURCING.md` first.** This repo's headline synthetic-data numbers (ADE/FDE, "59.8% reduction") are this project's own measurements on synthetic data and are **not** the paper's 88.03 PDMS NAVSIM number, which the paper's own authors flag as an exploratory, run-selected result. Every architectural detail beyond the abstract-level description in `SOURCING.md` is this project's disclosed reconstruction, not paper-sourced.

## What this is

An end-to-end trajectory planner that predicts the ego vehicle's future path directly from three synthetic front-facing "camera" feature grids, an ego-motion history, and a driving command -- no world model, no risk field, just a vision-foundation-model-style backbone feeding a geometry-guided, coarse-to-fine trajectory decoder. The core mechanism under test: **camera-projected cross-attention**, where each trajectory waypoint is projected into every camera via real pinhole geometry and used to sample that camera's features at the exact projected pixel, at every refinement stage.

## 📁 Folder structure

```
s2planner/
├── README.md                 # this file
├── SOURCING.md                # paper-sourced vs. reconstruction disclosure, selection notes, bugs found
├── requirements.txt
├── config.yaml                # model / data / training hyperparameters
├── src/
│   ├── __init__.py
│   ├── geometry.py            # pinhole camera rig + world-to-image projection (the real geometry)
│   ├── dataset.py              # synthetic multi-camera driving-scene generator
│   ├── model.py                # SpatialTuningAdapter, EgoConditionedInitializer,
│   │                           #   CameraProjectedCrossAttention, CoarseToFineDecoderLayer, S2Planner
│   └── losses.py               # coarse-to-fine trajectory loss + ADE/FDE metrics
├── tests/
│   └── test_model.py           # geometry, shape, gradient-flow, and overfit/refinement sanity tests
├── train.py                    # trains S2Planner on the synthetic dataset, saves a checkpoint
├── simulate.py                  # renders a BEV + camera + telemetry simulation GIF from a checkpoint
└── outputs/                     # generated: checkpoint, train history, simulation GIF (gitignored in spirit)
```

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/ -v                                    # 7/7 should pass
python3 train.py --config config.yaml                 # ~6-7 min on CPU, 14 epochs
python3 simulate.py --config config.yaml --n_scenes 6  # renders outputs/s2planner_simulation.gif
```

## 📝 Core architecture (PyTorch)

The full model lives in `src/model.py`; the two pieces most worth reading in isolation are the camera-projected cross-attention (the paper's named core mechanism) and the ego-conditioned, geometry-aware trajectory initializer. Both are reproduced in full below, with the tensor shapes annotated through the forward pass.

```python
class CameraProjectedCrossAttention(nn.Module):
    """
    For every trajectory point's current (x, y) estimate, project into all
    `n_cam` cameras via the pinhole rig, bilinearly sample that scale's
    per-camera feature map at the projected pixel, then run a small
    attention over the (up to n_cam) camera observations per point.
    Points that project outside every camera's view fall back to a learned
    "unobserved" embedding rather than a zero vector.
    """

    def __init__(self, d_feat: int, rig: dict):
        super().__init__()
        self.rig = rig
        self.d_feat = d_feat
        self.q_proj = nn.Linear(d_feat, d_feat)
        self.k_proj = nn.Linear(d_feat, d_feat)
        self.v_proj = nn.Linear(d_feat, d_feat)
        self.out_proj = nn.Linear(d_feat, d_feat)
        self.unobserved = nn.Parameter(torch.randn(d_feat) * 0.02)

    def forward(self, point_feat, point_xy, scale_pyramid_level):
        # point_feat: (B, T, D)              -- one feature vector per future waypoint
        # point_xy:   (B, T, 2)               -- current BEV (x, y) estimate per waypoint
        # scale_pyramid_level: (B, n_cam, D, Hs, Ws) -- this stage's per-camera feature maps
        B, T, D = point_feat.shape
        n_cam = scale_pyramid_level.shape[1]
        Hs = scale_pyramid_level.shape[-2]

        # Lift each 2D waypoint to 3D (ground plane, z=0) and project into every camera.
        xyz = torch.cat([point_xy, torch.zeros(B, T, 1, device=point_xy.device)], dim=-1)
        pixel_uv, valid = project_points(self.rig, xyz)          # (n_cam, B, T, 2), (n_cam, B, T)
        grid = normalize_pixels_for_grid_sample(pixel_uv, Hs)     # (n_cam, B, T, 2), range [-1, 1]
        in_bounds = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)
        valid = valid & in_bounds                                  # (n_cam, B, T)

        # Bilinearly sample each camera's feature map at the point's projected pixel.
        sampled = []
        for c in range(n_cam):
            g = grid[c].unsqueeze(1)                              # (B, 1, T, 2)
            feat_map = scale_pyramid_level[:, c]                   # (B, D, Hs, Ws)
            s = F.grid_sample(feat_map, g, mode="bilinear", padding_mode="zeros", align_corners=True)
            s = s.squeeze(2).permute(0, 2, 1)                       # (B, T, D)
            mask = valid[c].unsqueeze(-1)                            # (B, T, 1)
            s = torch.where(mask, s, self.unobserved.view(1, 1, -1).expand_as(s))
            sampled.append(s)
        sampled = torch.stack(sampled, dim=2)                       # (B, T, n_cam, D)
        cam_valid = valid.permute(1, 2, 0)                            # (B, T, n_cam)

        # Attention: each waypoint queries its (up to n_cam) camera observations.
        q = self.q_proj(point_feat)                                   # (B, T, D)
        k = self.k_proj(sampled)                                       # (B, T, n_cam, D)
        v = self.v_proj(sampled)                                        # (B, T, n_cam, D)

        logits = (q.unsqueeze(2) * k).sum(-1) / math.sqrt(D)             # (B, T, n_cam)
        any_valid = cam_valid.any(dim=-1, keepdim=True)
        logits = logits.masked_fill(~cam_valid & any_valid, float("-1e4"))
        attn = torch.softmax(logits, dim=-1)                              # (B, T, n_cam)
        out = (attn.unsqueeze(-1) * v).sum(2)                              # (B, T, D)
        return self.out_proj(out), attn


class EgoConditionedInitializer(nn.Module):
    """
    Ego-conditioned, geometry-aware coarse trajectory initialization: predicts
    a per-step (speed, curvature) profile from ego-motion history + driving
    command, then differentiably integrates a unicycle model to obtain an
    initial (x, y) waypoint sequence -- a real kinematic rollout, not a raw
    coordinate regression.
    """

    def __init__(self, d_hidden: int = 64, fut_len: int = FUT_LEN, dt: float = DT):
        super().__init__()
        self.fut_len = fut_len
        self.dt = dt
        in_dim = HIST_LEN * 2 + 3                                          # flattened history + 3-way command one-hot
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, fut_len * 2),                                # (speed_t, kappa_t) per future step
        )

    def forward(self, ego_history, command_onehot):
        # ego_history: (B, HIST_LEN, 2) ; command_onehot: (B, 3)
        B = ego_history.shape[0]
        inp = torch.cat([ego_history.reshape(B, -1), command_onehot], dim=-1)
        params = self.mlp(inp).view(B, self.fut_len, 2)
        speed = 2.0 + 10.0 * torch.sigmoid(params[..., 0])                    # keep speeds in [2, 12] m/s
        kappa = 0.15 * torch.tanh(params[..., 1])                              # bounded curvature, 1/m

        x = torch.zeros(B, device=ego_history.device)
        y = torch.zeros(B, device=ego_history.device)
        theta = torch.zeros(B, device=ego_history.device)
        waypoints = []
        for t in range(self.fut_len):                                            # differentiable unicycle rollout
            theta = theta + kappa[:, t] * speed[:, t] * self.dt
            x = x + speed[:, t] * self.dt * torch.cos(theta)
            y = y + speed[:, t] * self.dt * torch.sin(theta)
            waypoints.append(torch.stack([x, y], dim=-1))
        return torch.stack(waypoints, dim=1)                                      # (B, fut_len, 2)
```

The full `S2Planner` module wires these together with a 3-level `MultiScaleCameraEncoder` (backbone stem + per-scale `SpatialTuningAdapter`) and 3 `CoarseToFineDecoderLayer`s (trajectory self-attention → camera-projected cross-attention → FFN → offset head), one per pyramid level, coarse to fine. See `src/model.py` for the complete, runnable implementation.

## 🎬 Simulation

`simulate.py` loads the trained checkpoint and renders a 3-panel animation across 6 held-out synthetic scenes:

1. **BEV panel** -- lane geometry, obstacle bounding boxes, ground truth (green) vs. the ego-conditioned coarse init (dotted grey) vs. the live coarse-to-fine refinement (blue → red as it converges).
2. **Center-camera panel** -- the synthetic per-camera feature view, with the current trajectory estimate and every obstacle's bounding box projected into the image via the *same* pinhole geometry the model's cross-attention uses to sample features.
3. **Telemetry panel** -- a live bar chart of displacement error at each refinement stage (init → L1 → L2 → final), plus the driving command and a running average.

## Verified (Day 22)

`pytest tests/ -v` → **7/7 passed**, including a geometry-correctness check (a point at camera height directly ahead projects to the image center; a point to the ego's left lands at a smaller image-`u`; a point behind the camera is correctly flagged invalid), a gradient-flow check (confirms predictions genuinely depend on the sampled camera features, not just the initializer), and an overfit/refinement sanity check (after training, the final coarse-to-fine stage must beat the coarse init by a wide margin).

`train.py --config config.yaml` (CPU, 14 epochs, 640 train / 96 val synthetic scenes, ~6m34s) → 59,849 parameters; train loss 4.96 → 0.85. Held-out validation (meters, synthetic data -- **not the paper's PDMS metric**): final-stage ADE **1.136m** / FDE **2.227m**, vs. the ego-conditioned coarse-init's own ADE of 2.822m -- a **59.8% ADE reduction** attributable to the camera-projected cross-attention refinement stages.

`simulate.py` → `s2planner_simulation.gif` rendered from the real trained checkpoint (42 drawn frames; Pillow's GIF writer merges pixel-identical held frames into longer-duration ones, reporting 24 stored frames with total playback duration unchanged -- confirmed via each frame's `duration` metadata, not assumed). Visually reviewed: the center-camera panel now correctly shows lane-line and obstacle splats with projected trajectory points and bounding-box wireframes (an earlier camera-intrinsics bug rendered this panel fully black -- see `SOURCING.md` for the fix and the retrained numbers, which improved after the fix).

## A known limitation, disclosed rather than hidden

Some synthetic scenes have sharp lane curvature that sweeps obstacles and lane markers far to one side; on those scenes a given camera can see very little of the corridor, and the model falls back heavily on the "unobserved" embedding for that camera. This is a realistic failure mode for a camera-projected mechanism (severe curvature genuinely does reduce what's in view) but is more frequent here than a production system would tolerate, since this dataset was not curated to avoid the extreme end of its own curvature range.
