"""SpatialBEV-style learned dynamics model and ablations.

Idea (from the paper's abstract): at EVERY rollout step, query BEV features in a window around the
*predicted* state, so each next-state prediction sees nearby geometric / semantic cues, instead of
conditioning on perception once or through a single point / global summary.

My implementation (the paper's architecture details were not readable; this is a reconstruction):
    s_t = (x, y, yaw, v), a_t = (steer, v_cmd)
    patch   = grid_sample(BEV, ego-aligned PxP window centred on (x, y, yaw))     [B, 3, P, P]
    z_terr  = CNN(patch) -> flatten (spatial)  |  global-pool (pooled)  |  1-cell lookup (point)
    delta   = physics_delta(s_t, a_t) + residual_MLP([z_terr, state/action features])
    s_t+1   = integrate(s_t, delta)                  (differentiable, so BPTT trains the rollout)
"""
import torch
import torch.nn as nn

from .terrain import E, apply_delta, ego_to_world, physics_delta, sample_map

MODES = ("blind", "point", "pooled", "spatial")
P = 16                                  # window cells per side
WIN_F = (-3.0, 9.0)                     # forward extent (m): a bit behind, mostly look-ahead
WIN_L = (-6.0, 6.0)                     # lateral extent (m)
RES_SCALE = torch.tensor([0.1, 0.05, 0.05, 0.2])   # residual scale (ego-frame delta units)


def _window_offsets():
    f = torch.linspace(*WIN_F, P)
    l = torch.linspace(WIN_L[1], WIN_L[0], P)                    # row 0 = left, last row = right
    ll, ff = torch.meshgrid(l, f, indexing="ij")                 # [P(l), P(f)]
    return ff.reshape(-1), ll.reshape(-1)


class SpatialBEVDynamics(nn.Module):
    def __init__(self, mode: str = "spatial", hidden: int = 128):
        super().__init__()
        assert mode in MODES, mode
        self.mode = mode
        off_f, off_l = _window_offsets()
        self.register_buffer("off_f", off_f)
        self.register_buffer("off_l", off_l)
        self.register_buffer("res_scale", RES_SCALE.clone())
        if mode in ("pooled", "spatial"):
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),      # 8 x 8
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU())      # 4 x 4
            terr_dim = 32 * 4 * 4 if mode == "spatial" else 64
            self.terr_head = nn.Sequential(nn.Linear(terr_dim, hidden), nn.ReLU())
        elif mode == "point":
            self.terr_head = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
        self.state_head = nn.Sequential(nn.Linear(5, 32), nn.ReLU())
        in_dim = 32 + (0 if mode == "blind" else (32 if mode == "point" else hidden))
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 4))
        nn.init.zeros_(self.mlp[-1].weight)       # start exactly at the physics baseline
        nn.init.zeros_(self.mlp[-1].bias)

    # --- terrain query -------------------------------------------------------------------------
    def window(self, state: torch.Tensor, terrain: torch.Tensor) -> torch.Tensor:
        """Ego-aligned BEV window around `state`: [B, 3, P, P] (rows = lateral, cols = forward),
        normalised, height made relative to the patch mean."""
        pts = ego_to_world(state, self.off_f, self.off_l)                    # [B, P*P, 2]
        w = sample_map(terrain, pts).view(-1, 3, P, P)
        h = w[:, 0:1] - w[:, 0:1].mean((2, 3), keepdim=True)
        return torch.cat([h / 0.5, (w[:, 1:2] - 0.7) / 0.25, w[:, 2:3] - 0.5], 1)

    def terrain_feature(self, state, terrain):
        if self.mode == "blind":
            return None
        if self.mode == "point":
            c = sample_map(terrain, state[:, None, :2])[:, :, 0]             # [B, 3] raw cell values
            c = torch.stack([torch.zeros_like(c[:, 0]), (c[:, 1] - 0.7) / 0.25, c[:, 2] - 0.5], 1)
            return self.terr_head(c)
        f = self.cnn(self.window(state, terrain))                            # [B, 32, 4, 4]
        if self.mode == "pooled":
            f = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)               # structure-free summary
        else:
            f = f.flatten(1)                                                 # keeps WHERE things are
        return self.terr_head(f)

    # --- dynamics ------------------------------------------------------------------------------
    def step(self, state, act, terrain):
        v, (steer, vc) = state[:, 3], act.unbind(-1)
        sa = torch.stack([v / 10, steer / 0.4, vc / 10, (vc - v) / 5, torch.sin(steer)], 1)
        feats = [self.state_head(sa)]
        zt = self.terrain_feature(state, terrain)
        if zt is not None:
            feats.append(zt)
        resid = self.mlp(torch.cat(feats, 1)) * self.res_scale
        return apply_delta(state, physics_delta(state, act) + resid)

    def rollout(self, state0, actions, terrain):
        """state0 [B,4], actions [B,H,2] -> predicted states [B,H,4] (autoregressive; the BEV window
        is re-queried at each *predicted* state)."""
        out, s = [], state0
        for k in range(actions.shape[1]):
            s = self.step(s, actions[:, k], terrain)
            out.append(s)
        return torch.stack(out, 1)


@torch.no_grad()
def physics_rollout(state0, actions):
    out, s = [], state0
    for k in range(actions.shape[1]):
        s = apply_delta(s, physics_delta(s, actions[:, k]))
        out.append(s)
    return torch.stack(out, 1)
