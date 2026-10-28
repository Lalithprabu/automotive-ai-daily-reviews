"""Synthetic off-road world + ground-truth vehicle dynamics (my own design, NOT the paper's data).

Terrain BEV channels (C=3), stored as one tensor [N, 3, S, S], row = y, col = x:
  0: height (m)        1: friction mu in [0.25, 1]       2: roughness in [0, 1]
The map covers [-E, E]^2 metres with E = S * RES / 2.

Ground-truth dynamics are deliberately *spatial*: traction is evaluated at the four wheel
contact patches, so a left/right friction difference produces a yaw moment, and slope /
roughness / mud change speed and sideways slip. A model that only sees the single cell
under the vehicle cannot recover left-vs-right structure; a model that looks at a local BEV
window can.  This is the property SpatialBEV argues for - built in by construction here.
"""
import math
import torch
import torch.nn.functional as F

S, RES = 128, 0.5            # 128 x 128 cells, 0.5 m / cell -> 64 m x 64 m
E = S * RES / 2.0            # half extent (m)
DT = 0.1                     # s
WHEELBASE, TRACK = 2.6, 1.6  # m
G = 9.81


def _smooth_noise(n: int, sigma_cells: float, gen: torch.Generator) -> torch.Tensor:
    """Unit-variance Gaussian random field [n, S, S] via FFT low-pass filtering."""
    white = torch.randn(n, S, S, generator=gen)
    f = torch.fft.rfft2(white)
    ky = torch.fft.fftfreq(S).view(S, 1) * 2 * math.pi
    kx = torch.fft.rfftfreq(S).view(1, -1) * 2 * math.pi
    filt = torch.exp(-0.5 * (kx ** 2 + ky ** 2) * sigma_cells ** 2)
    out = torch.fft.irfft2(f * filt, s=(S, S))
    return (out - out.mean((1, 2), keepdim=True)) / (out.std((1, 2), keepdim=True) + 1e-6)


def make_terrain(n_maps: int, seed: int) -> torch.Tensor:
    """-> [n_maps, 3, S, S] float32 (height m, friction, roughness)."""
    gen = torch.Generator().manual_seed(seed)
    height = 1.8 * _smooth_noise(n_maps, 14.0, gen)                      # rolling hills, ~10 deg max
    mu = 0.25 + 0.75 * torch.sigmoid(2.5 * (_smooth_noise(n_maps, 9.0, gen) + 0.25))  # mud patches
    rough = torch.sigmoid(1.5 * _smooth_noise(n_maps, 5.0, gen))
    return torch.stack([height, mu, rough], 1).float()


def sample_map(terrain: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Bilinear sample. terrain [B,C,S,S]; xy [B,K,2] world metres -> [B,C,K]."""
    grid = (xy / E).clamp(-1, 1).unsqueeze(1)                              # [B,1,K,2], (x, y) order
    out = F.grid_sample(terrain, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out.squeeze(2)


def ego_to_world(state: torch.Tensor, off_f: torch.Tensor, off_l: torch.Tensor) -> torch.Tensor:
    """state [B,4] (x,y,yaw,v); offsets (forward, left) in metres, any shape [K] -> [B,K,2]."""
    x, y, yaw = state[:, 0:1], state[:, 1:2], state[:, 2:3]
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([x + c * off_f - s * off_l, y + s * off_f + c * off_l], -1)


_WHEELS_F = torch.tensor([1.3, 1.3, -1.3, -1.3])
_WHEELS_L = torch.tensor([TRACK / 2, -TRACK / 2, TRACK / 2, -TRACK / 2])   # FL, FR, RL, RR


def truth_step(state: torch.Tensor, act: torch.Tensor, terrain: torch.Tensor) -> torch.Tensor:
    """Ground-truth one-step dynamics. state [B,4]=(x,y,yaw,v); act [B,2]=(steer rad, v_cmd m/s)."""
    x, y, yaw, v = state.unbind(-1)
    steer, v_cmd = act.unbind(-1)
    wheels = sample_map(terrain, ego_to_world(state, _WHEELS_F, _WHEELS_L))      # [B,3,4]
    mu_w = wheels[:, 1]                                                           # [B,4]
    mu_left, mu_right = mu_w[:, [0, 2]].mean(1), mu_w[:, [1, 3]].mean(1)
    mu = mu_w.mean(1)
    # slope from the height at the four contact patches
    h = wheels[:, 0]
    slope_f = (h[:, :2].mean(1) - h[:, 2:].mean(1)) / (2 * 1.3)                   # rise per metre, forward
    slope_l = (h[:, [0, 2]].mean(1) - h[:, [1, 3]].mean(1)) / TRACK               # rise per metre, to the left
    rough = sample_map(terrain, state[:, None, :2])[:, 2, 0]

    acc = 2.0 * mu * (v_cmd - v) - G * slope_f - 0.25 * rough * v
    v_new = (v + DT * acc).clamp(0.5, 14.0)
    tan_d = torch.tan(steer)
    yaw_rate = v * tan_d / WHEELBASE * (0.35 + 0.65 * mu) + 0.45 * (mu_left - mu_right) * v.clamp(max=10) / 10
    v_lat = (1 - mu) * v * tan_d * 0.8 - 3.0 * slope_l * (1 - mu)                 # sideways skid
    c, s = torch.cos(yaw), torch.sin(yaw)
    x_new = x + DT * (v * c - v_lat * s)
    y_new = y + DT * (v * s + v_lat * c)
    return torch.stack([x_new, y_new, yaw + DT * yaw_rate, v_new], -1)


def physics_delta(state: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """Terrain-agnostic kinematic-bicycle baseline, as a delta in the ego frame
    (d_forward, d_left, d_yaw, d_v)."""
    v = state[:, 3]
    steer, v_cmd = act.unbind(-1)
    zeros = torch.zeros_like(v)
    return torch.stack([DT * v, zeros, DT * v * torch.tan(steer) / WHEELBASE, DT * 1.6 * (v_cmd - v)], -1)


def apply_delta(state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Integrate an ego-frame delta (using the *old* yaw, like truth_step)."""
    x, y, yaw, v = state.unbind(-1)
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([x + c * delta[:, 0] - s * delta[:, 1],
                        y + s * delta[:, 0] + c * delta[:, 1],
                        yaw + delta[:, 2],
                        (v + delta[:, 3]).clamp(0.5, 14.0)], -1)


def sample_episodes(terrain_bank: torch.Tensor, n: int, hist: int, horizon: int, gen: torch.Generator):
    """Random driving episodes through the truth simulator.
    Returns states [n, hist+horizon+1, 4], actions [n, hist+horizon, 2], map_idx [n]."""
    T = hist + horizon
    idx = torch.randint(0, terrain_bank.shape[0], (n,), generator=gen)
    terr = terrain_bank[idx]
    xy = (torch.rand(n, 2, generator=gen) * 2 - 1) * 8.0
    yaw = (torch.rand(n, 1, generator=gen) * 2 - 1) * math.pi
    v0 = 3 + 7 * torch.rand(n, 1, generator=gen)
    state = torch.cat([xy, yaw, v0], 1)
    t = torch.arange(T).float() * DT
    a1, a2 = 0.30 * torch.rand(n, 1, generator=gen), 0.15 * torch.rand(n, 1, generator=gen)
    f1, f2 = 0.1 + 0.4 * torch.rand(n, 1, generator=gen), 0.3 + 0.7 * torch.rand(n, 1, generator=gen)
    p1, p2 = 6.28 * torch.rand(n, 1, generator=gen), 6.28 * torch.rand(n, 1, generator=gen)
    steer = a1 * torch.sin(6.28 * f1 * t + p1) + a2 * torch.sin(6.28 * f2 * t + p2)
    vc = (3 + 8 * torch.rand(n, 1, generator=gen)) + 2 * (torch.rand(n, 1, generator=gen) - 0.5) * t
    acts = torch.stack([steer, vc.clamp(2, 12)], -1)
    states = [state]
    for k in range(T):
        state = truth_step(state, acts[:, k], terr)
        states.append(state)
    return torch.stack(states, 1), acts, idx
