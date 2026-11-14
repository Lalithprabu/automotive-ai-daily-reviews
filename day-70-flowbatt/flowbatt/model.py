"""
FlowBatt - Flow Matching for Probabilistic Battery Degradation Prediction (arXiv 2611.03150, NeurIPS 2026 poster)
Toy PyTorch re-implementation of the CORE idea:

    early-life capacity matrix Q(cycle, V)  --[CNN encoder, LeakyReLU]-->  c_enc
    observed SOH prefix (10 pts)            --[MLP]-------------------->  c_pre
    flow time t in [0, 1]                   --[sinusoidal + MLP]------->  c_t
                         c = c_enc + c_pre + c_t   (conditioning vector, injected by adaptive LayerNorm)

    x_t (B, 256) noisy SOH trajectory --patchify--> 16 tokens --[DiT blocks w/ adaLN-Zero]--> velocity v_theta(x_t, t, c)

Conditional flow matching (linear / rectified path, the standard choice):
    x_0 ~ N(0, I)   (noise),  x_1 = normalised true SOH trajectory,   x_t = (1 - t) x_0 + t x_1
    target velocity u = x_1 - x_0           loss = || v_theta(x_t, t, c) - u ||^2
Inference: integrate dx/dt = v_theta from t = 0 (fresh noise) to t = 1 with Euler / Heun.
Every noise draw -> one plausible future trajectory; 100 draws -> empirical 90 % interval.

Paper / repo facts followed: DiT backbone, CNN encoder of the capacity matrix -> embedding combined with the
generative-time embedding -> adaptive normalisation; LeakyReLU encoder (Grad-CAM fix, slope 0.3 in the repo);
256-point trajectories; 100x100 capacity matrix; 10-point SOH prefix; 100 samples for intervals; AdamW 1e-3.
MY choices (not stated in what I could read): depth / width / heads / patch size, Euler 32 steps, linear path,
uniform t, the DiffBatt-lite DDPM schedule, how prefix is embedded.  Not the authors' code.

Shape legend: B batch, L=256 trajectory points, P patch, N=L/P tokens, D model width, E early cycles, V volt pts.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================ conditioning
class CapacityEncoder(nn.Module):
    """(B, E, V) capacity matrix -> (B, D).

    Two input channels: Q itself and dQ = Q_n(V) - Q_1(V) (the classic early-life degradation signal),
    the latter amplified because it is ~1e-2 of Q.  LeakyReLU everywhere: the paper's Grad-CAM study found
    initialisation-sensitive dead-ReLU failures in the encoder and switched to LeakyReLU.
    """

    def __init__(self, d: int, ch: int = 32, slope: float = 0.3):
        super().__init__()
        act = lambda: nn.LeakyReLU(slope, inplace=True)                     # noqa: E731
        self.net = nn.Sequential(
            nn.Conv2d(2, ch, 5, 2, 2), nn.GroupNorm(8, ch), act(),          # (B, ch, E/2, V/2)
            nn.Conv2d(ch, ch, 3, 2, 1), nn.GroupNorm(8, ch), act(),         # (B, ch, E/4, V/4)
            nn.Conv2d(ch, 2 * ch, 3, 2, 1), nn.GroupNorm(8, 2 * ch), act(), # (B, 2ch, E/8, V/8)
            nn.Conv2d(2 * ch, 2 * ch, 3, 2, 1), nn.GroupNorm(8, 2 * ch), act(),
        )
        self.proj = nn.Sequential(nn.Linear(4 * ch, d), act(), nn.Linear(d, d))

    def forward(self, cap: torch.Tensor) -> torch.Tensor:
        dq = (cap - cap[:, :1]) * 50.0                                      # (B, E, V)
        x = torch.stack([cap, dq], 1)                                       # (B, 2, E, V)
        h = self.net(x)                                                     # (B, 2ch, h, w)
        h = torch.cat([h.mean((-1, -2)), h.amax((-1, -2))], -1)             # (B, 4ch) avg + max pool
        return self.proj(h)


class PrefixEncoder(nn.Module):
    """Observed SOH prefix (B, K) -> (B, D).  Encoded as fade (1 - SOH) * 100 so values are O(1)."""

    def __init__(self, k: int, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.net((1.0 - prefix) * 100.0)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of a continuous time t in [0, 1] (scaled x1000 like DiT). (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = (t[:, None] * 1000.0) * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


class TimeEmbedder(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.d))


# ============================================================================ DiT
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN: x (B, N, D), shift/scale (B, D)."""
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """Pre-norm Transformer block whose LayerNorms are modulated by the conditioning vector c (adaLN-Zero).

    c (B, D) -> 6 vectors: (shift, scale, gate) for attention and for the MLP.  Gates are zero-initialised so
    every block starts as the identity - the trick that makes DiTs train stably.
    """

    def __init__(self, d: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, int(d * mlp_ratio)), nn.GELU(approximate="tanh"),
                                 nn.Linear(int(d * mlp_ratio), d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, -1)               # 6 x (B, D)
        h = modulate(self.n1(x), sh1, sc1)
        x = x + g1[:, None] * self.attn(h, h, h, need_weights=False)[0]     # (B, N, D)
        x = x + g2[:, None] * self.mlp(modulate(self.n2(x), sh2, sc2))
        return x


class TrajectoryDiT(nn.Module):
    """1-D DiT over a 256-point trajectory.  x (B, L), c (B, D) -> (B, L)."""

    def __init__(self, L: int = 256, patch: int = 16, d: int = 128, depth: int = 4, heads: int = 4):
        super().__init__()
        assert L % patch == 0
        self.L, self.P, self.N = L, patch, L // patch
        self.embed = nn.Linear(patch, d)                                    # patchify: (B, N, P) -> (B, N, D)
        self.pos = nn.Parameter(torch.zeros(1, self.N, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([DiTBlock(d, heads) for _ in range(depth)])
        self.nf = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, patch)                                      # unpatchify
        for m in (self.ada_f[-1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.embed(x.view(B, self.N, self.P)) + self.pos               # (B, N, D)
        for blk in self.blocks:
            h = blk(h, c)
        sh, sc = self.ada_f(c).chunk(2, -1)
        h = self.out(modulate(self.nf(h), sh, sc))                          # (B, N, P)
        return h.reshape(B, self.L)


class Conditioner(nn.Module):
    """c = Enc(capacity) + Enc(prefix) [+ Emb(t)].  Shared by every model so comparisons isolate the objective."""

    def __init__(self, d: int, prefix_len: int, enc_ch: int, slope: float):
        super().__init__()
        self.cap = CapacityEncoder(d, enc_ch, slope)
        self.pre = PrefixEncoder(prefix_len, d)

    def forward(self, cap, prefix):
        return self.cap(cap) + self.pre(prefix)


# ============================================================================ FlowBatt
class FlowBatt(nn.Module):
    """Conditional flow matching over normalised SOH trajectories.

    Trajectory normalisation (done outside, see data.py):  x = (fade - mu_l) / sd_l per grid point, fade = 1 - SOH.
    """

    def __init__(self, cfg_m: dict, L: int, prefix_len: int):
        super().__init__()
        d = cfg_m["d_model"]
        self.cond = Conditioner(d, prefix_len, cfg_m["enc_channels"], cfg_m["leaky_slope"])
        self.temb = TimeEmbedder(d)
        self.dit = TrajectoryDiT(L, cfg_m["patch"], d, cfg_m["depth"], cfg_m["heads"])
        self.L = L

    def velocity(self, x_t, t, c_static):
        return self.dit(x_t, c_static + self.temb(t))

    def loss(self, x1, cap, prefix):
        """x1 (B, L) normalised targets."""
        B = x1.shape[0]
        c = self.cond(cap, prefix)
        x0 = torch.randn_like(x1)                                           # noise
        t = torch.rand(B, device=x1.device)                                 # t ~ U(0, 1)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1                        # linear interpolation path
        u = x1 - x0                                                         # its (constant) velocity
        return F.mse_loss(self.velocity(xt, t, c), u)

    @torch.no_grad()
    def sample(self, cap, prefix, n: int, steps: int = 32, method: str = "euler", return_path: bool = False,
               generator: torch.Generator | None = None):
        """cap (B, E, V), prefix (B, K) -> (B, n, L) normalised trajectories (+ list of intermediate x_t)."""
        B = cap.shape[0]
        c = self.cond(cap, prefix).repeat_interleave(n, 0)                  # (B*n, D)
        x = torch.randn(B * n, self.L, device=cap.device, generator=generator)
        path = [x.view(B, n, -1).clone()] if return_path else None
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((B * n,), k * dt, device=cap.device)
            v = self.velocity(x, t, c)
            if method == "heun" and k < steps - 1:
                x_e = x + dt * v
                v2 = self.velocity(x_e, t + dt, c)
                x = x + dt * 0.5 * (v + v2)
            else:
                x = x + dt * v
            if return_path:
                path.append(x.view(B, n, -1).clone())
        out = x.view(B, n, -1)
        return (out, path) if return_path else out


# ============================================================================ DiffBatt-lite (baseline)
class DiffBattLite(nn.Module):
    """Same encoder + DiT, trained as a DDPM (epsilon prediction, cosine schedule). Mirrors the paper's DiffBatt
    comparison, which shares backbone and encoder, so the only difference is the generative objective."""

    def __init__(self, cfg_m: dict, L: int, prefix_len: int, T: int = 100):
        super().__init__()
        d = cfg_m["d_model"]
        self.cond = Conditioner(d, prefix_len, cfg_m["enc_channels"], cfg_m["leaky_slope"])
        self.temb = TimeEmbedder(d)
        self.dit = TrajectoryDiT(L, cfg_m["patch"], d, cfg_m["depth"], cfg_m["heads"])
        self.L, self.T = L, T
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        ab = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        ab = ab / ab[0]
        betas = torch.clip(1 - ab[1:] / ab[:-1], 1e-5, 0.999).float()
        self.register_buffer("betas", betas)
        self.register_buffer("abar", torch.cumprod(1 - betas, 0))

    def eps(self, x, k, c):
        return self.dit(x, c + self.temb(k.float() / self.T))

    def loss(self, x1, cap, prefix):
        B = x1.shape[0]
        c = self.cond(cap, prefix)
        k = torch.randint(0, self.T, (B,), device=x1.device)
        e = torch.randn_like(x1)
        ab = self.abar[k][:, None]
        xk = ab.sqrt() * x1 + (1 - ab).sqrt() * e
        return F.mse_loss(self.eps(xk, k, c), e)

    @torch.no_grad()
    def sample(self, cap, prefix, n: int, steps: int | None = None, generator=None, clip_x0: float | None = 12.0, **_):
        """Ancestral DDPM sampling.  clip_x0 = the standard 'clip_denoised' trick: the predicted clean trajectory
        x0_hat is clamped to the training range (|x| <= 11.1 in normalised units here) before forming the posterior
        mean.  Without it, ~1 in 10 cells produced a divergent sample in run 1 (normalised values up to 66)."""
        B = cap.shape[0]
        c = self.cond(cap, prefix).repeat_interleave(n, 0)
        x = torch.randn(B * n, self.L, device=cap.device, generator=generator)
        for k in reversed(range(self.T)):
            kk = torch.full((B * n,), k, device=cap.device, dtype=torch.long)
            b, ab = self.betas[k], self.abar[k]
            ab_prev = self.abar[k - 1] if k > 0 else torch.ones_like(ab)
            e = self.eps(x, kk, c)
            if clip_x0 is None:
                mean = (x - b / (1 - ab).sqrt() * e) / (1 - b).sqrt()
            else:                                                           # x0-parameterised posterior mean
                x0 = ((x - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-clip_x0, clip_x0)
                mean = (ab_prev.sqrt() * b / (1 - ab)) * x0 + ((1 - b).sqrt() * (1 - ab_prev) / (1 - ab)) * x
            if k > 0:
                var = b * (1 - self.abar[k - 1]) / (1 - ab)
                x = mean + var.sqrt() * torch.randn(x.shape, device=x.device, generator=generator)
            else:
                x = mean
        return x.view(B, n, -1)


# ============================================================================ deterministic baselines
class DetTransformer(nn.Module):
    """Same encoder + DiT trunk, fed a zero trajectory, trained with plain MSE: one point forecast per cell."""

    def __init__(self, cfg_m: dict, L: int, prefix_len: int):
        super().__init__()
        d = cfg_m["d_model"]
        self.cond = Conditioner(d, prefix_len, cfg_m["enc_channels"], cfg_m["leaky_slope"])
        self.dit = TrajectoryDiT(L, cfg_m["patch"], d, cfg_m["depth"], cfg_m["heads"])
        self.L = L

    def predict(self, cap, prefix):
        c = self.cond(cap, prefix)
        return self.dit(torch.zeros(cap.shape[0], self.L, device=cap.device), c)

    def loss(self, x1, cap, prefix):
        return F.mse_loss(self.predict(cap, prefix), x1)

    @torch.no_grad()
    def sample(self, cap, prefix, n: int = 1, **_):
        return self.predict(cap, prefix)[:, None].expand(-1, n, -1)


class DetMLP(nn.Module):
    """Encoder features -> MLP -> full trajectory (BatLiNet-style 'encode then regress', much simplified)."""

    def __init__(self, cfg_m: dict, L: int, prefix_len: int):
        super().__init__()
        d = cfg_m["d_model"]
        self.cond = Conditioner(d, prefix_len, cfg_m["enc_channels"], cfg_m["leaky_slope"])
        self.head = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, 2 * d), nn.SiLU(),
                                  nn.Linear(2 * d, L))

    def predict(self, cap, prefix):
        return self.head(self.cond(cap, prefix))

    def loss(self, x1, cap, prefix):
        return F.mse_loss(self.predict(cap, prefix), x1)

    @torch.no_grad()
    def sample(self, cap, prefix, n: int = 1, **_):
        return self.predict(cap, prefix)[:, None].expand(-1, n, -1)


MODELS = {"flowbatt": FlowBatt, "diffbatt": DiffBattLite, "transformer": DetTransformer, "mlp": DetMLP}


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
