"""Differentiable Continuous Wavelet Transform + scale compression (CWT-RUL-Mini).

Math: W(s, t) = IFFT[ FFT(x)(w) * conj(psi_hat(s*w)) * sqrt(2*pi*s) ], psi = analytic Morlet.
We return |W| (scalogram magnitude). 'Scale compression' = average groups of adjacent scales to cut
redundancy between neighbouring (highly correlated) scales -- this repo's reading of the paper's idea.
Tensor legend: B batch, C signal channels, T samples, S scales, S' = S // compress.
"""
import math
import torch
import torch.nn as nn


class MorletCWT(nn.Module):
    def __init__(self, T, n_scales=32, w0=6.0, s_min=1.5, s_max=40.0):
        super().__init__()
        self.T, self.w0 = T, w0
        scales = torch.logspace(math.log10(s_min), math.log10(s_max), n_scales)           # [S] log-spaced
        omega = 2 * math.pi * torch.fft.fftfreq(T)                                        # [T] angular freq
        arg = scales[:, None] * omega[None, :]                                            # [S,T]
        psi = (math.pi ** -0.25) * torch.exp(-0.5 * (arg - w0) ** 2) * (omega[None] > 0)  # analytic Morlet
        psi = psi * torch.sqrt(2 * math.pi * scales[:, None])                             # L2 normalisation
        self.register_buffer("psi", psi)                                                  # [S,T]
        self.register_buffer("scales", scales)

    def forward(self, x):                                   # x: [B,C,T]
        X = torch.fft.fft(x, dim=-1)                        # [B,C,T] complex
        W = torch.fft.ifft(X[:, :, None, :] * self.psi[None, None], dim=-1)  # [B,C,S,T]
        return W.abs()                                      # scalogram magnitude [B,C,S,T]


class ScaleCompression(nn.Module):
    """Average-pool groups of `factor` adjacent scales -> [B,C,S/factor,T], then log1p to tame dynamic range."""

    def __init__(self, factor=4):
        super().__init__()
        self.factor = factor

    def forward(self, w):
        B, C, S, T = w.shape
        S2 = S // self.factor
        w = w[:, :, : S2 * self.factor].reshape(B, C, S2, self.factor, T).mean(3)         # [B,C,S',T]
        return torch.log1p(w)
