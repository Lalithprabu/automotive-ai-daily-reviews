"""Fourier positional encoding for (x, y, t) query coordinates.

READ (arXiv:2609.12371) queries a learned risk field at arbitrary
continuous space-time coordinates. A raw (x, y, t) triple is far too
low-dimensional for an MLP/attention stack to represent a high-frequency
function of, so we lift each coordinate into a bank of sinusoids before
it ever touches a learned weight (the classic NeRF-style trick).

Reconstruction note / bug we already hit and fixed:
    An earlier version of this file used num_freqs=8 with a max frequency
    of 6.0. That let the coordinate MLP fit arbitrarily high-frequency
    functions of (x, y, t) -- including pure noise in the *gaps* between
    sparse training probes, since nothing constrained the field there.
    The visible symptom was a checkerboard/speckle artifact in the risk
    heatmap far from any agent. Two changes fixed it:
      1. Lower the encoding's maximum frequency (4.0, not 6.0) so the
         field's representational capacity better matches how sparsely
         we actually supervise it.
      2. Add a background-risk regularizer to the training loss (see
         `read_model.py`) that explicitly pulls risk -> 0 at random
         (x, y, t) points that are not near any agent.
    Both are baked into this reconstruction from the start.
"""

import math

import torch
import torch.nn as nn


class FourierPositionalEncoding(nn.Module):
    """Encodes N-D coordinates with sin/cos Fourier features.

    For each input dimension d and each frequency band f_i, we append
    sin(f_i * pi * x_d) and cos(f_i * pi * x_d) to the output, plus the
    raw coordinates themselves (a small but consistently helpful trick
    that lets the network still see the coordinate's absolute scale).

    Args:
        num_input_dims: number of raw coordinate dims (3 for x, y, t).
        num_freqs: number of frequency bands.
        max_freq: highest frequency band (log-spaced from 2^0 up to this).
    """

    def __init__(self, num_input_dims: int = 3, num_freqs: int = 8, max_freq: float = 4.0):
        super().__init__()
        self.num_input_dims = num_input_dims
        self.num_freqs = num_freqs
        self.max_freq = max_freq

        # Log-spaced frequency bands in [2^0, max_freq].
        freq_bands = torch.logspace(
            start=0.0,
            end=math.log2(max_freq),
            steps=num_freqs,
            base=2.0,
        )
        self.register_buffer("freq_bands", freq_bands)  # [num_freqs]

        # raw coords + (sin, cos) per freq per input dim
        self.output_dim = num_input_dims * (1 + 2 * num_freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [..., num_input_dims] raw (x, y, t) coordinates.
        Returns:
            [..., output_dim] Fourier-encoded coordinates.
        """
        # coords: [..., D] -> scaled: [..., D, num_freqs]
        scaled = coords.unsqueeze(-1) * self.freq_bands * math.pi  # [..., D, F]
        sin_feats = torch.sin(scaled)  # [..., D, F]
        cos_feats = torch.cos(scaled)  # [..., D, F]

        flat_sin = sin_feats.flatten(start_dim=-2)  # [..., D*F]
        flat_cos = cos_feats.flatten(start_dim=-2)  # [..., D*F]

        # Concatenate raw coords + sin bank + cos bank.
        encoded = torch.cat([coords, flat_sin, flat_cos], dim=-1)  # [..., D*(1+2F)]
        return encoded
