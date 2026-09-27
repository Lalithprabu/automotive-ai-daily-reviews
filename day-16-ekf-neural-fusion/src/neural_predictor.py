"""
NeuralTrajectoryPredictor: a compact reconstruction of Trajectron++'s core
recurrent-CVAE trajectory decoder, NOT the full graph-structured multi-agent
Trajectron++.

The real Trajectron++ (Salzmann et al., 2020) encodes a dynamic spatiotemporal
graph over multiple interacting agents plus semantic map context, and decodes
a discrete-latent CVAE over multimodal futures. Here, to keep the repo small,
CPU-trainable in minutes, and focused on the *fusion* mechanism that is this
paper's actual contribution, we reconstruct only its single-agent recurrent
core:

    history (GRU encoder) --> h_hist
    [training only] future (GRU encoder) --> h_fut
    posterior q(z | h_hist, h_fut)   vs.   prior p(z | h_hist)   (CVAE)
    z, h_hist --> GRU decoder (autoregressive) --> one best-guess future traj

This is an explicitly disclosed simplification (this project's standing
convention: baselines simplified for a scaled-down repo are labeled as such,
never presented as the full paper architecture).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick: z = mu + sigma * eps, eps ~ N(0, I)."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence(mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor:
    """Analytic KL( N(mu_q, var_q) || N(mu_p, var_p) ) for diagonal Gaussians,
    summed over the latent dimension, averaged over the batch.

    KL = 0.5 * sum[ log(var_p/var_q) + (var_q + (mu_q-mu_p)^2)/var_p - 1 ]
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (
        (logvar_p - logvar_q)
        + (var_q + (mu_q - mu_p) ** 2) / var_p
        - 1.0
    )
    return kl.sum(dim=-1).mean()


class NeuralTrajectoryPredictor(nn.Module):
    """
    Shapes (B = batch size, T_h = history_len, T_f = future_len):
        history : [B, T_h, 2]   observed past (x, y) positions
        future  : [B, T_f, 2]   ground-truth future (x, y) positions (train only)
        output  : [B, T_f, 2]   predicted future (x, y) positions
    """

    def __init__(self, hidden_size: int = 64, latent_size: int = 16,
                 encoder_layers: int = 1, decoder_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # --- encoders ---
        self.hist_gru = nn.GRU(input_size=2, hidden_size=hidden_size,
                                num_layers=encoder_layers, batch_first=True)
        self.fut_gru = nn.GRU(input_size=2, hidden_size=hidden_size,
                               num_layers=encoder_layers, batch_first=True)

        # --- CVAE latent: prior p(z|h_hist), posterior q(z|h_hist,h_fut) ---
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, latent_size * 2),  # -> [mu | logvar]
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, latent_size * 2),  # -> [mu | logvar]
        )

        # --- decoder: GRUCell run autoregressively over the horizon ---
        self.dec_init = nn.Linear(hidden_size + latent_size, hidden_size)
        self.decoder_cell = nn.GRUCell(input_size=2, hidden_size=hidden_size)
        self.output_head = nn.Linear(hidden_size, 2)  # -> predicted (dx, dy)

    def _encode_history(self, history: torch.Tensor) -> torch.Tensor:
        # history: [B, T_h, 2] -> GRU final hidden state of last layer: [B, H]
        _, h_n = self.hist_gru(history)
        return h_n[-1]  # [B, hidden_size]

    def _encode_future(self, future: torch.Tensor) -> torch.Tensor:
        # future: [B, T_f, 2] -> [B, H]
        _, h_n = self.fut_gru(future)
        return h_n[-1]

    def forward(self, history: torch.Tensor, future: torch.Tensor = None,
                future_len: int = 12, sample_z: bool = True):
        """
        Returns:
            pred_future : [B, future_len, 2]
            kl          : scalar tensor (KL loss term) if `future` is given,
                          else None
        """
        B = history.shape[0]
        h_hist = self._encode_history(history)  # [B, H]

        prior_out = self.prior_net(h_hist)  # [B, 2*latent]
        mu_p, logvar_p = prior_out.chunk(2, dim=-1)

        kl = None
        if future is not None:
            # --- training: use posterior latent, compute KL to prior ---
            h_fut = self._encode_future(future)  # [B, H]
            post_out = self.posterior_net(torch.cat([h_hist, h_fut], dim=-1))
            mu_q, logvar_q = post_out.chunk(2, dim=-1)
            z = _reparameterize(mu_q, logvar_q)
            kl = kl_divergence(mu_q, logvar_q, mu_p, logvar_p)
        else:
            # --- inference: no ground-truth future available, use the prior ---
            z = _reparameterize(mu_p, logvar_p) if sample_z else mu_p

        dec_hidden = torch.tanh(self.dec_init(torch.cat([h_hist, z], dim=-1)))  # [B, H]

        # autoregressive decoding: feed the previous predicted displacement
        # (dx, dy) as input to the next GRU step, starting from the last
        # observed velocity (history[:, -1] - history[:, -2]).
        cur_pos = history[:, -1, :]                       # [B, 2] last observed pos
        input_disp = history[:, -1, :] - history[:, -2, :]  # [B, 2] last observed vel*dt

        outputs = []
        for _ in range(future_len):
            dec_hidden = self.decoder_cell(input_disp, dec_hidden)  # [B, H]
            disp = self.output_head(dec_hidden)                      # [B, 2] predicted step displacement
            cur_pos = cur_pos + disp                                  # [B, 2] integrate to absolute position
            outputs.append(cur_pos)
            input_disp = disp

        pred_future = torch.stack(outputs, dim=1)  # [B, future_len, 2]
        return pred_future, kl
