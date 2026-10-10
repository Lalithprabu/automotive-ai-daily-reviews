"""Stand-in 'unreliable predictor' for the paper's frozen VLM (Qwen / Cosmos).

The paper treats the VLM as a black box emitting a collision judgement; SLLCP only needs a
per-label confidence score. This repo substitutes a tiny MLP trained briefly with plain CE
(no class balancing) so that it is -- like an off-the-shelf VLM -- overconfident toward the
majority 'safe' label. This is a disclosed reconstruction choice, not a paper-sourced model.
"""
import torch
import torch.nn as nn


class SafetyScorer(nn.Module):
    def __init__(self, in_dim: int = 10, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):  # x: (B, in_dim) -> logits (B, 2): [safe, unsafe]
        return self.net(x)

    @torch.no_grad()
    def probs(self, x):
        return torch.softmax(self(x), -1)  # (B, 2)
