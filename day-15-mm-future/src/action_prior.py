"""K-means Gaussian-mixture action prior.

Paper mechanism: "Rather than sampling from isotropic noise, the model
derives structured diversity from observed driving behavior. Training
trajectories are grouped using K-means clustering, with each cluster
defining a Gaussian-mixture prior."

We fit K-means on flattened ground-truth action-delta sequences
([dx, dy, sin(psi), cos(psi)] per future step, concatenated over the
horizon) from the training split, then store each cluster's mean and a
diagonal covariance (per-dimension std) in action-token space. At flow-matching
sampling time, each of the M proposals draws its source noise z0 from one of
these K Gaussian components (round-robin / random assignment across
proposals) instead of a single isotropic N(0, I) — giving the initial
proposals meaningfully different "intents" (e.g. go-left vs go-right)
before the flow-matching transformer even runs.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans


class ActionPrior:
    def __init__(self, num_clusters: int, future_frames: int, action_dim: int):
        self.num_clusters = num_clusters
        self.future_frames = future_frames
        self.action_dim = action_dim
        self.means = None   # [K, future_frames, action_dim]
        self.stds = None     # [K, future_frames, action_dim]

    def fit(self, action_deltas: np.ndarray):
        """action_deltas: [N, future_frames, action_dim]"""
        n = action_deltas.shape[0]
        flat = action_deltas.reshape(n, -1)
        k = min(self.num_clusters, n)
        km = KMeans(n_clusters=k, n_init=10, random_state=15).fit(flat)
        labels = km.labels_
        means, stds = [], []
        for c in range(k):
            members = flat[labels == c]
            if len(members) < 2:
                members = flat
            means.append(members.mean(axis=0))
            stds.append(members.std(axis=0) + 0.05)
        self.means = torch.tensor(np.stack(means), dtype=torch.float32).reshape(
            k, self.future_frames, self.action_dim
        )
        self.stds = torch.tensor(np.stack(stds), dtype=torch.float32).reshape(
            k, self.future_frames, self.action_dim
        )
        self.num_clusters = k
        return self

    def sample(self, batch_size: int, num_proposals: int, device) -> torch.Tensor:
        """Returns z0 action noise: [B, M, future_frames, action_dim], where
        proposal m draws from cluster (m % K)."""
        assert self.means is not None, "ActionPrior.fit() must be called first"
        k = self.num_clusters
        cluster_ids = torch.arange(num_proposals) % k
        means = self.means[cluster_ids].to(device)   # [M, future_frames, action_dim]
        stds = self.stds[cluster_ids].to(device)
        noise = torch.randn(batch_size, num_proposals, self.future_frames, self.action_dim, device=device)
        return means.unsqueeze(0) + noise * stds.unsqueeze(0)
