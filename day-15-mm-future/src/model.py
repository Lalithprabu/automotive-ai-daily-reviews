"""MMFuture: full reconstruction of the joint scene-action model from
arXiv:2609.20377 ("MM-Future: Multi-Mode Joint World-Action Modeling for
Autonomous Driving").

Pipeline per training step:
  1. MMTokenizer encodes the history camera clip -> `history_tokens`.
  2. MMTokenizer also encodes the *ground-truth future* camera clip -> the
     flow-matching target for the scene stream (`scene_z1_target`, stop-gradient).
  3. `ActionPrior` (K-means Gaussian mixture) draws M different action noise
     samples `z0_action` — one per proposal.
  4. For each of the M proposals, `BidirectionalFlowTransformer` predicts the
     velocity field at a random interpolation time t, jointly for the action
     and scene streams (block-diagonal across proposals: each is a separate
     item in a folded batch dimension).
  5. A one-step z1 estimate turns each proposal's predicted velocity into a
     candidate trajectory; the proposal closest to the real ground-truth
     trajectory is the "winning mode" (best-of-many selection).
  6. Only the winning mode's flow-matching losses (action + scene) are
     backpropagated. `ProposalScorer` is trained (cross-entropy) to rank the
     winning proposal highest among all M.
  7. A small auxiliary BEV decoder (this project's own addition, *not* part
     of the paper) turns the winning proposal's generated scene tokens into a
     coarse occupancy grid, purely so `simulate.py` has something visual to
     render for "predicted future scene" — the paper itself notes its scene
     representation is implicit / not directly interpretable.

`generate()` runs the same machinery at inference time with multi-step Euler
integration (no ground truth available) and returns ranked trajectories.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .action_prior import ActionPrior
from .flow_transformer import BidirectionalFlowTransformer
from .losses import (flow_matching_loss, gather_winner, one_step_z1_estimate,
                      scoring_loss, select_best_of_many)
from .scorer import ProposalScorer
from .tokenizer import MMTokenizer


def _group_into_chunks(cameras: torch.Tensor, frames_per_chunk: int) -> torch.Tensor:
    """[B, T, num_cam, C, crop, crop] -> [B, T//fpc, fpc, num_cam, C, crop, crop]"""
    b, t, nc, c, h, w = cameras.shape
    assert t % frames_per_chunk == 0, f"T={t} not divisible by frames_per_chunk={frames_per_chunk}"
    return cameras.reshape(b, t // frames_per_chunk, frames_per_chunk, nc, c, h, w)


class BEVDecoder(nn.Module):
    """Auxiliary, paper-independent visualization head: scene MM-Tokens for
    one future chunk -> a coarse occupancy grid, so simulate.py can show
    "predicted future scene" as a picture rather than an opaque token blob."""

    def __init__(self, d_model: int, chunk_tokens: int, out_size: int):
        super().__init__()
        self.out_size = out_size
        self.net = nn.Sequential(
            nn.Linear(chunk_tokens * d_model, out_size * out_size),
        )

    def forward(self, scene_tokens_chunk: torch.Tensor) -> torch.Tensor:
        # scene_tokens_chunk: [..., chunk_tokens, d_model]
        flat = scene_tokens_chunk.reshape(*scene_tokens_chunk.shape[:-2], -1)
        logits = self.net(flat).reshape(*scene_tokens_chunk.shape[:-2], self.out_size, self.out_size)
        return torch.sigmoid(logits)


class MMFuture(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        d = cfg["data"]
        m = cfg["model"]
        self.cfg = cfg
        self.frames_per_chunk = d["frames_per_chunk"]
        self.num_history_chunks = d["history_frames"] // d["frames_per_chunk"]
        self.num_future_chunks = d["future_frames"] // d["frames_per_chunk"]
        self.future_frames = d["future_frames"]
        self.num_proposals = m["num_proposals"]
        self.chunk_tokens = m["chunk_tokens"]
        self.d_model = m["d_model"]

        self.tokenizer = MMTokenizer(
            in_channels=2, patch_dim=m["patch_dim"], d_model=m["d_model"],
            num_cameras=d["num_cameras"], register_tokens=m["register_tokens_per_camera"],
            frames_per_chunk=d["frames_per_chunk"], chunk_tokens=m["chunk_tokens"],
            n_heads=m["n_heads"],
        )
        self.action_prior = ActionPrior(m["num_action_clusters"], d["future_frames"], m["action_dim"])
        self.flow_transformer = BidirectionalFlowTransformer(
            d_model=m["d_model"], n_heads=m["n_heads"], n_layers=m["n_flow_layers"],
            action_dim=m["action_dim"],
        )
        self.scorer = ProposalScorer(m["d_model"], m["action_dim"], m["n_heads"])
        self.bev_decoder = BEVDecoder(m["d_model"], m["chunk_tokens"], m["bev_decode_size"])

    # ------------------------------------------------------------------
    def encode_history(self, history_cameras: torch.Tensor):
        chunks = _group_into_chunks(history_cameras, self.frames_per_chunk)
        tokens = self.tokenizer(chunks)  # [B, chunks, chunk_tokens, D]
        b = tokens.shape[0]
        flat = tokens.reshape(b, -1, self.d_model)
        return flat, flat.mean(dim=1)

    def encode_future_target(self, future_cameras: torch.Tensor):
        chunks = _group_into_chunks(future_cameras, self.frames_per_chunk)
        tokens = self.tokenizer(chunks)  # [B, chunks, chunk_tokens, D]
        b = tokens.shape[0]
        return tokens.reshape(b, -1, self.d_model)

    # ------------------------------------------------------------------
    def forward_train(self, batch: dict, device) -> dict:
        history_cameras = batch["history_cameras"].to(device)
        future_cameras = batch["future_cameras"].to(device)
        action_gt = batch["action_deltas"].to(device)      # [B, Tf, 4]
        gt_trajectory = batch["gt_trajectory"].to(device)   # [B, Tf, 2]
        ego_history = batch["ego_history"].to(device)        # [B, Th, 2]
        frames = batch["frames"].to(device)                   # [B, T, 2, g, g]

        b = history_cameras.shape[0]
        m = self.num_proposals

        history_tokens, history_summary = self.encode_history(history_cameras)
        with torch.no_grad():
            scene_z1_target = self.encode_future_target(future_cameras)  # stop-gradient target
        ts = scene_z1_target.shape[1]
        ta = self.future_frames

        z0_action = self.action_prior.sample(b, m, device)               # [B, M, Ta, 4]
        z0_scene = torch.randn(b, m, ts, self.d_model, device=device)      # [B, M, Ts, D]
        z1_action = action_gt.unsqueeze(1).expand(-1, m, -1, -1)            # [B, M, Ta, 4]
        z1_scene = scene_z1_target.unsqueeze(1).expand(-1, m, -1, -1)        # [B, M, Ts, D]

        t = torch.rand(b, m, device=device)
        t_bc_a = t.unsqueeze(-1).unsqueeze(-1)
        z_t_action = (1 - t_bc_a) * z0_action + t_bc_a * z1_action
        z_t_scene = (1 - t_bc_a) * z0_scene + t_bc_a * z1_scene

        th = history_tokens.shape[1]
        hist_rep = history_tokens.unsqueeze(1).expand(-1, m, -1, -1).reshape(b * m, th, self.d_model)
        v_action, v_scene = self.flow_transformer(
            hist_rep,
            z_t_action.reshape(b * m, ta, -1),
            z_t_scene.reshape(b * m, ts, -1),
            t.reshape(b * m),
        )
        v_action = v_action.reshape(b, m, ta, -1)
        v_scene = v_scene.reshape(b, m, ts, -1)

        z1_action_hat = one_step_z1_estimate(z_t_action, v_action, t)
        z1_scene_hat = one_step_z1_estimate(z_t_scene, v_scene, t)

        start_pos = ego_history[:, -1, :].unsqueeze(1).unsqueeze(1)  # [B,1,1,2]
        trajectory_hat = start_pos + torch.cumsum(z1_action_hat[..., :2], dim=2)  # [B, M, Ta, 2]

        m_star = select_best_of_many(trajectory_hat, gt_trajectory)

        action_loss_pp = flow_matching_loss(v_action, z0_action, z1_action)  # [B, M]
        scene_loss_pp = flow_matching_loss(v_scene, z0_scene, z1_scene)        # [B, M]
        action_loss = gather_winner(action_loss_pp, m_star)
        scene_loss = gather_winner(scene_loss_pp, m_star)

        scores = self.scorer(history_summary, z1_action_hat.detach(), z1_scene_hat.detach())
        score_loss = scoring_loss(scores, m_star)

        # --- auxiliary BEV visualization decoder (not part of the paper) ---
        winner_scene = z1_scene_hat.gather(
            1, m_star.view(b, 1, 1, 1).expand(-1, 1, ts, self.d_model)
        ).squeeze(1)  # [B, Ts, D]
        winner_scene_chunks = winner_scene.reshape(b, self.num_future_chunks, self.chunk_tokens, self.d_model)
        bev_pred = self.bev_decoder(winner_scene_chunks)  # [B, num_future_chunks, out, out]
        bev_gt = self._downsample_future_occupancy(frames)
        bev_loss = torch.nn.functional.binary_cross_entropy(bev_pred, bev_gt)

        loss_cfg = self.cfg["loss"]
        total = (loss_cfg["action_weight"] * action_loss
                 + loss_cfg["scene_weight"] * scene_loss
                 + loss_cfg["score_weight"] * score_loss
                 + 0.05 * bev_loss)

        with torch.no_grad():
            ade = (trajectory_hat.gather(1, m_star.view(b, 1, 1, 1).expand(-1, 1, ta, 2)).squeeze(1)
                   - gt_trajectory).norm(dim=-1).mean()
            top1_acc = (scores.argmax(dim=1) == m_star).float().mean()

        return {
            "loss": total, "action_loss": action_loss.detach(), "scene_loss": scene_loss.detach(),
            "score_loss": score_loss.detach(), "bev_loss": bev_loss.detach(),
            "ade": ade, "top1_acc": top1_acc,
        }

    def _downsample_future_occupancy(self, frames: torch.Tensor) -> torch.Tensor:
        b, t, c, g, _ = frames.shape
        hist = self.num_history_chunks * self.frames_per_chunk
        future = frames[:, hist:, :, :, :]  # [B, Tf, 2, g, g]
        combined = future.amax(dim=2)  # collapse channels -> [B, Tf, g, g]
        chunks = combined.reshape(b, self.num_future_chunks, self.frames_per_chunk, g, g).amax(dim=2)
        out = self.bev_decoder.out_size
        pooled = torch.nn.functional.adaptive_avg_pool2d(chunks, (out, out))
        return pooled.clamp(0, 1)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, history_cameras: torch.Tensor, ego_last_pos: torch.Tensor,
                 device, ode_steps: int = 8):
        """Multi-step Euler integration from t=0 to t=1, run at inference
        time (no ground truth). Returns ranked trajectories + scores +
        decoded BEV occupancy for every proposal.
        """
        self.eval()
        history_cameras = history_cameras.to(device)
        b = history_cameras.shape[0]
        m = self.num_proposals
        history_tokens, history_summary = self.encode_history(history_cameras)
        th = history_tokens.shape[1]

        ts = self.num_future_chunks * self.chunk_tokens
        ta = self.future_frames

        z_action = self.action_prior.sample(b, m, device)
        z_scene = torch.randn(b, m, ts, self.d_model, device=device)

        hist_rep = history_tokens.unsqueeze(1).expand(-1, m, -1, -1).reshape(b * m, th, self.d_model)
        dt = 1.0 / ode_steps
        start = ego_last_pos.to(device).unsqueeze(1).unsqueeze(1)
        trace = [start.expand(-1, m, ta, -1) + torch.cumsum(z_action[..., :2], dim=2)]
        score_trace = [self.scorer(history_summary, z_action, z_scene)]
        for step in range(ode_steps):
            t_val = step * dt
            t = torch.full((b * m,), t_val, device=device)
            v_a, v_s = self.flow_transformer(
                hist_rep, z_action.reshape(b * m, ta, -1), z_scene.reshape(b * m, ts, -1), t
            )
            z_action = z_action + dt * v_a.reshape(b, m, ta, -1)
            z_scene = z_scene + dt * v_s.reshape(b, m, ts, -1)
            trace.append(start.expand(-1, m, ta, -1) + torch.cumsum(z_action[..., :2], dim=2))
            score_trace.append(self.scorer(history_summary, z_action, z_scene))

        trajectory = trace[-1]
        scores = score_trace[-1]
        probs = torch.softmax(scores, dim=1)

        scene_chunks = z_scene.reshape(b, m, self.num_future_chunks, self.chunk_tokens, self.d_model)
        bev_pred = self.bev_decoder(scene_chunks)  # [B, M, num_future_chunks, out, out]

        return {
            "trajectories": trajectory,     # [B, M, Ta, 2]
            "scores": scores,                # [B, M]
            "probs": probs,                   # [B, M]
            "bev_pred": bev_pred,              # [B, M, chunks, out, out]
            "trace": torch.stack(trace, dim=0),        # [ode_steps+1, B, M, Ta, 2]
            "score_trace": torch.stack(score_trace, dim=0),  # [ode_steps+1, B, M]
        }
