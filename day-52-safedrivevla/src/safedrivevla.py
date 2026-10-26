"""SafeDriveVLA-Mini: a small PyTorch reconstruction of the *idea* in SafeDriveVLA (arXiv:2610.20609):
  (1) a THREE-LEVEL navigation-compliance mode (FOLLOW / ADAPT / REFUSE) that separates conflict reasoning from
      action generation, and
  (2) a navigation-conditioned world model that 'dreams' what happens if the ego STRICTLY FOLLOWS the instruction
      (not the expert / planned action), so the compliance decision is grounded in a simulated consequence.
The paper is a full VLA on Bench2Drive; this repo keeps only the decision structure on a synthetic 3-lane toy.
All dimensions / heads here are this repo's own defaults (see SOURCING.md).

Tensor legend: B batch, K agents, D model dim, H dream horizon.
"""
import torch
import torch.nn as nn

NUM_NAV, NUM_MODES = 3, 3


class SceneEncoder(nn.Module):
    """Agents [B,K,6] (lane one-hot 3, s/30, v/6, valid) + ego lane one-hot [B,3] -> tokens [B,K+1,D], pooled z0 [B,D]."""

    def __init__(self, dim, heads, layers):
        super().__init__()
        self.agent_in, self.ego_in = nn.Linear(6, dim), nn.Linear(3, dim)
        enc = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)

    def forward(self, agents, ego):
        tok = torch.cat([self.ego_in(ego)[:, None], self.agent_in(agents)], 1)                  # [B,K+1,D]
        pad = torch.cat([torch.zeros_like(agents[:, :1, 0]), 1 - agents[..., 5]], 1).bool()     # True = ignore
        tok = self.tf(tok, src_key_padding_mask=pad)                                           # [B,K+1,D]
        keep = (~pad).float().unsqueeze(-1)
        z0 = (tok * keep).sum(1) / keep.sum(1).clamp(min=1)                                    # masked mean pool [B,D]
        return tok, pad, z0


class NavDreamer(nn.Module):
    """World model: latent z_t is rolled forward H steps while being CONDITIONED ON THE NAVIGATION INSTRUCTION
    (i.e. it dreams strict compliance). Each step re-reads the scene tokens via cross-attention (so it can
    extrapolate agent motion s + v*t) and emits a collision logit and a normalised min-gap."""

    def __init__(self, dim, heads, horizon):
        super().__init__()
        self.H = horizon
        self.nav_emb = nn.Embedding(NUM_NAV, dim)
        self.step_emb = nn.Embedding(horizon, dim)
        self.read = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cell = nn.GRUCell(2 * dim, dim)
        self.risk_head = nn.Linear(dim, 1)
        self.gap_head = nn.Linear(dim, 1)

    def forward(self, z0, tok, pad, nav):
        z, n = z0, self.nav_emb(nav)                                                           # [B,D]
        zs, risk, gap = [], [], []
        for t in range(self.H):
            step = self.step_emb.weight[t].expand_as(z)                                        # [B,D]
            q = (z + n + step)[:, None]                                                        # [B,1,D]
            ctx = self.read(q, tok, tok, key_padding_mask=pad, need_weights=False)[0][:, 0]    # [B,D]
            z = self.cell(torch.cat([ctx, n + step], -1), z)                                   # z_t -> z_{t+1}
            zs.append(z); risk.append(self.risk_head(z)); gap.append(self.gap_head(z))
        return torch.stack(zs, 1), torch.cat(risk, 1), torch.cat(gap, 1)                        # [B,H,D] [B,H] [B,H]


class SafeDriveVLAMini(nn.Module):
    def __init__(self, dim=64, heads=4, enc_layers=2, horizon=8, use_dream=True):
        super().__init__()
        self.H, self.use_dream = horizon, use_dream
        self.enc = SceneEncoder(dim, heads, enc_layers)
        self.dream = NavDreamer(dim, heads, horizon)
        self.nav_emb = nn.Embedding(NUM_NAV, dim)
        self.dream_pool = nn.Sequential(nn.Linear(dim + 2, dim), nn.GELU())
        self.attn_q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pool = nn.MultiheadAttention(dim, heads, batch_first=True)
        # compliance head: scene summary + instruction (+ dreamed consequence) -> {FOLLOW, ADAPT, REFUSE}
        self.mode_head = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, NUM_MODES))
        # action head conditioned on the compliance mode: lateral progress per step in [0,1]
        self.mode_emb = nn.Embedding(NUM_MODES, dim)
        self.action_head = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, horizon))

    def forward(self, agents, ego, nav, forced_mode=None):
        tok, pad, z0 = self.enc(agents, ego)                                                   # [B,K+1,D] [B,D]
        out = {}
        if self.use_dream:
            zs, risk, gap = self.dream(z0, tok, pad, nav)                                      # dream = strict following
            out["risk"], out["gap"] = risk, gap
            feat = self.dream_pool(torch.cat([zs, risk[..., None].sigmoid(), gap[..., None]], -1))   # [B,H,D]
            summ = self.pool(self.attn_q.expand(len(z0), -1, -1), feat, feat, need_weights=False)[0][:, 0]
        else:
            summ = z0                                                                          # ablation: no consequence rollout
        h = torch.cat([summ, self.nav_emb(nav)], -1)                                           # [B,2D]
        out["mode_logits"] = self.mode_head(h)                                                 # [B,3]
        mode = out["mode_logits"].argmax(-1) if forced_mode is None else forced_mode
        out["mode"] = mode
        a_in = torch.cat([summ, self.mode_emb(mode)], -1)
        out["profile"] = torch.sigmoid(self.action_head(a_in))                                 # [B,H] lateral progress
        return out


def compute_loss(out, batch, w):
    bce = nn.functional.binary_cross_entropy_with_logits
    parts = {}
    if "risk" in out:
        pos = batch["coll"].mean().clamp(1e-3, 1 - 1e-3)
        parts["risk"] = bce(out["risk"], batch["coll"], pos_weight=((1 - pos) / pos).clamp(max=5.0))
        parts["gap"] = nn.functional.smooth_l1_loss(out["gap"], batch["gap"])
    parts["mode"] = nn.functional.cross_entropy(out["mode_logits"], batch["mode"])
    parts["action"] = nn.functional.mse_loss(out["profile"], batch["profile"])
    total = sum(w.get(k, 1.0) * v for k, v in parts.items())
    return total, parts
