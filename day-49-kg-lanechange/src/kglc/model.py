"""Interpretable lane-change predictor: TransE knowledge-graph embeddings + Bayesian inference.

Knowledge graph (hypothesis-centric, this repo's reconstruction):
    (Maneuver, hasLatVel, left) (Maneuver, hasTTC_front, HIGH) ...
Every training sample contributes one triple per relation: (label, relation, observed symbolic value).
TransE learns  h + r ~= t  so that  d(h,r,t) = ||h + r - t||_2  is small for facts that co-occur.

Bayesian head (naive-Bayes in embedding space):
    P(h | e_1..e_K) ∝ P(h) * Π_k P(e_k | h),     P(e_k | h) = softmax_{v in V_rel(k)}( -d(h, r_k, v) / tau )
tau is a single scalar calibrated post-hoc on validation NLL. Missing evidence (sensor dropout) is
handled by simply omitting that factor -- a property black-box nets do not have for free.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import N_ENT, N_REL, REL_VALUES, MANEUVERS


class TransE(nn.Module):
    def __init__(self, dim: int = 100, gamma: float = 6.0, adv_temp: float = 1.0):
        super().__init__()
        self.dim, self.gamma, self.adv_temp = dim, gamma, adv_temp
        self.ent = nn.Embedding(N_ENT, dim)          # (N_ENT, D)
        self.rel = nn.Embedding(N_REL, dim)          # (N_REL, D)
        nn.init.uniform_(self.ent.weight, -6 / dim ** 0.5, 6 / dim ** 0.5)
        nn.init.uniform_(self.rel.weight, -6 / dim ** 0.5, 6 / dim ** 0.5)
        # per-relation candidate-value index tensors (support of the likelihood softmax)
        for r, vals in enumerate(REL_VALUES):
            self.register_buffer(f"vals_{r}", torch.tensor(vals), persistent=False)
        self.log_tau = nn.Parameter(torch.zeros(()), requires_grad=False)   # calibrated post hoc

    def vals(self, r: int) -> torch.Tensor:
        return getattr(self, f"vals_{r}")

    def dist(self, h_idx, r_idx, t_idx):
        """L2 translation distance.  h:(...,), r:(...,), t:(...,) -> (...,)"""
        h, r, t = self.ent(h_idx), self.rel(r_idx), self.ent(t_idx)   # (...,D)
        return torch.linalg.vector_norm(h + r - t, dim=-1)

    def cond_logp(self, h_idx: torch.Tensor, r: int) -> torch.Tensor:
        """log P(value | h, relation r) over the relation's candidate values.
        h_idx:(H,) -> (H, |V_r|)"""
        vals = self.vals(r)                                           # (V,)
        H, V = h_idx.shape[0], vals.shape[0]
        d = self.dist(h_idx[:, None].expand(H, V),
                      torch.full((H, V), r, device=h_idx.device, dtype=torch.long),
                      vals[None, :].expand(H, V))                     # (H,V)
        return F.log_softmax(-d / self.log_tau.exp(), dim=-1)

    # ---- training loss: self-adversarial negative sampling (RotatE-style) on TransE -----------
    def loss(self, ents: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """ents:(B,K) entity ids of observed values; label:(B,) maneuver id (0..2)."""
        B, K = ents.shape
        total = 0.0
        for r in range(K):
            vals = self.vals(r)                                        # (V,)
            V = vals.shape[0]
            pos_d = self.dist(label, torch.full_like(label, r), ents[:, r])           # (B,)
            neg_t = vals[None, :].expand(B, V)                                        # (B,V) all candidates...
            neg_d = self.dist(label[:, None].expand(B, V),
                              torch.full((B, V), r, device=label.device, dtype=torch.long), neg_t)
            mask = (neg_t != ents[:, r][:, None])                                     # ...except the positive
            w = torch.softmax((-neg_d * self.adv_temp).masked_fill(~mask, -1e9), dim=-1).detach()
            neg_term = -(w * F.logsigmoid(neg_d - self.gamma)).sum(-1)               # (B,)
            pos_term = -F.logsigmoid(self.gamma - pos_d)
            total = total + (pos_term + neg_term).mean()
        return total / K

    # ---- Bayesian inference ---------------------------------------------------------------------
    def log_posterior(self, ents: torch.Tensor, log_prior: torch.Tensor | None = None,
                      observed: torch.Tensor | None = None) -> torch.Tensor:
        """ents:(B,K); observed:(B,K) bool (False = evidence missing). Returns (B,3) normalised log P(h|e)."""
        B, K = ents.shape
        hyp = torch.arange(3, device=ents.device)
        lp = torch.zeros(B, 3, device=ents.device) if log_prior is None else log_prior.expand(B, 3).clone()
        for r in range(K):
            ll = self.cond_logp(hyp, r)                                # (3,V)
            vals = self.vals(r)
            idx = (ents[:, r][:, None] == vals[None, :]).float().argmax(-1)           # (B,) position in V
            term = ll[:, idx].T                                        # (B,3)
            if observed is not None:
                term = term * observed[:, r][:, None].float()
            lp = lp + term
        return F.log_softmax(lp, dim=-1)

    @torch.no_grad()
    def explain(self, ents_row: torch.Tensor, relation_names: list[str]) -> list[tuple[str, float]]:
        """Per-evidence log-likelihood-ratio contributions (nats): log P(e|LLC)-log P(e|LK) and RLC-LK."""
        out = []
        hyp = torch.arange(3)
        for r in range(ents_row.shape[0]):
            ll = self.cond_logp(hyp, r)
            vals = self.vals(r)
            k = int((ents_row[r] == vals).float().argmax())
            out.append((relation_names[r], float(ll[0, k] - ll[1, k]), float(ll[2, k] - ll[1, k])))
        return out

    @torch.no_grad()
    def calibrate_tau(self, ents, label, log_prior=None, grid=None):
        """1-D search for the temperature minimising validation NLL of the posterior."""
        grid = grid or [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
        best, best_nll = 1.0, 1e9
        for t in grid:
            self.log_tau.fill_(float(torch.log(torch.tensor(t))))
            nll = F.nll_loss(self.log_posterior(ents, log_prior), label).item()
            if nll < best_nll:
                best, best_nll = t, nll
        self.log_tau.fill_(float(torch.log(torch.tensor(best))))
        return best, best_nll


class MLPBaseline(nn.Module):
    """'Old way' black-box: MLP on standardised raw numeric features (7 -> H -> H -> 3)."""
    def __init__(self, hidden=64, in_dim=7):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.register_buffer("mu", torch.zeros(in_dim)); self.register_buffer("sd", torch.ones(in_dim))

    def forward(self, x):                      # (B,7) -> (B,3) logits
        # TTC features are log-compressed so the 0.8..30 s range is well scaled
        x = x.clone(); x[:, 2:] = torch.log(x[:, 2:].clamp(min=0.5))
        return self.net((x - self.mu) / self.sd)


def macro_f1(pred, true, n=3):
    f1s = []
    for c in range(n):
        tp = ((pred == c) & (true == c)).sum().item()
        fp = ((pred == c) & (true != c)).sum().item()
        fn = ((pred != c) & (true == c)).sum().item()
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return sum(f1s) / n
