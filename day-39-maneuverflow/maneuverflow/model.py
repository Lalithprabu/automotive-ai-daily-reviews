"""ManeuverFlow (this repo's reconstruction of the abstract-level design of arXiv:2610.13409).

Pipeline (one forward pass, no iterative sampling):
  SceneEncoder -> ManeuverRouter (top-k intents) -> Gaussian-mixture prior sample per intent
  -> ManeuverMoEMeanFlow (one-step MeanFlow: x = z - u(z, r=0, t=1))        [Section: generate]
  -> FutureWorldModel (predict other agents' futures) -> EvidenceScorer (current + future evidence
  per waypoint -> PDM factors) -> FrenetRefiner (local residuals) -> joint re-score of refined+retained.

Tensor conventions: B batch, N agents, C candidates, T=12 future steps, d model width.
Trajectories are normalised by NORM=(30 m, 4 m) inside the flow; all public outputs are in metres.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp
from .scoring import TF, DT, MANEUVERS

NORM = torch.tensor([30.0, 4.0])
POSN = torch.tensor([30.0, 4.0, 10.0, 2.0])      # for agent (x,y,vx,vy) normalisation
M = len(MANEUVERS)


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class SceneEncoder(nn.Module):
    """Ego history + agent tokens -> transformer. Returns cond (B,d) and agent tokens (B,N,d)."""
    def __init__(self, d, hist=8, heads=4, layers=2):
        super().__init__()
        self.ego = mlp(hist * 4, d, d)
        self.ag = mlp(6, d, d)
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, 0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)

    def forward(self, hist, agents):
        B = hist.shape[0]
        h = hist.clone()
        h[..., :2] = h[..., :2] / 30.0
        h[..., 2:] = h[..., 2:] / 10.0
        ego = self.ego(h.flatten(1))[:, None]                                   # (B,1,d)
        a = agents.clone()
        a[..., :4] = a[..., :4] / POSN.to(a)
        valid = agents[..., 5] > 0.5
        tok = self.ag(a)                                                        # (B,N,d)
        x = torch.cat([ego, tok], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~valid], 1)
        y = self.tf(x, src_key_padding_mask=pad)
        return y[:, 0], y[:, 1:], valid


class FutureWorldModel(nn.Module):
    """Stand-in for the abstract's 'world-model-predicted future features': predicts every agent's
    future positions as a residual on a constant-velocity extrapolation, plus per-step future tokens."""
    def __init__(self, d):
        super().__init__()
        self.pred = mlp(2 * d, d, TF * 2)
        self.tok = mlp(2 + d, d, d)
        nn.init.zeros_(self.pred[-1].weight); nn.init.zeros_(self.pred[-1].bias)   # start exactly at constant-velocity
        self.register_buffer("taus", torch.arange(1, TF + 1).float() * DT)

    def forward(self, cond, tok, agents):
        B, N, d = tok.shape
        res = self.pred(torch.cat([tok, cond[:, None].expand(-1, N, -1)], -1)).view(B, N, TF, 2)
        cv = agents[..., None, :2] + agents[..., None, 2:4] * self.taus[None, None, :, None]   # (B,N,T,2)
        pos = cv + res * torch.tensor([10.0, 3.0], device=res.device)
        ftok = self.tok(torch.cat([pos / NORM.to(pos)[None, None, None], tok[:, :, None].expand(-1, -1, TF, -1)], -1))
        return pos, ftok.permute(0, 2, 1, 3)                                     # pos (B,N,T,2); ftok (B,T,N,d)


class MeanFlowMoE(nn.Module):
    """Maneuver-routed mixture-of-experts MeanFlow field u(z, r, t | scene, maneuver).
    A shared trunk embeds (z, r, t, cond); a *hard-routed* expert head per maneuver outputs the average velocity."""
    def __init__(self, d, hid=128):
        super().__init__()
        self.rt = mlp(3, 32, 32)
        self.man = nn.Embedding(M, 32)
        self.trunk = mlp(TF * 2 + 32 + 32 + d, hid, hid, n=2)
        self.experts = nn.ModuleList([mlp(hid, hid, TF * 2, n=2) for _ in range(M)])

    def forward(self, z, r, t, cond, m):
        rt = self.rt(torch.stack([r, t, t - r], -1))
        h = F.silu(self.trunk(torch.cat([z, rt, self.man(m), cond], -1)))
        out = torch.zeros_like(z)
        for e in range(M):                                                       # hard routing by maneuver id
            sel = m == e
            if sel.any():
                out = out + torch.where(sel[:, None], self.experts[e](h), torch.zeros_like(z))
        return out


class EvidenceScorer(nn.Module):
    """Trajectory-specific evidence. Each waypoint queries (a) the *current* agent tokens and
    (b) the *predicted-future* agent tokens at its own timestep; pooled over time -> PDM factors [nc,dac,ep,comf]."""
    def __init__(self, d):
        super().__init__()
        self.wp = mlp(3, d, d)
        self.qk = nn.Linear(d, d)
        self.cur_k = nn.Linear(d, d)
        self.fut_k = nn.Linear(d, d)
        self.traj = mlp(TF * 2, d, d)
        self.head = mlp(5 * d + d, 2 * d, 4 + d)           # last d dims = evidence embedding reused by the refiner
        self.d = d

    def forward(self, traj, cond, tok, valid, ftok, use_future=True):
        B, C, T, _ = traj.shape
        d = self.d
        w = self.wp(torch.cat([traj / NORM.to(traj), (torch.arange(T, device=traj.device).float() / T)[None, None, :, None].expand(B, C, -1, 1)], -1))  # (B,C,T,d)
        q = self.qk(w)
        mask = valid[:, None, None, :]
        # current evidence: every waypoint attends over the N current tokens
        s = torch.einsum("bctd,bnd->bctn", q, self.cur_k(tok)) / math.sqrt(d)
        a = s.masked_fill(~mask, -1e9).softmax(-1) * mask
        cur = torch.einsum("bctn,bnd->bctd", a, tok)
        # future evidence: waypoint t attends over the predicted-future tokens *at step t*
        if use_future:
            s2 = torch.einsum("bctd,btnd->bctn", q, self.fut_k(ftok)) / math.sqrt(d)
            a2 = s2.masked_fill(~mask, -1e9).softmax(-1) * mask
            fut = torch.einsum("bctn,btnd->bctd", a2, ftok)
        else:
            fut = torch.zeros_like(cur)
        ev = torch.cat([w, cur, fut], -1)                                         # (B,C,T,3d)
        pooled = torch.cat([ev.mean(2), ev.amax(2)[..., d:]], -1)                 # (B,C,3d+2d)=5d
        tj = self.traj(traj.flatten(2) / NORM.to(traj).repeat(T))
        out = self.head(torch.cat([pooled, tj], -1))
        return out[..., :4], out[..., 4:]                                         # factor logits (B,C,4), evidence emb (B,C,d)


def score_from_logits(lg):
    p = torch.sigmoid(lg)
    return p[..., 0] * p[..., 1] * (0.8 * p[..., 2] + 0.2 * p[..., 3])


class FrenetRefiner(nn.Module):
    """Predicts bounded local residuals (ds, dd) per waypoint. Road axis == x on this synthetic road,
    so Frenet (s,d) == (x,y). Residuals are zero-initialised and bounded by tanh * (4 m, 1 m)."""
    def __init__(self, d):
        super().__init__()
        self.net = mlp(d + TF * 2, 2 * d, TF * 2)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("lim", torch.tensor([4.0, 1.0]))

    def forward(self, traj, emb):
        B, C = traj.shape[:2]
        r = self.net(torch.cat([emb, traj.flatten(2) / NORM.to(traj).repeat(TF)], -1)).view(B, C, TF, 2)
        return traj + torch.tanh(r) * self.lim


class ManeuverFlow(nn.Module):
    def __init__(self, d=64, topk=3, per_man=2, prior_mu=None, prior_sigma=0.5):
        super().__init__()
        self.d, self.topk, self.per_man, self.sigma = d, topk, per_man, prior_sigma
        self.enc = SceneEncoder(d)
        self.router = mlp(d, d, M)
        mu = prior_mu if prior_mu is not None else torch.zeros(M, TF * 2)
        self.prior_mu = nn.Parameter(mu.clone())                 # Gaussian-mixture prior: one component per maneuver
        self.flow = MeanFlowMoE(d)
        self.world = FutureWorldModel(d)
        self.scorer = EvidenceScorer(d)
        self.refiner = FrenetRefiner(d)

    # ---- helpers -------------------------------------------------------------------------------
    def prior_sample(self, m, gen=None):
        eps = torch.randn(m.shape[0], TF * 2, generator=gen, device=m.device)
        return self.prior_mu[m] + self.sigma * eps

    def encode(self, hist, agents):
        cond, tok, valid = self.enc(hist, agents)
        fpos, ftok = self.world(cond, tok, agents)
        return cond, tok, valid, fpos, ftok

    def meanflow_loss(self, x, cond, m):
        """MeanFlow identity: u = v - (t-r) * d/dt u,  v = eps - x, along z_t=(1-t)x+t*eps. Target is stop-gradient."""
        B = x.shape[0]
        eps = self.prior_sample(m)
        t = torch.rand(B, device=x.device)
        r = torch.rand(B, device=x.device) * t
        same = torch.rand(B, device=x.device) < 0.25                  # 25%: r=t -> plain flow-matching term
        r = torch.where(same, t, r)
        z = (1 - t)[:, None] * x + t[:, None] * eps
        v = eps - x
        f = lambda z_, r_, t_: self.flow(z_, r_, t_, cond, m)
        u, dudt = jvp(f, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
        u_tgt = (v - (t - r)[:, None] * dudt).detach()
        return F.mse_loss(u, u_tgt)

    @torch.no_grad()
    def _gen_one_step(self, m, cond, gen=None):
        z1 = self.prior_sample(m, gen)
        r = torch.zeros(m.shape[0], device=m.device); t = torch.ones_like(r)
        x = z1 - self.flow(z1, r, t, cond, m)                         # one-step MeanFlow sample
        return x.view(-1, TF, 2) * NORM.to(x)

    # ---- inference -------------------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, hist, agents, use_future=True, refine=True, gen=None, topk=None):
        """Returns dict with candidates (B,C,T,2), scores (B,C), best (B,T,2), router probs, future positions."""
        topk = topk or self.topk
        B = hist.shape[0]
        cond, tok, valid, fpos, ftok = self.encode(hist, agents)
        probs = self.router(cond).softmax(-1)
        sel = probs.topk(topk, -1).indices                                           # (B,k) selected maneuvers
        mm = sel[:, :, None].expand(-1, -1, self.per_man).reshape(B, -1)             # (B,C)
        C = mm.shape[1]
        cond_c = cond[:, None].expand(-1, C, -1).reshape(B * C, -1)
        cand = self._gen_one_step(mm.reshape(-1), cond_c, gen).view(B, C, TF, 2)
        lg, emb = self.scorer(cand, cond, tok, valid, ftok, use_future)
        if refine:
            ref = self.refiner(cand, emb)
            lg2, _ = self.scorer(ref, cond, tok, valid, ftok, use_future)
            all_c = torch.cat([cand, ref], 1)                                         # joint re-score: retained + refined
            all_s = torch.cat([score_from_logits(lg), score_from_logits(lg2)], 1)
            all_m = torch.cat([mm, mm], 1)
        else:
            all_c, all_s, all_m = cand, score_from_logits(lg), mm
        best = all_s.argmax(-1)
        return dict(cands=all_c, scores=all_s, maneuvers=all_m, best=all_c[torch.arange(B), best],
                    best_idx=best, probs=probs, fut_pos=fpos, raw=cand)
