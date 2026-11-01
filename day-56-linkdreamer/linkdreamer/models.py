"""Agentic-LinkDreamer-style components (my reconstruction from the paper's abstract; dimensions are this repo's toy).

Paper (arXiv 2610.39258): causal collaborator memory + relative-motion-conditioned recurrent world model that
imagines the missing collaborator BEV features during a V2X outage + a differentiable Takagi-Sugeno neuro-fuzzy
agent that picks one of five modes from a 5-D decision state.

Tensor legend: B batch, K memory frames, G BEV size, C hidden channels, R fuzzy rules.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .synthetic import G, EGO_CELLS, shift_rows

MODES = ("ego-only", "reuse (motion-comp.)", "imagine (world model)", "low-rate refresh", "high-rate refresh")
COMM = torch.tensor([0., 0., 0., .25, 1.])        # relative payload of each mode (1.0 = full BEV feature map)
LAT = torch.tensor([0., 0., 1., .5, 1.5])         # relative latency of each mode (toy units)


class ConvGRUCell(nn.Module):
    """h' = (1-z)*h + z*tanh(W[x, r*h]) with convolutional gates; x [B,Cx,G,G], h [B,C,G,G]."""

    def __init__(self, cx, ch, k=5):
        super().__init__()
        self.gates = nn.Conv2d(cx + ch, 2 * ch, k, padding=k // 2)
        self.cand = nn.Conv2d(cx + ch, ch, k, padding=k // 2)

    def forward(self, x, h):
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)          # update / reset gates [B,C,G,G]
        n = torch.tanh(self.cand(torch.cat([x, r * h], 1)))                          # candidate state
        return (1 - z) * h + z * n


class LatentWorldModel(nn.Module):
    """Relative-motion-conditioned recurrent world model.

    1) Encode the K most recent *received* collaborator frames (oldest->newest); between frames the hidden state is
       warped by the ego displacement (EGO_CELLS * gap) so the memory stays registered to the ego frame.
    2) For every missing frame since the last reception, warp the hidden state by one ego step and update the GRU with
       no observation (imagination).  The number of imagination steps = age of the memory (varies per sample).
    3) Decode mean (evidence, visibility) and a log-variance for each -> aleatoric+epistemic 'prediction uncertainty'.
    """

    def __init__(self, ch=16, k=5):
        super().__init__()
        self.ch = ch
        self.enc = nn.Sequential(nn.Conv2d(2, ch, 3, padding=1), nn.ReLU())
        self.cell = ConvGRUCell(ch + 2, ch, k)                                       # input = enc(obs) + [observed?, age/8]
        self.dec = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(), nn.Conv2d(ch, 4, 1))

    def forward(self, hist, gaps, ages):
        """hist [B,K,2,G,G] received frames (own ego frames); gaps [B,K-1] frames between them; ages [B] frames since newest.
        -> mean [B,2,G,G] in (0,1), logvar [B,2,G,G]."""
        B, K = hist.shape[:2]
        dev = hist.device
        h = torch.zeros(B, self.ch, G, G, device=dev)
        one, zero = torch.ones(B, 1, G, G, device=dev), torch.zeros(B, 1, G, G, device=dev)
        for k in range(K):                                                           # ---- encode memory
            if k > 0:
                h = shift_rows(h, EGO_CELLS * gaps[:, k - 1])                        # ego motion between receptions
            h = self.cell(torch.cat([self.enc(hist[:, k]), one, zero], 1), h)
        blank = torch.zeros(B, self.ch, G, G, device=dev)
        n_steps = int(ages.max().item()) if len(ages) else 0
        for a in range(1, n_steps + 1):                                              # ---- imagine missing frames
            cond = torch.cat([blank, zero, torch.full_like(zero, a / 8.0)], 1)
            h2 = self.cell(cond, shift_rows(h, torch.full_like(ages, EGO_CELLS)))
            m = (a <= ages).float()[:, None, None, None]                             # only samples that still need steps
            h = m * h2 + (1 - m) * h
        out = self.dec(h)
        return torch.sigmoid(out[:, :2]), out[:, 2:].clamp(-6, 2)


class TSNeuroFuzzy(nn.Module):
    """Differentiable zero/first-order Takagi-Sugeno controller.
    Rule r fires with  w_r(x) = softmax_r( -1/2 * sum_i ((x_i - c_ri)/s_ri)^2 )   (normalised Gaussian memberships)
    and its consequent is linear: y_r(x) = A_r x + b_r  (one score per mode).  Output = sum_r w_r y_r.   x in [0,1]^5."""

    def __init__(self, n_in=5, n_rules=9, n_out=5, centers=None):
        super().__init__()
        self.c = nn.Parameter(centers.clone() if centers is not None else torch.rand(n_rules, n_in))   # [R,5]
        self.log_s = nn.Parameter(torch.full((n_rules, n_in), -1.0))                                    # [R,5] (s~0.37)
        self.A = nn.Parameter(torch.randn(n_rules, n_out, n_in) * .01)                                  # [R,5,5]
        self.b = nn.Parameter(torch.zeros(n_rules, n_out))                                              # [R,5]

    def forward(self, x):                                                            # x [B,5] -> scores [B,5]
        z = (x[:, None, :] - self.c[None]) / self.log_s.exp()[None]                  # [B,R,5]
        w = torch.softmax(-0.5 * (z ** 2).sum(-1), dim=1)                            # [B,R] normalised firing strengths
        y = torch.einsum("roi,bi->bro", self.A, x) + self.b[None]                    # [B,R,5] linear consequents
        return (w[:, :, None] * y).sum(1)

    STATE = ("uncertainty", "complementarity", "freshness", "comm-quality", "budget-left")

    def describe(self, top=9):
        names = ["ego-only", "reuse", "imagine", "low-rate", "high-rate"]
        out = []
        for r in range(self.c.shape[0]):
            c, s = self.c[r].detach(), self.log_s[r].exp().detach()
            ante = ", ".join(f"{n}~{c[i]:.2f}" for i, n in enumerate(self.STATE))
            bias = self.b[r].detach()
            out.append(f"R{r}: IF {ante} THEN prefers {names[int(bias.argmax())]}  (bias {[round(float(v), 2) for v in bias]})")
        return out


class FusionDetector(nn.Module):
    """Centre-heatmap head on [ego(2), collaborator input(2), sigma(1), mode one-hot(5)] = 10 channels."""

    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(10, ch, 3, padding=1), nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=2, dilation=2), nn.ReLU(),
                                 nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(), nn.Conv2d(ch, 1, 1))

    def forward(self, ego, coll, sig, mode):
        """ego [B,2,G,G]; coll [B,2,G,G]; sig [B,1,G,G]; mode [B] long -> logits [B,1,G,G]."""
        oh = F.one_hot(mode, 5).float()[:, :, None, None].expand(-1, -1, G, G)
        return self.net(torch.cat([ego, coll, sig, oh], 1))


def build_modes(wm, col_cur, hist, gaps, ages, has_mem):
    """Collaborator input each mode would feed to the detector.  -> coll [B,5,2,G,G], sig [B,5,1,G,G], wm_sigma [B]
    col_cur is only used by the two refresh modes (it is what would arrive over the link)."""
    B = col_cur.shape[0]
    hm = has_mem.float()[:, None, None, None]
    mean, logvar = wm(hist, gaps, ages)
    sigma = (0.5 * logvar[:, :1]).exp()                                              # [B,1,G,G] std of the evidence channel
    reuse = shift_rows(hist[:, -1], EGO_CELLS * ages) * hm                           # [B,2,G,G] motion-compensated stale frame
    imagine = mean * hm
    low = F.interpolate(F.avg_pool2d(col_cur, 4), scale_factor=4, mode="nearest")    # [B,2,G,G] 4x coarser summary
    zeros = torch.zeros_like(col_cur)
    coll = torch.stack([zeros, reuse, imagine, low, col_cur], 1)
    z1 = torch.zeros_like(sigma)
    sig = torch.stack([z1, z1, sigma * hm + (1 - hm), z1, z1], 1)
    w_sigma = (sigma * hm).flatten(1).mean(1) * hm.flatten()                         # scalar uncertainty per sample
    return coll, sig, w_sigma, reuse


def decision_state(ego, reuse, wm_sigma, ages, has_mem, q, budget):
    """5-D decision state in [0,1]: prediction uncertainty, perceptual complementarity, memory freshness,
    communication quality, residual budget."""
    hm = has_mem.float()
    ev = reuse[:, :1]                                                                # stale collaborator evidence, ego frame
    comp = (ev * (1 - ego[:, 1:2])).flatten(1).sum(1) / (ev.flatten(1).sum(1) + 1e-3)   # share of it lying in the ego's blind area
    u = (wm_sigma / 0.4).clamp(0, 1)
    fresh = torch.exp(-ages.float() / 4.0) * hm
    return torch.stack([u, comp.clamp(0, 1) * hm, fresh, q, budget], 1)


def mode_mask(q, has_mem):
    """Availability [B,5] bool: reuse/imagine need memory; low-rate needs q>=0.5; high-rate needs q==1."""
    B = q.shape[0]
    m = torch.ones(B, 5, dtype=torch.bool)
    m[:, 1] = has_mem; m[:, 2] = has_mem
    m[:, 3] = q >= .5; m[:, 4] = q >= 1.
    return m
