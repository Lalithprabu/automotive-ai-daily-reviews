"""PyTorch core: ID-MAE, a multimodal masked-diffusion autoencoder for driver-impairment inference.

Re-implementation from the ABSTRACT of arXiv 2611.23211 (Gideon et al., Toyota Research Institute).  The abstract
says: impairment is predicted "as a masked token" while masked gaze and trajectory are reconstructed, as a
self-supervised objective on top of impairment prediction.  Everything below the level of that sentence (patching,
sizes, mask schedule, loss weights, sampler) is my construction; see SOURCING.md.

Tokens (B batch, T = 100 steps at 10 Hz = 10 s, patch P = 5 steps -> N = 20 patches per stream):

    gaze  stream  (B, T, 2) -> patches (B, N, 10) -> Linear -> (B, N, D)      maskable, reconstructed
    veh   stream  (B, T, 4) -> patches (B, N, 20) -> Linear -> (B, N, D)      maskable, reconstructed ("trajectory")
    road  stream  (B, T, 1) -> patches (B, N, 5)  -> Linear -> (B, N, D)      scene context, never masked
    label token   class id in {0..C-1} or [MASK]=C -> Embedding -> (B, 1, D)

    x_i = content_i + modality_emb[m_i] + sinusoidal_time(n_i)          (label token: modality emb only)
    sequence = [label | gaze_1..N | veh_1..N | road_1..N]               (B, 1 + 3N, D) = (B, 61, 128)
    H = TransformerEncoder(sequence)                                    bidirectional, pre-norm, 4 layers
    heads: label logits = W_c H[:, 0]            (B, C)
           gaze patch   = W_g H[:, 1:1+N]        (B, N, 10)
           veh  patch   = W_v H[:, 1+N:1+2N]     (B, N, 20)

Masked-diffusion training (absorbing-state "diffusion" over tokens, as in MDLM / MaskGIT-style models):
    t ~ U(t_min, 1) per sample  =  the noise level;   every maskable token (gaze, veh, label) is replaced by its
    modality's [MASK] embedding independently with probability t.
    L = CE(label logits, y)   on samples whose label token is masked AND labelled
      + lambda * MSE(reconstruction, target)   averaged over masked gaze / veh patches
    Unlabelled samples keep the label token masked and only contribute the reconstruction term.
    lambda = 0 gives the "no reconstruction" ablation (same model, same masking, label loss only).

Inference:
    classify  -> all gaze / veh / road visible, label masked, softmax(label logits)
    forecast  -> future patches masked, label masked (unknown), iterative unmasking in K steps (nearest first);
                 K = 1 is a one-shot prediction.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

P = 5                                   # steps per patch (0.5 s at 10 Hz)
CH = dict(gaze=2, veh=4, road=1)        # channels per stream


def patchify(x: torch.Tensor, p: int = P) -> torch.Tensor:
    """(B, T, C) -> (B, T/p, p*C), time-major inside each patch."""
    B, T, C = x.shape
    return x.reshape(B, T // p, p * C)


def unpatchify(x: torch.Tensor, c: int, p: int = P) -> torch.Tensor:
    """(B, N, p*C) -> (B, N*p, C)."""
    B, N, _ = x.shape
    return x.reshape(B, N * p, c)


def sinusoid(n: int, d: int) -> torch.Tensor:
    """Standard sinusoidal table (n, d)."""
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, dtype=torch.float32)
    ang = pos / (10000 ** (i / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang)
    return pe


class IDMAE(nn.Module):
    def __init__(self, n_classes: int = 4, n_patches: int = 20, d: int = 128, layers: int = 4, heads: int = 4,
                 ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.C, self.N, self.d = n_classes, n_patches, d
        self.inp = nn.ModuleDict({k: nn.Sequential(nn.Linear(P * c, d), nn.GELU(), nn.Linear(d, d))
                                  for k, c in CH.items()})
        self.mask_tok = nn.ParameterDict({k: nn.Parameter(torch.randn(d) * 0.02) for k in ("gaze", "veh")})
        self.label_emb = nn.Embedding(n_classes + 1, d)                 # index C = [MASK]
        self.mod_emb = nn.Embedding(4, d)                               # 0 label, 1 gaze, 2 veh, 3 road
        self.register_buffer("pe", sinusoid(n_patches, d), persistent=False)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head_c = nn.Linear(d, n_classes)
        self.head_g = nn.Linear(d, P * CH["gaze"])
        self.head_v = nn.Linear(d, P * CH["veh"])

    # ------------------------------------------------------------------------------------------ core pass
    def forward(self, gaze, veh, road, label, mask_g, mask_v):
        """gaze (B,T,2) veh (B,T,4) road (B,T,1); label (B,) with C meaning [MASK]; mask_g / mask_v (B,N) bool.

        Returns label logits (B,C), gaze recon (B,N,10), veh recon (B,N,20), embedding of the label token (B,D).
        """
        B = gaze.shape[0]
        eg = self.inp["gaze"](patchify(gaze))                          # (B, N, D)
        ev = self.inp["veh"](patchify(veh))                            # (B, N, D)
        er = self.inp["road"](patchify(road))                          # (B, N, D)
        eg = torch.where(mask_g[..., None], self.mask_tok["gaze"].expand_as(eg), eg)   # absorb masked tokens
        ev = torch.where(mask_v[..., None], self.mask_tok["veh"].expand_as(ev), ev)
        me = self.mod_emb.weight
        eg = eg + me[1] + self.pe                                      # + modality + time
        ev = ev + me[2] + self.pe
        er = er + me[3] + self.pe
        el = (self.label_emb(label) + me[0])[:, None]                  # (B, 1, D)
        h = self.norm(self.enc(torch.cat([el, eg, ev, er], 1)))        # (B, 1+3N, D)
        N = self.N
        return (self.head_c(h[:, 0]), self.head_g(h[:, 1:1 + N]), self.head_v(h[:, 1 + N:1 + 2 * N]), h[:, 0])

    # ------------------------------------------------------------------------------------------ training loss
    def loss(self, gaze, veh, road, y, labelled, lam: float = 1.0, t_min: float = 0.1):
        """Masked-diffusion objective.  y (B,) class ids; labelled (B,) bool.  Returns (total, parts dict)."""
        B, N, dev = gaze.shape[0], self.N, gaze.device
        t = torch.rand(B, 1, device=dev) * (1 - t_min) + t_min          # noise level per sample
        mask_g = torch.rand(B, N, device=dev) < t
        mask_v = torch.rand(B, N, device=dev) < t
        mask_l = (torch.rand(B, device=dev) < t[:, 0]) | ~labelled      # unlabelled -> label always masked
        lab_in = torch.where(mask_l, torch.full_like(y, self.C), y)
        logits, rg, rv, _ = self(gaze, veh, road, lab_in, mask_g, mask_v)
        sup = mask_l & labelled
        l_cls = F.cross_entropy(logits[sup], y[sup]) if sup.any() else logits.sum() * 0
        tg, tv = patchify(gaze), patchify(veh)
        eg = ((rg - tg) ** 2).mean(-1)                                  # (B, N) per-patch MSE
        ev = ((rv - tv) ** 2).mean(-1)
        l_rec = ((eg * mask_g).sum() + (ev * mask_v).sum()) / (mask_g.sum() + mask_v.sum()).clamp(min=1)
        return l_cls + lam * l_rec, dict(cls=float(l_cls.detach()), rec=float(l_rec.detach()))

    # ------------------------------------------------------------------------------------------ inference
    @torch.no_grad()
    def classify(self, gaze, veh, road):
        """Impairment posterior (B, C) and label-token embedding (B, D), with every observation visible."""
        B, N = gaze.shape[0], self.N
        z = torch.zeros(B, N, dtype=torch.bool, device=gaze.device)
        lab = torch.full((B,), self.C, dtype=torch.long, device=gaze.device)
        logits, _, _, emb = self(gaze, veh, road, lab, z, z)
        return logits.softmax(-1), emb

    @torch.no_grad()
    def forecast(self, gaze, veh, road, n_future: int, steps: int = 4, label: torch.Tensor | None = None):
        """Predict the last n_future patches of gaze and veh by iterative unmasking (nearest patch first).

        gaze / veh: their future part is ignored (will be masked).  road must cover the full window (map known).
        label: optional (B,) known condition; default [MASK].  Returns full gaze (B,T,2) and veh (B,T,4) with the
        future filled in.
        """
        B, N = gaze.shape[0], self.N
        g, v = gaze.clone(), veh.clone()
        masked = torch.zeros(B, N, dtype=torch.bool, device=g.device)
        masked[:, N - n_future:] = True
        lab = label if label is not None else torch.full((B,), self.C, dtype=torch.long, device=g.device)
        order = list(range(N - n_future, N))
        chunks = [order[round(i * len(order) / steps):round((i + 1) * len(order) / steps)] for i in range(steps)]
        for ch in chunks:
            _, rg, rv, _ = self(g, v, road, lab, masked, masked)
            pg, pv = unpatchify(rg, 2), unpatchify(rv, 4)
            for n in ch:                                              # commit these patches as "observed"
                sl = slice(n * P, (n + 1) * P)
                g[:, sl], v[:, sl] = pg[:, sl], pv[:, sl]
                masked[:, n] = False
        return g, v


class GRUClassifier(nn.Module):
    """'Old way' baseline: a supervised GRU over the concatenated per-step signals (7 channels), labels only."""

    def __init__(self, n_classes: int = 4, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(7, hidden, layers, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, n_classes))

    def forward(self, gaze, veh, road):
        out, _ = self.gru(torch.cat([gaze, veh, road], -1))           # (B, T, H)
        z = torch.cat([out[:, -1], out.mean(1)], -1)                  # last + mean pooling
        return self.head(z)

    @torch.no_grad()
    def classify(self, gaze, veh, road):
        return self(gaze, veh, road).softmax(-1), None


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":                                           # shape smoke test
    m = IDMAE()
    B, T = 3, 100
    g, v, r = torch.randn(B, T, 2), torch.randn(B, T, 4), torch.randn(B, T, 1)
    L, parts = m.loss(g, v, r, torch.tensor([0, 1, 2]), torch.tensor([True, False, True]))
    p, e = m.classify(g, v, r)
    fg, fv = m.forecast(g, v, r, n_future=4)
    print(tuple(p.shape), tuple(e.shape), tuple(fg.shape), tuple(fv.shape), parts, n_params(m), "params",
          math.isfinite(float(L)))
