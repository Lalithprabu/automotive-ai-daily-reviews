"""INTERACT-style object-centric world model + planner (my reconstruction from the abstract-level description).

Stage 1  recurrent slot attention  : frames -> K slots/frame, slots of frame t-1 initialise frame t (temporal structure)
Stage 2  latent ego action         : inverse-dynamics head infers a discrete latent action z_t from (S_t, S_{t+1});
                                     a slot transformer predicts S_{t+1} from (S_t, z_t)   (ego + surrounding entities co-evolve)
Stage 3  planner distillation      : planner scores candidate ego actions using slots + *predicted future slots*
Tensor legend: B batch, L frames, K slots, d slot dim, A candidate actions, F future steps.
"""
import math
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG = 48


def grid(h, w, device=None):
    ys, xs = torch.linspace(0, 1, h, device=device), torch.linspace(0, 1, w, device=device)
    g = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), -1)                      # [h,w,2]
    return torch.cat([g, 1 - g], -1)                                                # [h,w,4]


class SlotAttention(nn.Module):
    """Locatello et al. slot attention: softmax over *slots* (competition), weighted-mean update, GRU + residual MLP."""
    def __init__(s, d=64, iters=3, hidden=128):
        super().__init__()
        s.iters, s.d = iters, d
        s.n_in, s.n_s, s.n_m = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        s.q, s.k, s.v = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        s.gru = nn.GRUCell(d, d)
        s.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def forward(s, tokens, slots, iters=None):
        B, K, d = slots.shape
        tokens = s.n_in(tokens)                                                     # [B,N,d]
        k, v = s.k(tokens), s.v(tokens)
        for _ in range(iters or s.iters):
            prev = slots
            q = s.q(s.n_s(slots))                                                   # [B,K,d]
            att = torch.softmax(torch.einsum("bkd,bnd->bkn", q, k) / math.sqrt(d), dim=1) + 1e-8   # compete over slots
            att = att / att.sum(-1, keepdim=True)                                   # weighted mean over tokens
            upd = torch.einsum("bkn,bnd->bkd", att, v)                              # [B,K,d]
            slots = s.gru(upd.reshape(-1, d), prev.reshape(-1, d)).view(B, K, d)
            slots = slots + s.mlp(s.n_m(slots))
        return slots


class SlotEncoder(nn.Module):
    """[B,L,3,48,48] -> slots [B,L,K,d]. Recurrent: S_{t-1} initialises slot attention at frame t."""
    def __init__(s, K=7, d=64):
        super().__init__()
        s.K, s.d = K, d
        s.cnn = nn.Sequential(nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.Conv2d(32, 32, 5, stride=2, padding=2), nn.ReLU(),
                              nn.Conv2d(32, 32, 5, padding=2), nn.ReLU(), nn.Conv2d(32, d, 5, padding=2))   # 48 -> 24
        s.pos = nn.Linear(4, d)
        s.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        s.sa = SlotAttention(d)
        s.init = nn.Parameter(torch.randn(K, d) * 0.5)                              # learned, per-slot-identity init
        s.cond = nn.Sequential(nn.Linear(8, d), nn.ReLU(), nn.Linear(d, d))                    # first-frame object conditioning

    def init_from_frame(s, frame):
        """SAVi-style first-frame conditioning: one slot per connected component (centroid, size, class colour), rest = background.
        frame [B,3,H,W] -> slot init [B,K,d]. (Why: unconditioned slot attention collapsed to identical slots in my CPU budget.)"""
        B = frame.shape[0]; feats = torch.zeros(B, s.K, 8); present = torch.zeros(B, s.K, 1)
        fg = (frame.amax(1) > 0.5).cpu().numpy()
        for b in range(B):
            lab, n = ndimage.label(fg[b])
            comps = []
            for i in range(1, n + 1):
                rr, cc = np.nonzero(lab == i)
                if len(rr) >= 3: comps.append((len(rr), rr, cc, i))
            for k, (area, rr, cc, i) in enumerate(sorted(comps, key=lambda z: -z[0])[: s.K - 1]):
                col = frame[b][:, torch.as_tensor(rr), torch.as_tensor(cc)].mean(1)                      # which channel(s) are lit
                feats[b, k] = torch.tensor([rr.mean() / 48, cc.mean() / 48, (rr.max() - rr.min() + 1) / 48, (cc.max() - cc.min() + 1) / 48, *col.tolist(), 1.0])
                present[b, k] = 1
        return s.init[None] + s.cond(feats.to(frame.device)) * present.to(frame.device)

    def features(s, x):
        f = s.cnn(x).permute(0, 2, 3, 1)                                            # [B,24,24,d]
        f = f + s.pos(grid(f.shape[1], f.shape[2], x.device))
        return s.mlp(f.flatten(1, 2))                                               # [B,576,d]

    def step(s, frame, prev=None):
        """frame [B,3,H,W]; prev slots [B,K,d] or None -> slots [B,K,d]."""
        tok = s.features(frame)
        if prev is None:
            return s.sa(tok, s.init_from_frame(frame), iters=3)
        return s.sa(tok, prev, iters=2)

    def forward(s, frames):
        B, L = frames.shape[:2]
        tok = s.features(frames.flatten(0, 1)).view(B, L, -1, s.d)
        out, prev = [], None
        for t in range(L):
            prev = s.sa(tok[:, t], s.init_from_frame(frames[:, t]), iters=3) if prev is None else s.sa(tok[:, t], prev, iters=2)
            out.append(prev)
        return torch.stack(out, 1)


class SlotDecoder(nn.Module):
    """Spatial-broadcast decoder: slots [B,K,d] -> image [B,3,48,48], masks [B,K,48,48] (mask softmax over slots)."""
    def __init__(s, d=64, c=24):
        super().__init__()
        s.pos = nn.Linear(4, d)
        s.net = nn.Sequential(nn.ConvTranspose2d(d, c, 3, 2, 1, output_padding=1), nn.ReLU(),
                              nn.ConvTranspose2d(c, c, 3, 2, 1, output_padding=1), nn.ReLU(),
                              nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, 4, 3, padding=1))
        with torch.no_grad(): s.net[-1].bias[:3] = -4.0                                   # start from a black image (sigmoid(-4)~0.02)

    def forward(s, slots):
        B, K, d = slots.shape
        x = slots.reshape(B * K, 1, 1, d) + s.pos(grid(12, 12, slots.device))[None]  # broadcast over a 12x12 grid
        out = s.net(x.permute(0, 3, 1, 2)).view(B, K, 4, IMG, IMG)
        masks = torch.softmax(out[:, :, 3], dim=1)                                  # [B,K,H,W]
        img = (masks[:, :, None] * torch.sigmoid(out[:, :, :3])).sum(1)
        return img, masks


class LatentAction(nn.Module):
    """Inverse dynamics: (S_t, S_{t+1}) -> one of `ncode` discrete latent ego actions (Gumbel straight-through).
    No action labels are used to train it: the 2.3-bit bottleneck forces it to keep what best explains the change."""
    def __init__(s, K=7, d=64, ncode=5):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(2 * K * d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, ncode))
        s.codebook = nn.Parameter(torch.randn(ncode, d) * 0.5)

    def forward(s, s0, s1, tau=1.0):
        logits = s.net(torch.cat([s0.flatten(1), (s1 - s0).flatten(1)], -1))        # [B,ncode]
        probs = torch.softmax(logits, -1)
        y = F.gumbel_softmax(logits, tau=tau, hard=True) if s.training else F.one_hot(logits.argmax(-1), logits.shape[-1]).float()
        return y @ s.codebook, probs, y                                             # code vector [B,d]


class FuturePredictor(nn.Module):
    """S_{t+1} = S_t + f(S_t, z_t): self-attention over the K slots (+1 action token) so entities interact."""
    def __init__(s, d=64, heads=4, layers=2):
        super().__init__()
        s.cproj = nn.Linear(d, d)
        s.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True), layers)
        s.out = nn.Linear(d, d)

    def forward(s, slots, code):
        h = s.tr(torch.cat([slots, s.cproj(code)[:, None]], 1))                     # [B,K+1,d]
        return slots + s.out(h[:, :-1])

    def rollout(s, slots, code, n):
        outs = []
        for _ in range(n):
            slots = s(slots, code); outs.append(slots)
        return torch.stack(outs, 1)                                                 # [B,n,K,d]


class WorldModel(nn.Module):
    def __init__(s, K=7, d=64, ncode=5):
        super().__init__()
        s.enc, s.dec, s.lat, s.pred = SlotEncoder(K, d), SlotDecoder(d), LatentAction(K, d, ncode), FuturePredictor(d)
        s.K, s.d = K, d


class SlotPlanner(nn.Module):
    """Scores each candidate ego action (grounded latent-action embedding). use_future=True feeds the predicted slots."""
    def __init__(s, K=7, d=64, use_future=True):
        super().__init__()
        s.use_future = use_future
        n = K * d * (3 if use_future else 1) + d
        s.net = nn.Sequential(nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(s, S0, fut, aemb):
        """S0 [B,K,d]; fut [B,A,F,K,d] (predicted slots per candidate action); aemb [A,d] -> costs [B,A]."""
        B, A = S0.shape[0], aemb.shape[0]
        ln = lambda z: F.layer_norm(z, z.shape[-1:])                                # per-slot standardisation (slot scale is arbitrary)
        parts = [ln(S0).flatten(1)[:, None].expand(B, A, -1)]
        if s.use_future:
            parts += [ln(fut[:, :, 0]).flatten(2), ln(fut[:, :, -1]).flatten(2)]
        parts.append(aemb[None].expand(B, A, -1))
        return s.net(torch.cat(parts, -1)).squeeze(-1)


class DensePlanner(nn.Module):
    """'Old way' baseline: dense conv features of the last two frames -> costs; interactions implicit, no object structure."""
    def __init__(s, A=5):
        super().__init__()
        s.cnn = nn.Sequential(nn.Conv2d(6, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU())  # 48->6
        s.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 36, 256), nn.ReLU(), nn.Linear(256, A))

    def forward(s, f_prev, f_cur):
        return s.head(s.cnn(torch.cat([f_prev, f_cur], 1)))


@torch.no_grad()
def candidate_futures(wm, S0, aemb, n=3):
    """Roll the predictor n steps for every candidate action embedding. S0 [B,K,d], aemb [A,d] -> [B,A,n,K,d]."""
    B, A = S0.shape[0], aemb.shape[0]
    s = S0[:, None].expand(B, A, -1, -1).reshape(B * A, *S0.shape[1:])
    c = aemb[None].expand(B, A, -1).reshape(B * A, -1)
    return wm.pred.rollout(s, c, n).view(B, A, n, *S0.shape[1:])
