"""eCoMAE-style model (my reconstruction from the paper's method description; sizes/loss weights partly the paper's, see SOURCING.md).

Masked-autoencoder over multivariate time windows with
  * a visible-token-only Transformer encoder (masked steps are NOT fed to the encoder),
  * a decoder with learnable mask embeddings that outputs a reconstruction AND a Normal-Inverse-Gamma (NIG) head per step/feature,
  * observability alignment: (1) inverse-variance-weighted mean consistency to an unmasked reference pass on visible steps,
    (2) evidence suppression on masked steps (2*nu+alpha <= cap), (3) cosine representation consistency across mask views.
Tensor legend: B batch, L steps, F features, d model dim, K mask views.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn


def block_mask(valid, rate, block, gen=None):
    """Block-wise time-step masking on valid steps only.  valid [B,L] bool -> masked [B,L] bool (masked implies valid)."""
    B, L = valid.shape
    masked = torch.zeros_like(valid)
    n_target = (valid.sum(1).float() * rate).round().long()                              # [B] steps to hide
    for b in range(B):
        idx = valid[b].nonzero().squeeze(1)
        if len(idx) == 0: continue
        got = 0
        while got < n_target[b]:
            s = idx[torch.randint(len(idx), (1,), generator=gen)].item()
            sel = [j for j in range(s, min(s + block, L)) if valid[b, j] and not masked[b, j]]
            sel = sel[: int(n_target[b]) - got]
            if not sel: 
                if masked[b][idx].all(): break
                continue
            masked[b, sel] = True; got += len(sel)
    return masked


class eCoMAE(nn.Module):
    def __init__(self, n_feat=5, L=32, d=64, enc_layers=2, dec_layers=1, heads=4, n_cls=4):
        super().__init__()
        self.L, self.d, self.F = L, d, n_feat
        self.inp = nn.Linear(n_feat, d)
        self.pos = nn.Parameter(torch.zeros(1, L, d)); nn.init.trunc_normal_(self.pos, std=0.02)
        mk = lambda: nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(mk(), enc_layers, enable_nested_tensor=False)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, d)); nn.init.trunc_normal_(self.mask_tok, std=0.02)
        self.decoder = nn.TransformerEncoder(mk(), dec_layers, enable_nested_tensor=False)
        self.rec_head = nn.Linear(d, n_feat)
        self.nig_head = nn.Linear(d, 4 * n_feat)                                          # (gamma, nu, alpha, beta) per feature
        self.cls_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_cls))

    # ---------- encoder on visible tokens only ----------
    def encode(self, x, visible):
        """x [B,L,F], visible [B,L] bool -> tokens placed back at their positions [B,L,d] (zeros elsewhere).
        Only visible tokens enter attention: they are packed to the front, padded, and the pad is key-masked."""
        B, L, _ = x.shape
        order = torch.argsort((~visible).int(), dim=1, stable=True)                      # visible indices first   [B,L]
        n = visible.sum(1)                                                               # [B]
        m = max(int(n.max()), 1)
        idx = order[:, :m]                                                               # [B,m]
        h = self.inp(x) + self.pos                                                       # [B,L,d]
        h = torch.gather(h, 1, idx[..., None].expand(-1, -1, self.d))                    # [B,m,d] packed visible tokens
        pad = torch.arange(m, device=x.device)[None] >= n[:, None]                       # [B,m] True = padding
        h = self.encoder(h, src_key_padding_mask=pad)
        h = h.masked_fill(pad[..., None], 0)
        out = torch.zeros(B, L, self.d, device=x.device).scatter(1, idx[..., None].expand(-1, -1, self.d), h)
        return out

    def pool(self, tok, visible):
        """Masked mean over visible steps -> [B,d]."""
        w = visible.float()[..., None]
        return (tok * w).sum(1) / w.sum(1).clamp(min=1)

    # ---------- decoder with NIG head ----------
    def decode(self, tok, visible, valid):
        """tok [B,L,d] encoder outputs at visible steps; every other step gets the mask embedding.  -> dict of [B,L,F]."""
        h = torch.where(visible[..., None], tok, self.mask_tok.expand_as(tok)) + self.pos
        h = self.decoder(h, src_key_padding_mask=~valid)
        rec = self.rec_head(h)
        g, nu, a, b = self.nig_head(h).chunk(4, dim=-1)
        nu, a, b = Fn.softplus(nu) + 1e-4, Fn.softplus(a) + 1.0 + 1e-3, Fn.softplus(b) + 1e-4     # nu>0, alpha>1, beta>0
        return dict(rec=rec, mu=g, nu=nu, alpha=a, beta=b)

    @staticmethod
    def variances(o):
        """Predictive var = beta(1+nu)/(nu(alpha-1)); aleatoric = beta/(alpha-1); epistemic = beta/(nu(alpha-1))."""
        ale = o["beta"] / (o["alpha"] - 1)
        epi = o["beta"] / (o["nu"] * (o["alpha"] - 1))
        return ale + epi, ale, epi

    # ---------- classification (fine-tuning / inference) ----------
    def classify(self, x, valid):
        z = self.pool(self.encode(x, valid), valid)                                      # all valid steps visible
        return self.cls_head(z)


def nig_nll(y, g, nu, a, b):
    """Amini et al. 2020 Student-t NLL of the NIG evidential distribution (elementwise)."""
    om = 2 * b * (1 + nu)
    return 0.5 * torch.log(math.pi / nu) - a * torch.log(om) + (a + 0.5) * torch.log(nu * (y - g) ** 2 + om) + torch.lgamma(a) - torch.lgamma(a + 0.5)


def nig_reg(y, g, nu, a):
    return (y - g).abs() * (2 * nu + a)


def _mean(t, m):                                                                         # masked mean of [B,L,F] with [B,L] mask
    m = m[..., None].float()
    return (t * m).sum() / (m.sum() * t.shape[-1]).clamp(min=1)


def pretrain_loss(model, x, valid, rate=0.6, block=3, k_views=2, w=None, phi_cap=2.0, use_evi=True):
    """Self-supervised eCoMAE objective.  Returns (total, dict of terms).  use_evi=False -> plain Ti-MAE (reconstruction only)."""
    w = w or dict(rec=1.0, evi=1.0, reg=1e-2, mu=0.05, alpha=0.05, rep=0.10)
    with torch.no_grad():                                                                # unmasked reference pass (no gradient)
        ref = model.decode(model.encode(x, valid), valid, valid)
        var_r, _, _ = model.variances(ref)
        omega = 1.0 / (var_r + 1e-2)
        omega = omega / omega.mean()
    toks, vis_list, terms = [], [], dict(rec=0., evi=0., mu=0., alpha=0.)
    for _ in range(k_views):
        masked = block_mask(valid, rate, block)
        vis = valid & ~masked
        tok = model.encode(x, vis)
        o = model.decode(tok, vis, valid)
        terms["rec"] = terms["rec"] + _mean((o["rec"] - x) ** 2, masked)                  # L_rec on masked steps
        if use_evi:
            nll = nig_nll(x, o["mu"], o["nu"], o["alpha"], o["beta"]) + w["reg"] * nig_reg(x, o["mu"], o["nu"], o["alpha"])
            terms["evi"] = terms["evi"] + _mean(nll, masked)                              # L_evi on masked steps
            hub = Fn.huber_loss(o["mu"], ref["mu"], reduction="none", delta=1.0)
            terms["mu"] = terms["mu"] + _mean(omega * hub, vis)                           # L_mu: visible-step consistency to reference
            terms["alpha"] = terms["alpha"] + _mean(Fn.relu(2 * o["nu"] + o["alpha"] - phi_cap), masked)   # L_alpha: evidence cap on masked steps
        toks.append(tok); vis_list.append(vis)
    terms = {k: v / k_views for k, v in terms.items()}
    rep = x.new_zeros(())
    if use_evi and k_views > 1:                                                          # L_rep: cosine over overlap-visible positions
        n_pair, cnt = 0, 0
        for i in range(k_views):
            for j in range(i + 1, k_views):
                ov = vis_list[i] & vis_list[j]
                ok = ov.any(1)
                if ok.any():
                    zi, zj = model.pool(toks[i], ov)[ok], model.pool(toks[j], ov)[ok]
                    rep = rep + (1 - Fn.cosine_similarity(zi, zj, dim=-1)).mean(); n_pair += 1
        terms["rep"] = rep / max(n_pair, 1)
    else:
        terms["rep"] = rep
    total = w["rec"] * terms["rec"] + (w["evi"] * terms["evi"] + w["mu"] * terms["mu"] + w["alpha"] * terms["alpha"] + w["rep"] * terms["rep"] if use_evi else 0.0)
    return total, {k: float(torch.as_tensor(v).detach()) for k, v in terms.items()}
