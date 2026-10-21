"""Compact xLSTM (sLSTM + mLSTM) cells, written from the published xLSTM equations (Beck et al. 2024).
Simplified reconstruction: no up/down-projection blocks, no causal conv; used per-BEV-cell over time."""
import math
import torch
import torch.nn as nn


class SLSTMCell(nn.Module):
    """Scalar-memory LSTM with exponential gating + log-domain stabiliser m."""
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.W = nn.Linear(d, 4 * d)
        self.R = nn.Linear(d, 4 * d, bias=False)

    def init_state(self, N, device):
        z = torch.zeros(N, self.d, device=device)
        return dict(h=z, c=z.clone(), n=z.clone(), m=z.clone())

    def forward(self, x, st):
        zi, zf, zz, zo = (self.W(x) + self.R(st["h"])).chunk(4, -1)
        m_new = torch.maximum(zf + st["m"], zi)
        i = torch.exp(zi - m_new); f = torch.exp(zf + st["m"] - m_new)
        c = f * st["c"] + i * torch.tanh(zz)
        n = f * st["n"] + i
        h = torch.sigmoid(zo) * c / n.clamp_min(1e-6)
        return h, dict(h=h, c=c, n=n, m=m_new)


class MLSTMCell(nn.Module):
    """Matrix-memory LSTM: C_t = f C_{t-1} + i v k^T, h = o * (C q) / max(|n.q|, 1), multi-head."""
    def __init__(self, d, heads=4):
        super().__init__()
        assert d % heads == 0
        self.d, self.H, self.dh = d, heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.gates = nn.Linear(d, 2 * heads)   # input / forget pre-activations per head
        self.o = nn.Linear(d, d)

    def init_state(self, N, device):
        return dict(C=torch.zeros(N, self.H, self.dh, self.dh, device=device),
                    n=torch.zeros(N, self.H, self.dh, device=device),
                    m=torch.zeros(N, self.H, device=device))

    def forward(self, x, st):
        N = x.shape[0]
        q, k, v = self.qkv(x).view(N, 3, self.H, self.dh).unbind(1)
        k = k / math.sqrt(self.dh)
        zi, zf = self.gates(x).chunk(2, -1)
        m_new = torch.maximum(zf + st["m"], zi)
        i = torch.exp(zi - m_new)[..., None]; f = torch.exp(zf + st["m"] - m_new)[..., None]
        C = f[..., None] * st["C"] + i[..., None] * torch.einsum("nhd,nhe->nhde", v, k)
        n = f * st["n"] + i * k
        num = torch.einsum("nhde,nhe->nhd", C, q)
        den = (n * q).sum(-1, keepdim=True).abs().clamp_min(1.0)
        h = (num / den).reshape(N, self.d) * torch.sigmoid(self.o(x))
        return h, dict(C=C, n=n, m=m_new)


class XLSTMStack(nn.Module):
    """Pre-LN residual stack over a sequence; `layout` like 'msm' (paper's best: m-s-m)."""
    def __init__(self, d, layout="msm", heads=4):
        super().__init__()
        self.cells = nn.ModuleList([MLSTMCell(d, heads) if c == "m" else SLSTMCell(d) for c in layout])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in layout])

    def forward(self, x):  # x: [N, K, d] -> last hidden [N, d]
        N, K, d = x.shape
        states = [c.init_state(N, x.device) for c in self.cells]
        out = None
        for t in range(K):
            h = x[:, t]
            for li, (cell, ln) in enumerate(zip(self.cells, self.norms)):
                y, states[li] = cell(ln(h), states[li])
                h = h + y
            out = h
        return out
