"""CWT-RUL-Mini: early-cycle battery life regression from compressed wavelet scalograms.

Pipeline: cycles [B,N,C,T] -> CWT -> scale compression -> shared 2-D CNN per cycle -> mean-pool over the N
input cycles (N can be 1..M, so the same net handles 'single-cycle' and multi-cycle input) -> MLP -> log(life).
GroupNorm (not BatchNorm): batch statistics differ between observation-window sizes and made eval unstable.
"""
import torch
import torch.nn as nn
from .cwt import MorletCWT, ScaleCompression


def block(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(4, cout), nn.GELU())


class CWTRULNet(nn.Module):
    def __init__(self, T=128, n_ch=3, n_scales=32, compress=4, width=24, emb=64, dropout=0.1, w0=6.0,
                 s_min=1.5, s_max=40.0):
        super().__init__()
        self.cwt = MorletCWT(T, n_scales, w0, s_min, s_max)
        self.comp = ScaleCompression(compress)
        self.norm_in = nn.GroupNorm(1, n_ch, affine=False)   # per-sample norm (BatchNorm was unstable across window sizes)
        self.cnn = nn.Sequential(block(n_ch, width, 1), block(width, width, 2),            # [B,w,S'/2,T/2]
                                 block(width, 2 * width, 2), block(2 * width, emb, 2))      # [B,emb,S'/8,T/8]
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(emb, emb), nn.GELU(), nn.Linear(emb, 1))

    def scalogram(self, x):                       # x: [B,N,C,T] -> [B*N,C,S',T]
        B, N, C, T = x.shape
        return self.comp(self.cwt(x.reshape(B * N, C, T)))

    def forward(self, x):                         # x: [B,N,C,T]
        B, N = x.shape[:2]
        s = self.norm_in(self.scalogram(x))                       # [B*N,C,S',T]
        f = self.cnn(s).mean((2, 3)).reshape(B, N, -1)            # [B,N,emb]  global pool per cycle
        return self.head(f.mean(1)).squeeze(-1)                   # [B]  predicted log(cycle life)
