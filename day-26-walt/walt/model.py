"""WALT-style reconstruction: trajectory latent space aligned to a frozen world model.
Paper-sourced (abstract-level): dual-branch trajectory autoencoder; frozen pretrained driving world model;
JEPA / REPA-style alignment. Everything else (dims, losses, planner) is this repo's own reconstruction."""
import torch, torch.nn as nn, torch.nn.functional as F
from .data import T

class FrozenWorldModel(nn.Module):
    """Stand-in 'visual world model': encodes BEV -> tokens, predicts FUTURE BEV. Pretrained then frozen."""
    def __init__(self, d=32):
        super().__init__()
        self.d = d
        self.enc = nn.Sequential(nn.Conv2d(2, 16, 3, 2, 1), nn.GELU(), nn.Conv2d(16, d, 3, 2, 1), nn.GELU())  # (B,d,8,8)
        self.pos = nn.Parameter(torch.randn(1, 64, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, 4, 64, 0.0, batch_first=True, norm_first=True)
        self.dyn = nn.TransformerEncoder(layer, 2)          # latent dynamics: current tokens -> future tokens
        self.dec = nn.Sequential(nn.ConvTranspose2d(d, 16, 4, 2, 1), nn.GELU(), nn.ConvTranspose2d(16, 2, 4, 2, 1))
    def tokens(self, x):                                     # (B,2,32,32) -> (B,64,d)
        return self.enc(x).flatten(2).transpose(1, 2) + self.pos
    def future_tokens(self, tok): return self.dyn(tok)      # (B,64,d)
    def decode(self, tok):
        B = tok.shape[0]
        return self.dec(tok.transpose(1, 2).reshape(B, self.d, 8, 8))
    def forward(self, x):
        return self.decode(self.future_tokens(self.tokens(x)))

class TrajAutoencoder(nn.Module):
    """Dual-branch trajectory AE.
    geometry branch: waypoints -> latent tokens (pure kinematics).
    semantic branch: waypoint-derived queries cross-attend to FROZEN world-model future tokens (scene cues).
    z = fuse(geo, sem): (B, L, dz). decoder(z) -> waypoints (B,T,2)."""
    def __init__(self, dz=32, L=4, d_wm=32, use_semantic=True):
        super().__init__()
        self.L, self.dz, self.use_semantic = L, dz, use_semantic
        self.geo = nn.Sequential(nn.Linear(T * 2, 128), nn.GELU(), nn.Linear(128, L * dz))
        self.q = nn.Sequential(nn.Linear(T * 2, 128), nn.GELU(), nn.Linear(128, L * dz))
        self.kv = nn.Linear(d_wm, dz)
        self.xattn = nn.MultiheadAttention(dz, 4, batch_first=True)
        self.fuse = nn.Linear(2 * dz, dz)
        self.dec = nn.Sequential(nn.Linear(L * dz, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, T * 2))
        self.proj = nn.Linear(dz, d_wm)                      # REPA projector: latent -> WM feature space
    def encode(self, wp, wm_future_tokens):
        B = wp.shape[0]
        g = self.geo(wp.flatten(1)).view(B, self.L, self.dz)
        if not self.use_semantic: return g
        q = self.q(wp.flatten(1)).view(B, self.L, self.dz)
        s, _ = self.xattn(q, self.kv(wm_future_tokens), self.kv(wm_future_tokens))   # (B,L,dz)
        return self.fuse(torch.cat([g, s], -1))
    def decode(self, z): return self.dec(z.flatten(1)).view(-1, T, 2)
    def align_loss(self, z, wm_future_tokens):
        """REPA-style: cosine alignment of projected latent tokens to (stop-grad) pooled WM future feature."""
        target = wm_future_tokens.detach().mean(1, keepdim=True)          # (B,1,d_wm) scene-level cue
        return 1 - F.cosine_similarity(self.proj(z), target.expand(-1, z.shape[1], -1), dim=-1).mean()

class LatentPlanner(nn.Module):
    """Scene tokens (frozen WM) -> learned queries -> transformer decoder -> output_dim per query.
    Same architecture is used for BOTH old way (raw waypoints out) and new way (latent z out)."""
    def __init__(self, d_wm=32, out_tokens=4, out_dim=32, d=64):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_wm), nn.Linear(d_wm, d))  # LN: frozen-WM tokens have small variance
        self.queries = nn.Parameter(torch.randn(1, out_tokens, d) * 0.02)
        self.mem_pos = nn.Parameter(torch.randn(1, 64, d) * 0.5)   # absolute position of each WM token (8x8 grid)
        layer = nn.TransformerDecoderLayer(d, 4, 128, 0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerDecoder(layer, 2)
        self.out = nn.Linear(d, out_dim)
    def forward(self, tok):
        m = self.inp(tok) + self.mem_pos
        return self.out(self.tr(self.queries.expand(tok.shape[0], -1, -1), m))   # (B,out_tokens,out_dim)

class WALTPlanner(nn.Module):
    """New way: planner outputs latent z, frozen AE decoder maps to waypoints."""
    def __init__(self, wm, ae):
        super().__init__()
        self.wm, self.ae = wm, ae
        self.planner = LatentPlanner(out_tokens=ae.L, out_dim=ae.dz)
    def forward(self, x):
        with torch.no_grad(): tok = self.wm.tokens(x)
        z = self.planner(tok)
        return self.ae.decode(z), z

class DirectPlanner(nn.Module):
    """Old way: identical planner, regresses raw waypoints (T tokens x 2)."""
    def __init__(self, wm):
        super().__init__()
        self.wm = wm
        self.planner = LatentPlanner(out_tokens=T, out_dim=2)
    def forward(self, x):
        with torch.no_grad(): tok = self.wm.tokens(x)
        return self.planner(tok)
