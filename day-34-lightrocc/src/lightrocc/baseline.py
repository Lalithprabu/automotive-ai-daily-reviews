"""Dense-BEV occupancy forecaster ("old way"): 16x16 learned BEV queries cross-attend to the same image tokens,
then a conv decoder upsamples to the 32x32 grid and regresses (T+1)*Z voxel probabilities directly."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ImageEncoder


class DenseBEVOccForecaster(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m, w = cfg["model"], cfg["world"]
        d, self.nb = m["d_model"], m["bev_queries"]
        self.T1, self.Z = w["t_future"] + 1, w["grid_z"]
        self.encoder = ImageEncoder(cfg)
        self.bev = nn.Parameter(torch.randn(1, self.nb * self.nb, d) * 0.02)
        layer = nn.TransformerDecoderLayer(d, m["n_heads"], m["ff"], 0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, m["dec_layers"])
        self.up = nn.Sequential(
            nn.ConvTranspose2d(d, 48, 2, stride=2), nn.GroupNorm(4, 48), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.GroupNorm(4, 48), nn.GELU(),
            nn.Conv2d(48, self.T1 * self.Z, 1))
        nn.init.constant_(self.up[-1].bias, -3.0)

    def forward(self, img, **_):
        B = img.shape[0]
        mem = self.encoder(img)
        b = self.decoder(self.bev.expand(B, -1, -1), mem)                 # [B,256,d]
        b = b.transpose(1, 2).reshape(B, -1, self.nb, self.nb)            # [B,d,16,16]
        logits = self.up(b)                                               # [B,T1*Z,32,32]  (axes: x, y)
        X, Y = logits.shape[-2:]
        p = torch.sigmoid(logits).reshape(B, self.T1, self.Z, X, Y).permute(0, 1, 3, 4, 2)  # [B,T1,X,Y,Z]
        return dict(probs=p.reshape(B, self.T1, -1))                       # same [B,T1,V] layout as LighTROcc
