"""WZDrive-style multimodal ego-trajectory predictor (this repo's reconstruction).

Inputs  (all in the ego frame at t=0)
  kin   (B,Th,4)  ego history  [x,y,vx,vy]
  ctrl  (B,Th,3)  driver control [steer, throttle, brake]
  gaze  (B,Th,3)  driver gaze   [yaw, pitch, on_target]
  scene (B,N,8)   work-zone objects [is_cone,is_worker,x,y,vx,vy,valid,pad]
Output  traj (B,K,Tf,2) metres, logits (B,K)

Core custom layer: GazeConditionedSceneAttention - a driver-state query attends over scene
tokens with an additive Gaussian bias on the angle between the driver's gaze ray and each
object's bearing, so objects the driver is *looking at* get more attention.
"""
import math
import torch
import torch.nn as nn

POS, VEL = 20.0, 10.0


class GazeConditionedSceneAttention(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.log_sigma = nn.Parameter(torch.full((heads,), math.log(0.15)))   # gaze beam width (rad), per head
        self.gate = nn.Parameter(torch.ones(heads))                           # strength of the gaze prior, per head

    def forward(self, query, scene_tok, bearing, gaze_yaw, valid, use_gaze=True):
        """query (B,d) driver state; scene_tok (B,N,d); bearing (B,N) rad; gaze_yaw (B,) rad; valid (B,N) bool."""
        B, N, d = scene_tok.shape
        q = self.q(query).view(B, 1, self.h, self.dk).transpose(1, 2)            # (B,h,1,dk)
        k = self.k(scene_tok).view(B, N, self.h, self.dk).transpose(1, 2)        # (B,h,N,dk)
        v = self.v(scene_tok).view(B, N, self.h, self.dk).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)                  # (B,h,1,N)
        if use_gaze:
            dtheta = (bearing - gaze_yaw[:, None])[:, None, None, :]             # (B,1,1,N)
            sigma = self.log_sigma.exp().view(1, -1, 1, 1)
            logits = logits - self.gate.view(1, -1, 1, 1) * dtheta ** 2 / (2 * sigma ** 2)
        logits = logits.masked_fill(~valid[:, None, None, :], float("-inf"))
        attn = logits.softmax(-1)
        ctx = (attn @ v).transpose(1, 2).reshape(B, d)
        return self.o(ctx), attn.mean(1).squeeze(1)                              # (B,d), (B,N)


class WZDrive(nn.Module):
    def __init__(self, d_model=64, n_heads=4, n_modes=6, n_dec_layers=2, dropout=0.1,
                 hist=8, fut=12, use_gaze=True, use_control=True):
        super().__init__()
        self.K, self.Tf, self.d = n_modes, fut, d_model
        self.use_gaze, self.use_control = use_gaze, use_control
        self.kin_enc = nn.GRU(4, d_model, batch_first=True)
        self.ctrl_enc = nn.GRU(3, d_model, batch_first=True)
        self.gaze_enc = nn.GRU(3, d_model, batch_first=True)
        self.scene_enc = nn.Sequential(nn.Linear(7, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.driver_fuse = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.ReLU())
        self.gca = GazeConditionedSceneAttention(d_model, n_heads)
        self.type_emb = nn.Parameter(torch.randn(4, d_model) * 0.02)           # kin, ctrl, gaze, gca-ctx
        self.mode_q = nn.Parameter(torch.randn(n_modes, d_model) * 0.5)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, n_dec_layers)
        self.traj_head = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.ReLU(), nn.Linear(2 * d_model, fut * 2))
        self.cls_head = nn.Linear(d_model, 1)

    def forward(self, kin, ctrl, gaze, scene, return_attn=False):
        B = kin.shape[0]
        kin_n = torch.cat([kin[..., :2] / POS, kin[..., 2:] / VEL], -1)
        if not self.use_control: ctrl = torch.zeros_like(ctrl)
        if not self.use_gaze: gaze = torch.zeros_like(gaze)
        gaze = gaze * gaze.new_tensor([5.0, 10.0, 1.0])                         # rad -> ~unit scale (raw ~0.2 rad was too small for the GRU)
        e_kin = self.kin_enc(kin_n)[1][0]                                       # (B,d)  final hidden
        e_ctrl = self.ctrl_enc(ctrl)[1][0]
        e_gaze = self.gaze_enc(gaze)[1][0]
        valid = scene[..., 6] > 0.5                                             # (B,N)
        s_in = torch.stack([scene[..., 0], scene[..., 1], scene[..., 2] / 30.0, scene[..., 3] / 5.0,
                            scene[..., 4] / 3.0, scene[..., 5] / 3.0, scene[..., 6]], -1)
        s_tok = self.scene_enc(s_in)                                            # (B,N,d)
        bearing = torch.atan2(scene[..., 3], scene[..., 2].clamp(min=1.0))      # (B,N)
        gaze_yaw = gaze[:, -3:, 0].mean(1) / 5.0                                # smoothed latest gaze direction, back in rad
        driver = self.driver_fuse(torch.cat([e_kin, e_ctrl, e_gaze], -1))
        ctx, attn = self.gca(driver, s_tok, bearing, gaze_yaw, valid, use_gaze=self.use_gaze)
        mem = torch.cat([(e_kin + self.type_emb[0])[:, None], (e_ctrl + self.type_emb[1])[:, None],
                         (e_gaze + self.type_emb[2])[:, None], (ctx + self.type_emb[3])[:, None], s_tok], 1)
        pad = torch.cat([torch.zeros(B, 4, dtype=torch.bool, device=kin.device), ~valid], 1)
        h = self.dec(self.mode_q[None].expand(B, -1, -1) + driver[:, None], mem, memory_key_padding_mask=pad)  # (B,K,d)
        inc = self.traj_head(h).view(B, self.K, self.Tf, 2)                     # per-step increments (normalised)
        traj = inc.cumsum(2) * POS                                              # metres, cumulative -> smooth paths
        logits = self.cls_head(h).squeeze(-1)                                   # (B,K)
        return (traj, logits, attn) if return_attn else (traj, logits)
