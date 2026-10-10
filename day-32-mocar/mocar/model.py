"""MoCAR-style reconstruction: continuous motion-code tokenizer + decoder-only autoregressive code predictor.

Paper-level idea (alphaxiv summary of arXiv:2610.06210): forecast trajectories as sequential prediction of
*continuous motion codes*; each code encodes local segment geometry AND the reference-frame transition the
segment induces; the predictor feeds its own predicted codes back (latent memory, no trajectory-space
re-tokenization). Every dimension below is this repo's own reconstruction default (see SOURCING.md).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

SEG = 5  # steps per segment


def rot(theta):  # theta (...,) -> (..., 2, 2) rotation matrices
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def wrap(a): return torch.atan2(torch.sin(a), torch.cos(a))


def segment(points, headings):
    """Split a polyline into endpoint-sharing segments, each expressed in its own start frame.
    points (B, J*SEG+1, 2), headings (B, J*SEG+1) -> local (B, J, SEG, 2) [excludes the start pt], dtheta (B, J),
    start poses (B, J, 3) = (x, y, theta)."""
    B, N, _ = points.shape
    J = (N - 1) // SEG
    idx = torch.arange(J, device=points.device) * SEG
    start = points[:, idx]                                                  # (B,J,2)
    th0 = headings[:, idx]                                                  # (B,J)
    win = idx[:, None] + torch.arange(1, SEG + 1, device=points.device)[None]  # (J,SEG)
    seg = points[:, win] - start[:, :, None]                                # (B,J,SEG,2) world offsets
    local = torch.einsum("bjsc,bjcd->bjsd", seg, rot(th0))                  # R(-th0) v = v^T R(th0)
    dth = wrap(headings[:, idx + SEG] - th0)
    return local, dth, torch.cat([start, th0[..., None]], -1)


def compose(local, dth, pose0):
    """Chain per-segment local geometry from an initial pose (B,3) -> world points (B,J*SEG,2),
    start poses (B,J,3), end poses (B,J,3). Fully differentiable."""
    B, J = dth.shape
    pos, th = pose0[:, :2], pose0[:, 2]
    pts, starts, ends = [], [], []
    for j in range(J):
        starts.append(torch.cat([pos, th[:, None]], -1))
        R = rot(th)                                                         # (B,2,2)
        pts.append(pos[:, None] + torch.einsum("bsc,bdc->bsd", local[:, j], R))
        pos = pos + torch.einsum("bc,bdc->bd", local[:, j, -1], R)
        th = th + dth[:, j]
        ends.append(torch.cat([pos, th[:, None]], -1))
    return torch.cat(pts, 1), torch.stack(starts, 1), torch.stack(ends, 1)


class MotionCodeTokenizer(nn.Module):
    """VAE-style continuous tokenizer. Segment geometry + frame transition <-> code z in R^{code_dim}."""
    def __init__(self, code_dim=16, hidden=96):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(SEG * 2 + 2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.mu, self.logvar = nn.Linear(hidden, code_dim), nn.Linear(hidden, code_dim)
        self.dec = nn.Sequential(nn.Linear(code_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, SEG * 2 + 1))

    def encode(self, local, dth):  # (B,J,SEG,2),(B,J) -> mu, logvar (B,J,D)
        x = torch.cat([local.flatten(-2), torch.sin(dth)[..., None], torch.cos(dth)[..., None]], -1)
        h = self.enc(x)
        return self.mu(h), self.logvar(h).clamp(-8, 4)

    def decode(self, z):  # (B,J,D) -> local (B,J,SEG,2), dth (B,J)
        o = self.dec(z)
        return o[..., :SEG * 2].reshape(*z.shape[:-1], SEG, 2), o[..., -1]


class PoseEmbed(nn.Module):
    """'Coordinate-aware': embeds the reference frame (x, y, sin, cos) a code ends in."""
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
    def forward(self, pose):  # (...,3)
        return self.net(torch.stack([pose[..., 0] / 20, pose[..., 1] / 20, torch.sin(pose[..., 2]), torch.cos(pose[..., 2])], -1))


class SceneEncoder(nn.Module):
    """Lane polylines (PointNet-style) -> tokens. Output (B, N_lanes, d) and padding mask."""
    def __init__(self, d):
        super().__init__()
        self.pt = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
    def forward(self, lanes, lane_mask):
        h = self.pt(lanes / 20.0)                                           # (B,L,P,d)
        tok = self.out(torch.cat([h.max(2).values, h.mean(2)], -1))
        return tok, lane_mask < 0.5


class MoCAR(nn.Module):
    def __init__(self, code_dim=16, d_model=64, n_heads=4, n_layers=2, n_modes=3, hist_segs=4, fut_segs=6,
                 coord_aware=True, kl_beta=1e-4):
        super().__init__()
        self.D, self.M, self.Jh, self.Jf, self.coord_aware, self.kl_beta = code_dim, n_modes, hist_segs, fut_segs, coord_aware, kl_beta
        self.tok = MotionCodeTokenizer(code_dim)
        self.scene = SceneEncoder(d_model)
        self.code_in = nn.Linear(code_dim, d_model)
        self.pose_emb = PoseEmbed(d_model)
        self.pos_emb = nn.Parameter(torch.randn(hist_segs + fut_segs, d_model) * 0.02)
        self.mode_emb = nn.Parameter(torch.randn(n_modes, d_model) * 0.5)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, 0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, n_layers)
        self.next_code = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, code_dim))
        self.score = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    # ---- helpers -------------------------------------------------------------------------------
    def _tokens(self, codes, end_poses, mode):  # codes (BM,K,D), end_poses (BM,K,3) -> (BM,K,d)
        K = codes.shape[1]
        x = self.code_in(codes) + self.pos_emb[:K][None] + self.mode_emb[mode][:, None]
        return x + self.pose_emb(end_poses) if self.coord_aware else x

    def _run(self, x, mem, mem_pad):
        K = x.shape[1]
        causal = torch.triu(torch.ones(K, K, dtype=torch.bool, device=x.device), 1)
        return self.dec(x, mem, tgt_mask=causal, memory_key_padding_mask=mem_pad)

    def tokenize(self, hist_pts, hist_hd, fut_pts=None, fut_hd=None):
        out = {}
        lh, dh, sh = segment(hist_pts, hist_hd)
        out["h"] = (lh, dh, sh); mu, lv = self.tok.encode(lh, dh); out["h_mu"], out["h_lv"] = mu, lv
        if fut_pts is not None:
            lf, df, sf = segment(fut_pts, fut_hd)
            out["f"] = (lf, df, sf); mu, lv = self.tok.encode(lf, df); out["f_mu"], out["f_lv"] = mu, lv
        return out

    # ---- forward -------------------------------------------------------------------------------
    def forward(self, b, teacher_forcing=False):
        """b: batch dict. Returns dict with per-mode future points (B,M,30,2), scores (B,M), pred codes, etc.
        teacher_forcing=True feeds ground-truth future codes/poses; False = free-running autoregression where each
        predicted code (continuous, never re-tokenized from xy) is appended to the latent sequence."""
        hp, hh = b["hist_pts"], b["hist_hd"]
        B, M, Jh, Jf, D = hp.shape[0], self.M, self.Jh, self.Jf, self.D
        tk = self.tokenize(hp, hh, b.get("fut_pts"), b.get("fut_hd")) if teacher_forcing else self.tokenize(hp, hh)
        hist_codes = tk["h_mu"]                                              # (B,Jh,D)
        _, h_starts, h_ends = compose(*self.tok.decode(hist_codes), hist_pts_pose0(hp, hh))
        # the history chain is decoded from codes, but anchored so that its END is the agent frame (origin, 0 heading)
        shift_pose = h_ends[:, -1]                                           # decoded end pose of history chain
        h_ends = _to_frame(h_ends, shift_pose)
        mem, mem_pad = self.scene(b["lanes"], b["lane_mask"])
        mem = mem[:, None].expand(B, M, *mem.shape[1:]).reshape(B * M, *mem.shape[1:])
        mem_pad = mem_pad[:, None].expand(B, M, -1).reshape(B * M, -1)
        mode = torch.arange(M, device=hp.device).repeat(B)
        codes = hist_codes[:, None].expand(B, M, Jh, D).reshape(B * M, Jh, D)
        ends = h_ends[:, None].expand(B, M, Jh, 3).reshape(B * M, Jh, 3)
        origin = torch.zeros(B * M, 3, device=hp.device)
        pred_codes, hid = [], None
        if teacher_forcing:
            fc = tk["f_mu"][:, None].expand(B, M, Jf, D).reshape(B * M, Jf, D)
            f_ends = tk["f"][2]  # start poses of future segs (GT) -> end pose of seg j == start pose of seg j+1
            gt_end = torch.cat([f_ends[:, 1:], torch.cat([b["fut_pts"][:, -1], b["fut_hd"][:, -1:]], -1)[:, None]], 1)
            gt_end = gt_end[:, None].expand(B, M, Jf, 3).reshape(B * M, Jf, 3)
            seq = torch.cat([codes, fc[:, :-1]], 1); seq_e = torch.cat([ends, gt_end[:, :-1]], 1)
            h = self._run(self._tokens(seq, seq_e, mode), mem, mem_pad)
            hid = h[:, Jh - 1:]
            pred = self.next_code(hid)                                       # (BM,Jf,D)
        else:
            cur_c, cur_e, preds = codes, ends, []
            for j in range(Jf):
                h = self._run(self._tokens(cur_c, cur_e, mode), mem, mem_pad)
                c_next = self.next_code(h[:, -1])                            # (BM,D) latent code, fed back as-is
                preds.append(c_next)
                lp, dp = self.tok.decode(torch.stack(preds, 1))
                _, _, e_all = compose(lp, dp, origin)                        # coordinate chain from *decoded* codes
                cur_c = torch.cat([codes, torch.stack(preds, 1)], 1)
                cur_e = torch.cat([ends, e_all], 1)
            hid = h[:, Jh - 1:]                                              # hidden states of the Jf future positions
            pred = torch.stack(preds, 1)
        lp, dp = self.tok.decode(pred)
        pts, _, _ = compose(lp, dp, origin)                                  # (BM,Jf*SEG,2)
        pts = pts.reshape(B, M, Jf * SEG, 2)
        sc = self.score(hid.mean(1)).reshape(B, M)
        return dict(pts=pts, scores=sc, pred_codes=pred.reshape(B, M, Jf, D), tk=tk)

    # ---- loss ----------------------------------------------------------------------------------
    def loss(self, b):
        tf = self.forward(b, teacher_forcing=True)
        fr = self.forward(b, teacher_forcing=False)
        tk = tf["tk"]
        # tokenizer reconstruction (+ weak KL) on history and future segments
        rec, kl = 0.0, 0.0
        for k in ("h", "f"):
            loc, dth, _ = tk[k]
            mu, lv = tk[k + "_mu"], tk[k + "_lv"]
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
            dl, dd = self.tok.decode(z)
            rec = rec + F.mse_loss(dl, loc) + F.mse_loss(dd, dth)
            kl = kl + (-0.5 * (1 + lv - mu ** 2 - lv.exp())).mean()
        gt = b["fut_pts"][:, 1:]
        total = rec + self.kl_beta * kl
        stats = {"rec": rec.item(), "kl": kl.item()}
        for name, out, w in (("tf", tf, 0.5), ("fr", fr, 1.0)):
            ade = (out["pts"] - gt[:, None]).norm(dim=-1).mean(-1)           # (B,M)
            best = ade.argmin(1)
            ar = torch.arange(len(best))
            l_wta = F.smooth_l1_loss(out["pts"][ar, best], gt, beta=0.5) + 0.05 * F.smooth_l1_loss(out["pts"], gt[:, None].expand_as(out["pts"]), beta=0.5)
            l_cls = F.cross_entropy(out["scores"], best)
            # latent alignment: predicted code should land on the tokenizer's code for the true next segment
            l_align = F.mse_loss(out["pred_codes"][ar, best], tk["f_mu"].detach())
            total = total + w * (l_wta + 0.3 * l_cls + 0.1 * l_align)
            stats[name + "_ade"] = ade[ar, best].mean().item()
        return total, stats


def hist_pts_pose0(hp, hh):
    """Pose of the start of the history chain (used only to anchor the decoded history chain; real frame = agent at t0)."""
    return torch.cat([hp[:, 0], hh[:, :1]], -1)


def _to_frame(poses, ref):
    """Express poses (B,K,3) in the frame of `ref` (B,3)."""
    d = poses[..., :2] - ref[:, None, :2]
    loc = torch.einsum("bkc,bcd->bkd", d, rot(ref[:, 2]))
    return torch.cat([loc, wrap(poses[..., 2] - ref[:, None, 2])[..., None]], -1)


class OneShotBaseline(nn.Module):
    """'Old way': same scene encoder, but all M futures are regressed at once from learned queries -- no codes,
    no autoregression. Mirrors a MultiPath/DenseTNT-style one-shot multi-modal regressor."""
    def __init__(self, d_model=64, n_heads=4, n_layers=2, n_modes=3, fut_steps=30):
        super().__init__()
        self.M, self.T = n_modes, fut_steps
        self.scene = SceneEncoder(d_model)
        self.hist = nn.Sequential(nn.Linear(21 * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.q = nn.Parameter(torch.randn(n_modes, d_model) * 0.5)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, 4 * d_model, 0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, fut_steps * 2))
        self.score = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, b, **_):
        mem, pad = self.scene(b["lanes"], b["lane_mask"])
        hist = self.hist(b["hist_pts"].flatten(1) / 20.0)[:, None]
        mem = torch.cat([mem, hist], 1); pad = torch.cat([pad, torch.zeros_like(pad[:, :1])], 1)
        h = self.dec(self.q[None].expand(len(mem), -1, -1), mem, memory_key_padding_mask=pad)
        pts = self.head(h).reshape(len(mem), self.M, self.T, 2).cumsum(2)       # cumulative offsets -> smooth paths
        return dict(pts=pts, scores=self.score(h).squeeze(-1))

    def loss(self, b):
        out = self(b); gt = b["fut_pts"][:, 1:]
        ade = (out["pts"] - gt[:, None]).norm(dim=-1).mean(-1); best = ade.argmin(1); ar = torch.arange(len(best))
        l = F.smooth_l1_loss(out["pts"][ar, best], gt, beta=0.5) + 0.05 * F.smooth_l1_loss(out["pts"], gt[:, None].expand_as(out["pts"]), beta=0.5) + 0.3 * F.cross_entropy(out["scores"], best)
        return l, {"ade": ade[ar, best].mean().item()}
