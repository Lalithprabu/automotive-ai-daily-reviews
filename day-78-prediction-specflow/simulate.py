"""Real-time synthetic "AI in action" simulation: SpecFlow forecasting every agent at an intersection.

Each clip scene plays in four acts:
    1. OBSERVE  (3.2 s) tracked boxes stream in with their 8-frame history (grey trails, ID + speed tags)
    2. DENOISE  10 Euler steps of the flow: 20 hypotheses per agent move from Gaussian noise in the ACCELERATION
                SPECTRUM to structured futures; the right panel shows the DCT coefficients settling
    3. PLAY     (4.8 s) the world moves on along the ground truth (green) while the predicted fans (blue) stay frozen
    4. HOLD     minADE / minFDE per agent
Scene 1 = the test scene whose SpecFlow minADE is closest to the median (>= 4 agents, at least one turning vehicle).
Scene 2 = a scene where top-1 endpoint conditioning picks the wrong maneuver (the failure mode), shown against the
confidence-stratified endpoints (magenta).

Layout 1280 x 720:  left 820 x 640 BEV  |  right 448 px: acceleration spectrum, speed/accel telemetry (focus agent,
SpecFlow vs position-state flow), held-out study table  |  bottom strip: legend + SYNTHETIC disclaimer

    python simulate.py --out assets/specflow_simulation.gif --mp4 assets/specflow_simulation.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from dsf.data import DT, MANEUVERS, PED, T_OBS, VEH, local_to_world
from evaluate import load_model
from train import load_data, to_batch

ROOT = Path(__file__).resolve().parent
TXT, GRID, DIM = (235, 235, 235), (52, 52, 52), (150, 150, 150)
C = dict(gt=(90, 220, 90), sf=(255, 170, 50), pos=(70, 70, 235), strat=(220, 80, 220), obs=(185, 185, 185),
         ep=(80, 230, 255), road=(58, 58, 58))                                           # BGR
SUB = 4                                                                                   # render frames per 0.4 s


def put(img, txt, org, scale=0.45, col=TXT, th=1):
    cv2.putText(img, txt, (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def poly(img, pts, col, th=1, aa=True):
    pts = np.asarray(pts)
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) > 1:
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, col, th, cv2.LINE_AA if aa else 8)


def dashed(img, pts, col, th=1, seg=6):
    pts = np.asarray(pts, float)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        n = max(1, int(np.linalg.norm(b - a) / seg))
        for j in range(0, n, 2):
            u, v = a + (b - a) * j / n, a + (b - a) * min(n, j + 1) / n
            cv2.line(img, tuple(np.round(u).astype(int)), tuple(np.round(v).astype(int)), col, th, cv2.LINE_AA)


class View:
    """World metres -> BEV pixels (y up)."""

    def __init__(self, pts: np.ndarray, W: int, H: int, pad: float = 6.0):
        lo, hi = pts.min(0) - pad, pts.max(0) + pad
        ctr = (lo + hi) / 2
        span = max((hi - lo)[0] / W, (hi - lo)[1] / H)
        self.s = 1.0 / span
        self.ctr, self.W, self.H = ctr, W, H

    def __call__(self, p):
        p = np.asarray(p, float)
        x = (p[..., 0] - self.ctr[0]) * self.s + self.W / 2
        y = self.H / 2 - (p[..., 1] - self.ctr[1]) * self.s
        return np.stack([x, y], -1)


def draw_road(img, V: View):
    """Four 14 m-wide arms, lane lines, stop lines at |coord| = 9, crosswalks."""
    for k in range(2):
        a = np.array([[-200, -7], [200, -7], [200, 7], [-200, 7]], float)
        if k:
            a = a[:, ::-1]
        cv2.fillPoly(img, [np.round(V(a)).astype(np.int32)], C["road"])
    for arm in range(4):
        R = np.array([[np.cos(arm * np.pi / 2), -np.sin(arm * np.pi / 2)], [np.sin(arm * np.pi / 2), np.cos(arm * np.pi / 2)]])
        dashed(img, V(np.array([[-200, -3.5], [-9, -3.5]]) @ R.T), (120, 120, 120), 1, 8)
        dashed(img, V(np.array([[-200, 3.5], [-9, 3.5]]) @ R.T), (120, 120, 120), 1, 8)
        poly(img, V(np.array([[-200, 0], [-9, 0]]) @ R.T), (60, 160, 200), 1)
        poly(img, V(np.array([[-9, -7], [-9, 0]]) @ R.T), (220, 220, 220), 2)
        for yy in np.arange(-6.5, 7, 1.3):
            q = V(np.array([[-11.5, yy], [-10.0, yy]]) @ R.T)
            poly(img, q, (95, 95, 95), max(1, int(V.s * 0.5)))


def box(img, V, p, h, col, kind, th=2):
    if kind == VEH:
        L, Wd = 4.6, 1.9
        c, s = np.cos(h), np.sin(h)
        corners = np.array([[L / 2, Wd / 2], [L / 2, -Wd / 2], [-L / 2, -Wd / 2], [-L / 2, Wd / 2]]) @ np.array([[c, s], [-s, c]]) + p
        q = np.round(V(corners)).astype(np.int32)
        cv2.polylines(img, [q], True, col, th, cv2.LINE_AA)
        nose = V(p + 2.3 * np.array([c, s]))
        cv2.circle(img, tuple(np.round(nose).astype(int)), 2, col, -1)
    else:
        cv2.circle(img, tuple(np.round(V(p)).astype(int)), max(3, int(0.5 * V.s)), col, th, cv2.LINE_AA)


def heading_of(track, i):
    j0, j1 = max(0, i - 2), min(len(track) - 1, i + 1)
    d = track[j1] - track[j0]
    return np.arctan2(d[1], d[0]) if np.linalg.norm(d) > 0.3 else None


def scene_preds(models, d, si, ep_mode="top1", keep=False, seed=7):
    b = to_batch(d, np.array([si]))
    g = torch.Generator().manual_seed(seed)
    return models.sample(b, H=20, steps=10, ep_mode=ep_mode, keep_path=keep, generator=g)


def world(p_local, d, si, a):
    return local_to_world(p_local, d["origin"][si, a], d["psi"][si, a])


def render_scene(si, d, mf, mp, S, label, frames, show_strat=False, focus=None):
    mask = d["mask"][si]
    A = np.nonzero(mask)[0]
    kind = d["kind"][si]
    clean = d["clean"][si]                                       # (A, 20, 2) world
    meas = d["pos"][si]
    o = scene_preds(mf, d, si, keep=True)
    op = scene_preds(mp, d, si)
    os_ = scene_preds(mf, d, si, ep_mode="sampled") if show_strat else None
    P = o["p"][0].numpy()                                        # (H, A, T, 2) local
    Pp = op["p"][0].numpy()
    path = o["path"][:, 0].numpy()                               # (11, H, A, 24) normalised state
    ep = o["endpoints"][0].numpy(); lg = o["logits"][0].softmax(-1).numpy()
    gtl = d["fut_clean"][si]
    err = np.linalg.norm(P - gtl[None], axis=-1)                 # (H, A, T)
    ade, fde = err.mean(-1).min(0), err[..., -1].min(0)
    if show_strat:
        err_s = np.linalg.norm(os_["p"][0].numpy() - gtl[None], axis=-1)
        ade_s, fde_s = err_s.mean(-1).min(0), err_s[..., -1].min(0)
    if focus is None:
        turners = [a for a in A if d["man"][si, a] in (1, 2)]
        focus = turners[0] if turners else A[0]
    Pw = np.stack([world(P[:, a], d, si, a) for a in range(len(mask))], 1)       # (H, A, T, 2)
    Ppw = np.stack([world(Pp[:, a], d, si, a) for a in range(len(mask))], 1)
    Psw = np.stack([world(os_["p"][0, :, a].numpy(), d, si, a) for a in range(len(mask))], 1) if show_strat else None
    epw = np.stack([world(ep[a], d, si, a) for a in range(len(mask))])          # (A, K, 2)
    # intermediate denoising states -> trajectories (fixed integration of each partial state)
    v0 = torch.as_tensor(d["v0"][si])[None].expand(20, -1, -1)
    traj_steps = []
    for k in range(path.shape[0]):
        _, _, pk = mf.kinematics(torch.as_tensor(path[k]), v0)
        pk = pk.numpy()
        traj_steps.append(np.stack([world(pk[:, a], d, si, a) for a in range(len(mask))], 1))
    pts = np.concatenate([clean[A].reshape(-1, 2), Pw[:, A].reshape(-1, 2)])
    pts = np.clip(pts, -42, 42)                                 # keep the intersection in frame
    pts = np.concatenate([pts, [[-25, -25], [25, 25]]])          # at least a 50 m x 50 m window
    V = View(pts, 812, 600, pad=4.0)
    # focus-agent telemetry
    def kin_series(Pl):                                         # local (H, T, 2) -> speed, accel (H, T)
        v0f = d["v0"][si, focus]
        pp = np.concatenate([np.zeros((Pl.shape[0], 1, 2)), Pl], 1)
        vv = np.diff(pp, axis=1) / DT
        sp = np.linalg.norm(vv, axis=-1)
        acc = np.diff(np.concatenate([np.repeat(v0f[None, None], Pl.shape[0], 0), vv], 1), axis=1) / DT
        return sp, np.linalg.norm(acc, axis=-1) * np.sign((acc * vv).sum(-1) + 1e-9)
    sp_sf, ac_sf = kin_series(P[:, focus])
    sp_ps, ac_ps = kin_series(Pp[:, focus])
    sp_gt, ac_gt = kin_series(gtl[None, focus])
    n_obs, n_den, n_play, n_hold = (T_OBS - 1) * SUB + 1, 10 * 3, 12 * SUB, 18
    total = n_obs + n_den + n_play + n_hold
    C_dct = mf.kin.C.numpy()
    for fr in range(total):
        img = np.full((720, 1280, 3), 22, np.uint8)
        bev = np.full((600, 812, 3), 34, np.uint8)
        draw_road(bev, V)
        if fr < n_obs:
            phase, tt = "OBSERVE", fr / SUB
        elif fr < n_obs + n_den:
            phase, tt = "DENOISE", T_OBS - 1
        elif fr < n_obs + n_den + n_play:
            phase, tt = "PLAY", T_OBS - 1 + (fr - n_obs - n_den + 1) / SUB
        else:
            phase, tt = "HOLD", T_OBS - 1 + 12
        dstep = min(10, (fr - n_obs) // 3 + 1) if phase == "DENOISE" else 10
        show_pred = phase != "OBSERVE"
        for a in A:
            col_a = C["sf"]
            # ground-truth future (dim until PLAY, then revealed up to the current time)
            g = np.concatenate([clean[a, T_OBS - 1:T_OBS], clean[a, T_OBS:]])
            if phase in ("PLAY", "HOLD"):
                dashed(bev, V(g), C["gt"], 2)
            elif show_pred:
                dashed(bev, V(g), (55, 105, 55), 1)
            if show_pred:
                Tset = traj_steps[dstep] if phase == "DENOISE" else Pw
                origin = clean[a, T_OBS - 1:T_OBS] * 0 + meas[a, T_OBS - 1]
                for h in range(20):
                    poly(bev, V(np.concatenate([origin, Tset[h, a]])), col_a, 1)
                if show_strat and a == focus and phase != "DENOISE":
                    for h in range(20):
                        poly(bev, V(np.concatenate([origin, Psw[h, a]])), C["strat"], 1)
                if a == focus and phase in ("PLAY", "HOLD"):
                    for h in range(0, 20, 4):
                        dashed(bev, V(np.concatenate([origin, Ppw[h, a]])), C["pos"], 1, 4)
                if kind[a] == VEH and d["man"][si, a] >= 0:
                    top = int(lg[a].argmax())
                    for k in range(ep.shape[1]):
                        if lg[a, k] < 0.08 and k != top:
                            continue
                        q = np.round(V(epw[a, k])).astype(int)
                        r = int(4 + 8 * lg[a, k])
                        cv2.drawMarker(bev, tuple(q), C["ep"], cv2.MARKER_STAR if k == top else cv2.MARKER_DIAMOND,
                                       int(r * 1.4) if k == top else r, 2 if k == top else 1, cv2.LINE_AA)
            # history trail + box at the current time
            ti = min(tt, T_OBS - 1) if phase in ("OBSERVE", "DENOISE") else tt
            i0 = int(np.floor(ti)); fr_ = ti - i0
            cur = clean[a, i0] if i0 >= 19 else (1 - fr_) * clean[a, i0] + fr_ * clean[a, min(19, i0 + 1)]
            hist_upto = min(int(np.floor(min(ti, T_OBS - 1))), T_OBS - 1)
            for j in range(hist_upto + 1):
                cv2.circle(bev, tuple(np.round(V(meas[a, j])).astype(int)), 2, C["obs"], -1)
            poly(bev, V(meas[a, :hist_upto + 1]), (120, 120, 120), 1)
            h_ = heading_of(clean[a], min(19, int(round(ti))))
            h_ = h_ if h_ is not None else (heading_of(clean[a], T_OBS - 1) or 0.0)
            col_box = (255, 255, 255) if a == focus else (210, 210, 210)
            box(bev, V, cur, h_, col_box, kind[a], 2)
            j1 = min(19, i0 + 1)
            spd = np.linalg.norm(clean[a, j1] - clean[a, j1 - 1]) / DT
            tag = f"#{a} {'car' if kind[a] == VEH else 'ped'} {spd * 3.6:.0f} km/h"
            q = V(cur) + np.array([8, -8])
            put(bev, tag, q, 0.36, col_box)
            if phase == "HOLD":
                put(bev, f"minADE {ade[a]:.2f} minFDE {fde[a]:.2f}", q + np.array([0, 13]), 0.34, C["gt"])
                if show_strat and a == focus:
                    put(bev, f"stratified: minADE {ade_s[a]:.2f} minFDE {fde_s[a]:.2f}", q + np.array([0, 26]), 0.34, C["strat"])
        put(bev, f"{label}   [{phase}]", (8, 18), 0.5, TXT, 1)
        clock = (tt - (T_OBS - 1)) * DT
        put(bev, f"t = {clock:+.1f} s", (8, 40), 0.48, TXT)
        if phase == "DENOISE":
            put(bev, f"flow step {dstep}/10: noise -> acceleration spectrum -> integrate", (8, 62), 0.45, C["sf"])
        if phase == "PLAY":
            put(bev, "model input frozen at t = 0; world moves on along the ground truth", (8, 62), 0.42, DIM)
        img[4:604, 4:816] = bev
        # ================= right column
        R = img[4:610, 826:1276]
        w = 448
        # (1) spectrum
        cv2.rectangle(R, (1, 1), (w - 3, 196), (70, 70, 70), 1)
        put(R, f"acceleration spectrum, agent #{focus} (DCT, along-track)", (6, 16), 0.42)
        psi = d["psi"][si, focus]
        k_use = dstep if show_pred else 0
        st = torch.as_tensor(path[k_use][:, focus]).unflatten(-1, (12, 2)) * mf.sd + mf.mu   # (H, 12, 2) DCT coeffs (local)
        st = st.numpy()
        _, a_gt = mf.kin.differentiate(torch.as_tensor(gtl[focus])[None], torch.as_tensor(d["v0"][si, focus])[None])
        c_gt = (C_dct @ a_gt[0].numpy())                                       # (12, 2)
        base, sc = 120, 22
        cv2.line(R, (20, base), (w - 15, base), (90, 90, 90), 1)
        for k in range(12):
            x0 = 24 + k * 34
            vals = st[:, k, 0]
            lo, hi = np.percentile(vals, 10), np.percentile(vals, 90)
            y_hi, y_lo = int(base - np.clip(hi, -4, 4) * sc), int(base - np.clip(lo, -4, 4) * sc)
            cv2.rectangle(R, (x0, min(y_hi, y_lo - 3)), (x0 + 12, y_lo), C["sf"], -1)
            if show_pred:
                yv = int(base - np.clip(c_gt[k, 0], -4, 4) * sc)
                cv2.line(R, (x0 - 3, yv), (x0 + 15, yv), C["gt"], 2)
            put(R, f"c{k}", (x0 - 2, 188), 0.3, DIM)
        put(R, "bars = 10-90% of 20 hypotheses | green = truth" if show_pred else "waiting for the observation window",
            (6, 34), 0.36, DIM)
        # (2) telemetry
        y2 = 202
        cv2.rectangle(R, (1, y2), (w - 3, y2 + 196), (70, 70, 70), 1)
        put(R, f"agent #{focus} future: speed (top) and accel (bottom)", (6, y2 + 16), 0.42)
        tx = lambda k: 40 + k * (w - 60) / 11
        sy = lambda v: y2 + 110 - np.clip(v, 0, 16) / 16 * 64
        ay = lambda v: y2 + 160 - np.clip(v, -8, 8) / 8 * 30
        cv2.line(R, (40, int(ay(0))), (w - 20, int(ay(0))), (80, 80, 80), 1)
        for v6 in (-6, 6):
            dashed(R, [(40, ay(v6)), (w - 20, ay(v6))], (80, 80, 120), 1)
        put(R, "16 m/s", (2, sy(16) + 4), 0.3, DIM); put(R, "0", (24, sy(0) + 4), 0.3, DIM)
        put(R, "+6", (14, ay(6) + 4), 0.3, DIM); put(R, "-6 m/s2", (2, ay(-6) + 10), 0.3, DIM)
        if show_pred and phase != "DENOISE":
            n_show = 12 if phase == "HOLD" else max(1, int((tt - (T_OBS - 1)) * 1)) if phase == "PLAY" else 12
            n_show = 12
            for h in range(0, 20, 2):
                poly(R, np.stack([tx(np.arange(12)), ay(ac_ps[h])], -1), C["pos"], 1)
                poly(R, np.stack([tx(np.arange(12)), sy(sp_ps[h])], -1), C["pos"], 1)
            for h in range(0, 20, 2):
                poly(R, np.stack([tx(np.arange(12)), ay(ac_sf[h])], -1), C["sf"], 1)
                poly(R, np.stack([tx(np.arange(12)), sy(sp_sf[h])], -1), C["sf"], 1)
            poly(R, np.stack([tx(np.arange(12)), sy(sp_gt[0])], -1), C["gt"], 2)
            poly(R, np.stack([tx(np.arange(12)), ay(ac_gt[0])], -1), C["gt"], 2)
            if phase == "PLAY":
                xx = int(tx(min(11, (tt - (T_OBS - 1)) - 1)))
                cv2.line(R, (xx, y2 + 24), (xx, y2 + 192), (200, 200, 200), 1)
            put(R, "blue = SpecFlow | red = position-state flow | green = truth", (6, y2 + 34), 0.34, DIM)
        # (3) study table
        y3 = 404
        cv2.rectangle(R, (1, y3), (w - 3, 604), (70, 70, 70), 1)
        M = S["metrics"]
        put(R, f"held-out: {S['n_scenes']} scenes, {S['n_agents']} agents, best-of-20", (6, y3 + 16), 0.42)
        put(R, "                   minADE  minFDE   jerk  |a|>6", (6, y3 + 36), 0.37, (170, 170, 170))
        yy = y3 + 54
        rows = (("specflow", "SpecFlow", C["sf"]), ("pos_state", "position state", C["pos"]),
                ("acc_time", "accel, no DCT", (200, 200, 200)), ("no_phase", "no phase loss", (200, 200, 200)),
                ("no_endpoint", "no endpoint", (200, 200, 200)), ("specflow_stratified", "stratified ep.", C["strat"]),
                ("cv", "const. velocity", (200, 200, 200)))
        for k, nm, col in rows:
            r = M[k]
            j = f"{r['jerk_veh']:5.1f}" if "jerk_veh" in r else "    -"
            inf = f"{r['infeas_veh'] * 100:4.0f}%" if "infeas_veh" in r else "    -"
            put(R, f"{nm:<17} {r['ade']:6.2f}  {r['fde']:6.2f}  {j}  {inf}", (6, yy), 0.37, col)
            yy += 18
        put(R, f"truth (clean)                     {M['truth']['jerk_veh']:5.1f}  {M['truth']['infeas_veh'] * 100:4.0f}%", (6, yy), 0.37, C["gt"])
        put(R, "paper (ETH/UCY avg): minADE 0.15 / minFDE 0.22", (6, yy + 20), 0.36, DIM)
        # bottom strip
        put(img, "boxes = tracked agents | grey dots = observed history (input) | green dashed = ground truth | "
                 "blue = 20 SpecFlow hypotheses | stars/diamonds = endpoint candidates (star = top-1)", (8, 630), 0.37, (200, 200, 200))
        extra = " | magenta = confidence-stratified endpoints" if show_strat else ""
        put(img, f"red dashed (focus agent) = position-state flow{extra}", (8, 650), 0.37, (200, 200, 200))
        put(img, f"{label}: test scene #{si}" + (" (median-error scene, not the best)" if not show_strat else
                                                   " (chosen as a top-1 endpoint FAILURE case)"), (8, 674), 0.4, (180, 180, 180))
        put(img, "SYNTHETIC intersection simulator, my re-implementation from the abstract of arXiv 2611.26390 - "
                 "not ETH/UCY/SDD/NBA, not the authors' model or code", (8, 702), 0.4, (150, 150, 150))
        frames.append(img)
    return dict(scene=int(si), focus=int(focus), strat_fde_focus=float(fde_s[focus]) if show_strat else None, ade=ade[A].round(2).tolist(), fde=fde[A].round(2).tolist(),
                man=[MANEUVERS[d["man"][si, a]] for a in A])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/specflow_simulation.gif")
    ap.add_argument("--mp4", default="")
    ap.add_argument("--gif_every", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(2)
    d = load_data()["test"]
    S = json.load(open(ROOT / "results/summary.json"))
    mf, _ = load_model("specflow", 0)
    mp, _ = load_model("pos_state", 0)
    frames = []
    title = np.full((720, 1280, 3), 22, np.uint8)
    put(title, "Day 78 - Every agent at the intersection, 20 futures each", (230, 290), 1.0, TXT, 2)
    put(title, "SpecFlow: flow matching in the ACCELERATION SPECTRUM + endpoint-conditioned denoiser", (150, 340), 0.72, C["sf"], 2)
    put(title, "SYNTHETIC demo - toy re-implementation of arXiv 2611.26390", (330, 390), 0.7, (160, 160, 160), 1)
    frames += [title] * 15
    s1, s2 = S["clip_scenes"][0], S["failure_scene"]
    info = [render_scene(s1, d, mf, mp, S, "scene 1", frames)]
    # failure scene: focus = the agent with the largest top-1 vs stratified FDE gap
    o1 = scene_preds(mf, d, s2); o2 = scene_preds(mf, d, s2, ep_mode="sampled")
    gt = d["fut_clean"][s2]
    f1 = np.linalg.norm(o1["p"][0].numpy()[..., -1, :] - gt[None, :, -1], axis=-1).min(0)
    f2 = np.linalg.norm(o2["p"][0].numpy()[..., -1, :] - gt[None, :, -1], axis=-1).min(0)
    gap = np.where(d["mask"][s2], f1 - f2, -1e9)
    info.append(render_scene(s2, d, mf, mp, S, "scene 2", frames, show_strat=True, focus=int(gap.argmax())))
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave(out, [cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA) for f in rgb[::a.gif_every]],
                    duration=0.1 * a.gif_every, loop=0)
    if a.mp4:
        imageio.mimsave(ROOT / a.mp4, rgb, fps=10, codec="libx264", quality=7, macro_block_size=8)
    n1 = 15 + (T_OBS - 1) * SUB + 1 + 30 + 12 * SUB + 5
    cv2.imwrite(str(out.with_name("still.png")), frames[n1])
    print(f"{len(frames)} frames -> {out}")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
