"""Real-time synthetic "AI in action" simulation for ForeseeDrive (OpenCV).

Layout (1280 x 720, dark UI)
  top (1280 x 350)  BEV of the scene, animated over the 4 s horizon:
        INPUT      actor history tails (grey) + present boxes
        GROUND TRUTH  actors move along their LOGGED future (green), expert plan (green dashed)
        PREDICTION base generator: K candidates (thin grey) + selected plan / ghost car (orange)
                   ForeseeDrive (predicted tokens): candidates (thin cyan) + selected / ghost car (cyan)
                   ForeseeDrive (logged-future "oracle" tokens, diagnostic only): magenta dashed
        future tokens drawn in place: route anchors on the ego lane + occupancy cells (red = predicted
        boundary risk r), actor tokens (r, crossing time) next to each ranked actor
  bottom-left   token panel: 36 queries (actor | route | occupancy | margin) teacher r vs predicted r
  bottom-middle telemetry: clearance-to-actors over time (true future) and speed profiles
  bottom-right  running PDMS / collision tally over the scenes shown

    python simulate.py --out assets/foresee_simulation.gif --mp4 assets/foresee_simulation.mp4
SYNTHETIC toy world - not NAVSIM / nuScenes, not the authors' code.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from foresee import data as D
from foresee import losses as L
from foresee import tokens as TK
from foresee import world as W
from foresee.model import inject
from evaluate import pdm_t
from train import CFG, CK, batched, build_cond, get_data, load_base, load_pred

C_GT, C_BASE, C_FD, C_ORC = (90, 230, 90), (0, 150, 255), (255, 220, 0), (230, 80, 230)
C_TXT, BG = (235, 235, 235), (22, 22, 22)
X0, X1, Y0, Y1 = -8.0, 52.0, -6.5, 9.5
BW, BH = 1280, 350
SX = BW / (X1 - X0)
EVENT_TXT = {1: "lead vehicle BRAKES HARD (hidden)", 2: "pedestrian STEPS OUT (weak cue)", 3: "left-lane car CUTS IN (weak cue)"}


def put(img, txt, org, scale=0.45, col=C_TXT, th=1):
    (w, h), b = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + b), BG, -1)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def px(xy):
    xy = np.asarray(xy, float)
    return np.stack([(xy[..., 0] - X0) * SX, BH - (xy[..., 1] - Y0) * SX], -1).astype(np.int32)


def box(img, c, heading, length, width, col, th=2, fill=False):
    ca, sa = np.cos(heading), np.sin(heading)
    corners = np.array([[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]])
    pts = c + corners @ np.array([[ca, sa], [-sa, ca]])
    if fill:
        cv2.fillPoly(img, [px(pts)], col)
    cv2.polylines(img, [px(pts)], True, col if not fill else (20, 20, 20), th, cv2.LINE_AA)


def dashed(img, pts, col, th=2, seg=8):
    p = px(pts)
    for i in range(len(p) - 1):
        a, b = p[i].astype(float), p[i + 1].astype(float)
        n = max(1, int(np.linalg.norm(b - a) / seg))
        for j in range(0, n, 2):
            u, v = a + (b - a) * j / n, a + (b - a) * min(n, j + 1) / n
            cv2.line(img, tuple(u.astype(int)), tuple(v.astype(int)), col, th, cv2.LINE_AA)


def interp(traj, t, start=None):
    """traj (H, 2) sampled at T_FUT; position at time t (s) starting from `start` (default origin)."""
    start = np.zeros(2) if start is None else start
    full = np.concatenate([start[None], traj], 0)
    tt = np.concatenate([[0.0], W.T_FUT])
    return np.array([np.interp(t, tt, full[:, 0]), np.interp(t, tt, full[:, 1])])


def heading_at(traj, t, start=None):
    a, b = interp(traj, max(0, t - 0.1), start), interp(traj, t + 0.1, start)
    d = b - a
    return float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-3 else 0.0


def clearance_curve(traj, sc):
    tt = np.linspace(0.05, 4.0, 80)
    out = []
    for t in tt:
        e = interp(traj, t)
        cur = sc["hist"][:, -1, :2]
        best = np.inf
        for j in np.where(sc["valid"])[0]:
            a = interp(sc["fut"][j], t, cur[j])
            best = min(best, np.linalg.norm(e - a) - W.R_EGO - sc["radius"][j])
        out.append(best)
    return tt, np.array(out)


def speed_curve(traj, v0):
    p = np.concatenate([[[0, 0]], traj], 0)
    return np.concatenate([[v0], np.linalg.norm(np.diff(p, axis=0), axis=-1) / W.DT])


def plot(img, x0, y0, w, h, series, ymin, ymax, title, xmax=4.0, hlines=()):
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
    put(img, title, (x0 + 4, y0 + 14), 0.42)
    for yv, col in hlines:
        yy = int(y0 + h - (yv - ymin) / (ymax - ymin) * (h - 20))
        cv2.line(img, (x0, yy), (x0 + w, yy), col, 1)
    for k, (xs, ys, col, lab) in enumerate(series):
        ys = np.clip(ys, ymin, ymax)
        P = np.stack([x0 + xs / xmax * w, y0 + h - (ys - ymin) / (ymax - ymin) * (h - 20)], -1).astype(np.int32)
        cv2.polylines(img, [P], False, col, 2, cv2.LINE_AA)
        xs_lab = x0 + w - 6 - sum(cv2.getTextSize(l_[3], cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0] + 12 for l_ in series[k:])
        put(img, lab, (xs_lab, y0 + 32), 0.38, col)


def pick_scenes(te, res, n_show):
    """Deterministic, honest mix: wins, a failure the oracle fixes, and a quiet exact no-op scene."""
    hidden = (te["event"] > 0).any(-1).numpy()
    pb, pf, po = res["base"], res["fd"], res["orc"]
    moved = res["moved"]
    wins = np.where(hidden & (pf - pb > 0.4))[0]
    fail = np.where(hidden & (pb < 0.1) & (pf < 0.1) & (po > 0.8))[0]
    quiet = np.where(~moved & (pb > 0.9))[0]
    ev = te["event"].numpy().max(-1)
    picks = []
    for e in (1, 2, 3):                                                     # one win per event type if possible
        c = [i for i in wins if ev[i] == e]
        if c:
            picks.append(int(c[0]))
    n_win = n_show - 1 - int(len(quiet) > 0)
    extra = [int(i) for i in wins if i not in picks]
    while len(picks) < n_win and extra:
        picks.append(extra.pop(0))
    picks = picks[:n_win]
    if len(fail):
        picks.append(int(fail[0]))
    if len(quiet):
        picks.append(int(quiet[0]))
    return picks[:n_show]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=5)
    ap.add_argument("--fps-per-step", type=int, default=4)
    ap.add_argument("--out", default="assets/foresee_simulation.gif")
    ap.add_argument("--mp4", default="")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(CFG)); tc = cfg["train"]
    _, te = get_data(cfg)
    n = len(te["v_ego"])
    base, pred = load_base(cfg), load_pred(cfg)
    fd = build_cond(cfg, base, "full")
    fd.load_state_dict(torch.load(CK / "cond_full_s0.pt"), strict=False); fd.eval()
    tb, lb = batched(lambda i: base(D.batch(te, i))[:2], n)
    Pe, Pc, Pr = batched(lambda i: pred(D.batch(te, i), tb[i]), n)
    tE, tC, tR = L.teacher_tokens(tb, te)
    v_pred, v_orc = inject(Pe, Pc, Pr, True), inject(tE, tC, tR, False)

    def run(v):
        tf, lf = batched(lambda i: (lambda o: (o["traj"], o["logit"]))(fd(D.batch(te, i), v[i])), n)
        tf, _ = L.apply_guard(tf, tb, te, tc["eps_b"], tc["eps_g"])
        return tf, lf
    tf, lf = run(v_pred)
    to, lo = run(v_orc)
    ar = torch.arange(n)
    sel = lambda t, l: t[ar, l.argmax(-1)]
    sb, sf, so = sel(tb, lb), sel(tf, lf), sel(to, lo)
    P = lambda t: pdm_t(t[:, None], te)
    mb, mf, mo = P(sb), P(sf), P(so)
    res = dict(base=mb["pdms"][:, 0].numpy(), fd=mf["pdms"][:, 0].numpy(), orc=mo["pdms"][:, 0].numpy(),
               moved=((tf - tb).abs().amax((1, 2, 3)) > 0).numpy())
    picks = pick_scenes(te, res, a.scenes)
    print("scenes", picks)

    frames, tally = [], dict(base=[], fd=[], orc=[], cb=[], cf=[], co=[])
    steps = W.H * a.fps_per_step
    for si, i in enumerate(picks):
        sc = D.scene_dict(te, i)
        cur = sc["hist"][:, -1, :2]
        rank = te["rank"][i].numpy()
        rp = torch.sigmoid(Pr[i]).numpy(); cp = torch.sigmoid(Pc[i]).numpy()
        tr_ = tR[i].numpy(); tc_ = tC[i].numpy()
        b_sel, f_sel, o_sel = sb[i].numpy(), sf[i].numpy(), so[i].numpy()
        ev = int(sc["event"].max())
        tt, cb_ = clearance_curve(b_sel, sc); _, cf_ = clearance_curve(f_sel, sc); _, co_ = clearance_curve(o_sel, sc)
        tally["base"].append(res["base"][i]); tally["fd"].append(res["fd"][i]); tally["orc"].append(res["orc"][i])
        tally["cb"].append(float(mb["nc"][i, 0] == 0)); tally["cf"].append(float(mf["nc"][i, 0] == 0)); tally["co"].append(float(mo["nc"][i, 0] == 0))
        for f in range(steps + 8):
            t = min(f, steps) / steps * 4.0
            cv = np.full((720, 1280, 3), 22, np.uint8)
            bev = np.full((BH, BW, 3), 34, np.uint8)
            # road
            cv2.rectangle(bev, tuple(px([X0, W.Y_MAX])), tuple(px([X1, W.Y_MIN])), (55, 55, 60), -1)
            for yy in (W.Y_MIN, W.Y_MAX):
                cv2.line(bev, tuple(px([X0, yy])), tuple(px([X1, yy])), (200, 200, 200), 2)
            dashed(bev, np.array([[X0, 1.75], [X1, 1.75]]), (200, 200, 200), 1, 14)
            for yy in (-3.5, 7.0):
                cv2.rectangle(bev, tuple(px([X0, yy + 1.2])), tuple(px([X1, yy - 1.2])), (48, 52, 48), -1)
            # occupancy cells + route anchors coloured by PREDICTED boundary risk
            q = TK.N_A + TK.N_R
            for lane in TK.OCC_LANES:
                for bb in range(4):
                    r = rp[q] * (cp[q] >= 0.5)
                    if r > 0.3:
                        ov = bev.copy()
                        cv2.rectangle(ov, tuple(px([TK.OCC_X[bb], lane + 1.7])), tuple(px([TK.OCC_X[bb + 1], lane - 1.7])),
                                      (40, 40, int(90 + 160 * r)), -1)
                        bev = cv2.addWeighted(ov, 0.35, bev, 0.65, 0)
                    q += 1
            for k, s in enumerate(TK.ROUTE_S):
                r = rp[TK.N_A + k]
                cv2.circle(bev, tuple(px([s, 0.0])), 5, (60, 60, int(80 + 175 * r)) if r > 0.3 else (110, 110, 110), -1)
            # candidates
            for k in range(base.K):
                cv2.polylines(bev, [px(np.concatenate([[[0, 0]], tb[i, k].numpy()]))], False, (110, 110, 110), 1, cv2.LINE_AA)
                cv2.polylines(bev, [px(np.concatenate([[[0, 0]], tf[i, k].numpy()]))], False, (120, 110, 30), 1, cv2.LINE_AA)
            dashed(bev, np.concatenate([[[0, 0]], sc["expert"]]), C_GT, 2)
            dashed(bev, np.concatenate([[[0, 0]], o_sel]), C_ORC, 2, 5)
            cv2.polylines(bev, [px(np.concatenate([[[0, 0]], b_sel]))], False, C_BASE, 3, cv2.LINE_AA)
            cv2.polylines(bev, [px(np.concatenate([[[0, 0]], f_sel]))], False, C_FD, 3, cv2.LINE_AA)
            # actors: history tails, ground-truth motion
            for j in np.where(sc["valid"])[0]:
                for p_ in sc["hist"][j, :-1, :2]:
                    cv2.circle(bev, tuple(px(p_)), 2, (150, 150, 150), -1)
                pj = interp(sc["fut"][j], t, cur[j])
                if sc["typ"][j] == W.TYPE_PED:
                    cv2.circle(bev, tuple(px(pj)), int(0.5 * SX), C_GT, 2)
                else:
                    box(bev, pj, heading_at(sc["fut"][j], t, cur[j]), 4.5, 1.9, C_GT, 2)
            for slot, j in enumerate(rank):
                if j >= 0 and cp[slot] >= 0.5:
                    pj = interp(sc["fut"][j], t, cur[j])
                    put(bev, f"A{slot} r {rp[slot]:.2f} | true {tr_[slot]:.0f}", px(pj) + [-40, -18], 0.36,
                        (90, 90, 255) if rp[slot] > 0.5 else (200, 200, 200))
            # ghost egos
            hit_b = False
            for traj, col in ((b_sel, C_BASE), (f_sel, C_FD)):
                pe = interp(traj, t)
                box(bev, pe, heading_at(traj, t), 4.5, 1.9, col, 2, fill=True)
                if col == C_BASE:
                    k_ = int(np.clip(np.searchsorted(tt, t), 0, len(tt) - 1))
                    hit_b = cb_[:k_ + 1].min() < 0 if t > 0 else False
            box(bev, np.zeros(2), 0.0, 4.5, 1.9, (230, 160, 60), 1)
            for curve, traj, col, name, dy in ((cb_, b_sel, (40, 40, 255), "BASE", 42), (cf_, f_sel, (255, 255, 120), "ForeseeDrive", 64)):
                if (curve < 0).any() and t >= tt[int(np.argmax(curve < 0))]:
                    t_hit = tt[int(np.argmax(curve < 0))]
                    ph = px(interp(traj, t_hit))
                    cv2.drawMarker(bev, tuple(ph), col, cv2.MARKER_TILTED_CROSS, 30, 3)
                    put(bev, f"{name} collides at t = {t_hit:.1f} s", ph + [-90, dy], 0.48, col, 2)
            put(bev, f"scene {si + 1}/{len(picks)}  t = {t:3.1f} s   " +
                (f"hidden future: {EVENT_TXT[ev]}" if ev else "quiet scene: no interaction tokens fire"), (8, 18), 0.5)
            put(bev, "green = ground truth (actors + expert dashed)   orange = base generator   cyan = ForeseeDrive (predicted tokens)"
                     "   magenta dashed = ForeseeDrive w/ logged-future tokens (diagnostic)", (8, BH - 10), 0.38)
            cv[:BH] = bev

            # token panel
            tp = cv[BH + 10:715, 6:426]
            put(tp, "future tokens: boundary risk r  (top teacher / bottom predicted)", (4, 14), 0.4)
            Q = len(rp)
            cw = 410 / Q
            for qq in range(Q):
                for row, val, on in ((0, tr_[qq], tc_[qq]), (1, rp[qq], cp[qq] >= 0.5)):
                    col = (40, 40, int(60 + 195 * val)) if on else (45, 45, 45)
                    x_ = int(5 + qq * cw); y_ = 30 + row * 70
                    cv2.rectangle(tp, (x_, y_), (int(x_ + cw - 2), y_ + 60), col, -1)
            for lab, a0, a1 in (("actor", 0, TK.N_A), ("route", TK.N_A, TK.N_A + TK.N_R),
                                ("occ", TK.N_A + TK.N_R, TK.N_SCENE_Q), ("margin (per candidate)", TK.N_SCENE_Q, Q)):
                put(tp, lab, (int(5 + a0 * cw), 182), 0.36)
                cv2.line(tp, (int(5 + a0 * cw), 168), (int(5 + a1 * cw) - 3, 168), (160, 160, 160), 1)
            live = int((v_pred[i].abs().sum(-1) > 0).sum())
            put(tp, f"live predicted tokens: {live}/{Q}   (grey = suppressed, c < 0.5)", (4, 210), 0.4)
            put(tp, "all tokens zero -> u = 0 -> output IDENTICAL to base (tested: max diff 0.0)", (4, 232), 0.4, C_FD)
            put(tp, f"scene PDMS  base {res['base'][i]:.2f}  ForeseeDrive {res['fd'][i]:.2f}  oracle-token {res['orc'][i]:.2f}",
                (4, 262), 0.4)
            put(tp, "SYNTHETIC toy world - not NAVSIM, not the authors' code", (4, 340), 0.38, (150, 150, 150))
            # telemetry
            tl = cv[BH + 10:715, 432:862]
            m = tt <= max(t, 0.05)
            plot(tl, 2, 2, 424, 170, [(tt[m], cb_[m], C_BASE, "base"), (tt[m], cf_[m], C_FD, "ForeseeDrive"),
                                      (tt[m], co_[m], C_ORC, "oracle tokens")], -2, 8,
                 "clearance to nearest actor (m, true future)", hlines=((0, (60, 60, 255)), (0.5, (90, 90, 160))))
            ts = np.concatenate([[0.0], W.T_FUT]); ms_ = ts <= t + 1e-6
            plot(tl, 2, 180, 424, 165, [(ts[ms_], speed_curve(sc["expert"], sc["v_ego"])[ms_], C_GT, "expert"),
                                        (ts[ms_], speed_curve(b_sel, sc["v_ego"])[ms_], C_BASE, "base"),
                                        (ts[ms_], speed_curve(f_sel, sc["v_ego"])[ms_], C_FD, "ForeseeDrive")], 0, 15,
                 "speed (m/s)")
            # tally
            tb_ = cv[BH + 10:715, 868:1274]
            put(tb_, f"running tally over {si + 1} scene(s) shown", (4, 14), 0.42)
            for r_, (lab, key, ckey, col) in enumerate((("base", "base", "cb", C_BASE), ("ForeseeDrive", "fd", "cf", C_FD),
                                                        ("oracle tokens", "orc", "co", C_ORC))):
                v = float(np.mean(tally[key])); c = int(np.sum(tally[ckey]))
                y_ = 40 + r_ * 50
                cv2.rectangle(tb_, (120, y_), (120 + int(250 * v), y_ + 26), col, -1)
                cv2.rectangle(tb_, (120, y_), (370, y_ + 26), (90, 90, 90), 1)
                put(tb_, f"{lab}", (4, y_ + 20), 0.42, col)
                put(tb_, f"PDMS {v:.2f} | collisions {c}", (126, y_ + 46), 0.38, col)
            g = json_cache()
            if g:
                r0 = g["results"]
                put(tb_, f"full test split ({g['n_test']} scenes), PDMS / collision %:", (4, 210), 0.4)
                for r_, (lab, key, col) in enumerate((("base", "base", C_BASE), ("ForeseeDrive (pred)", "foresee_pred", C_FD),
                                                      ("shuffled tokens", "foresee_shuffled", (180, 180, 180)),
                                                      ("oracle tokens", "foresee_oracle", C_ORC))):
                    put(tb_, f"{lab:22s} {r0[key]['pdms']:.3f} / {r0[key]['collision']:.1f}%", (4, 236 + 22 * r_), 0.4, col)
            frames.append(cv)
            if a.live:
                cv2.imshow("ForeseeDrive", cv)
                if cv2.waitKey(30) == 27:
                    return
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rgb = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in frames]
    imageio.mimsave(a.out, [cv2.resize(fr, (960, 540), interpolation=cv2.INTER_AREA) for fr in rgb[::2]], duration=0.16, loop=0)
    if a.mp4:
        imageio.mimsave(a.mp4, rgb, fps=12, codec="libx264", quality=7, macro_block_size=8)
    cv2.imwrite(str(Path(a.out).with_name("still.png")), frames[min(len(frames) - 1, 24)])
    print(f"{len(frames)} frames -> {a.out} {a.mp4}")


_J = None


def json_cache():
    global _J
    if _J is None:
        import json
        p = CK / "metrics_s0.json"
        _J = json.load(open(p)) if p.exists() else {}
    return _J


if __name__ == "__main__":
    main()
