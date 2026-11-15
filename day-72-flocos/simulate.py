"""Real-time synthetic "AI in action" simulation for FloCOS (OpenCV).

Layout 1280 x 720, dark UI
  top (1280 x 380) bird's-eye view, animated over the 3 s horizon (target-centred frame):
      INPUT         target history (grey dots) + present box, lane markings, ego in the adjacent lane
      GROUND TRUTH  target moves along its REAL future (green box + green dashed path)
      PREDICTION    FloCOS union of M convex polygons at the current step (cyan, translucent; faint outline of
                    the whole 3 s tube), DetReg single footprint (orange box), physics reach set (grey box)
      PLANNING      three ego ghosts that plan a merge against each prediction (cyan / orange / grey);
                    a red X marks any real collision, a "STUCK" tag marks a merge that was never attempted
  bottom-left    coverage of the true footprint over time, per method
  bottom-middle  occupied area over time (m^2, log scale)
  bottom-right   running tally (merge completed / collisions) over the scenes shown + full-split numbers

    python simulate.py --out assets/flocos_simulation.gif --mp4 assets/flocos_simulation.mp4
SYNTHETIC toy world - not the Waymo Open Motion Dataset, not the authors' code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from flocos import data as D
from flocos import geometry as G
from flocos import planner as PL
from flocos import world as W
from evaluate import fixed_noise, metrics, predictors
from flocos.model import footprint_set, traj_heading
from train import CK, ROOT, get_data

C_GT, C_DET, C_FLO, C_RCH = (90, 230, 90), (0, 150, 255), (255, 220, 0), (170, 170, 170)
C_TXT, BG, C_RED = (235, 235, 235), (22, 22, 22), (60, 60, 255)
X0, X1, Y0, Y1 = -25.0, 80.0, -9.0, 9.0
BW, BH = 1280, 380
SX = BW / (X1 - X0)
SY = BH / (Y1 - Y0)


def put(img, txt, org, scale=0.45, col=C_TXT, th=1):
    (w, h), b = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + b), BG, -1)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def px(xy):
    """World (target frame, metres) -> BEV pixels.  y is exaggerated (SY > SX) so lanes stay readable."""
    xy = np.asarray(xy, float)
    return np.stack([(xy[..., 0] - X0) * SX, BH - (xy[..., 1] - Y0) * SY], -1).astype(np.int32)


def rect(img, c, psi, col, th=2, fill=False, L=W.VEH_L, Wd=W.VEH_W):
    pts = W.footprint_corners(np.asarray(c, float), np.asarray(psi, float), L, Wd)
    if fill:
        cv2.fillPoly(img, [px(pts)], col)
    cv2.polylines(img, [px(pts)], True, col if not fill else (15, 15, 15), th, cv2.LINE_AA)


def poly(img, c, s, col, alpha=None, th=1):
    v = G.polygon_vertices(np.asarray(c, float), np.asarray(s, float))
    if len(v) < 3:
        return
    if alpha is not None:
        ov = img.copy()
        cv2.fillPoly(ov, [px(v)], col)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, dst=img)
    cv2.polylines(img, [px(v)], True, col, th, cv2.LINE_AA)


def dashed(img, pts, col, th=2, seg=10):
    p = px(pts)
    for i in range(len(p) - 1):
        a, b = p[i].astype(float), p[i + 1].astype(float)
        n = max(1, int(np.linalg.norm(b - a) / seg))
        for j in range(0, n, 2):
            u, v = a + (b - a) * j / n, a + (b - a) * min(n, j + 1) / n
            cv2.line(img, tuple(u.astype(int)), tuple(v.astype(int)), col, th, cv2.LINE_AA)


def interp(path, t, start=None):
    """path (H, 2) at T_FUT -> position at time t (linear, from `start` at t=0)."""
    start = np.zeros(2) if start is None else start
    full = np.concatenate([start[None], path], 0)
    tt = np.concatenate([[0.0], W.T_FUT])
    return np.array([np.interp(t, tt, full[:, 0]), np.interp(t, tt, full[:, 1])])


def step_at(t):
    return int(np.clip(np.ceil(t / W.DT_F) - 1, 0, W.H - 1))


def plot(img, x0, y0, w, h, series, ymin, ymax, title, tmax, log=False, hlines=()):
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
    put(img, title, (x0 + 4, y0 + 15), 0.42)
    f = (lambda v: np.log10(np.clip(v, 1e-3, None))) if log else (lambda v: v)
    lo, hi = f(ymin), f(ymax)
    Y = lambda v: (y0 + h - 6 - (np.clip(f(v), lo, hi) - lo) / (hi - lo) * (h - 34)).astype(int)
    for yv, col, lab in hlines:
        yy = int(Y(np.array(yv)))
        cv2.line(img, (x0, yy), (x0 + w, yy), col, 1)
        put(img, lab, (x0 + 4, yy - 3), 0.33, col)
    xl = x0 + w - 6
    for xs, ys, col, lab in reversed(series):
        if len(xs) > 1:
            P = np.stack([x0 + xs / tmax * w, Y(ys)], -1).astype(np.int32)
            cv2.polylines(img, [P], False, col, 2, cv2.LINE_AA)
        tw = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0][0]
        xl -= tw + 10
        put(img, lab, (xl, y0 + 33), 0.36, col)


def pick_scenes(te, R, n_show):
    """Deterministic, honest mix: one FloCOS-completes / DetReg-collides scene per intent where possible, one
    scene where the reach set gets STUCK but FloCOS merges, and the first scene where FloCOS itself fails."""
    it = te["intent"].numpy()
    fc, dc, rc = R["flo"]["completion"].numpy(), R["det"]["collision"].numpy(), R["rch"]["completion"].numpy()
    fcol = R["flo"]["collision"].numpy()
    picks = []
    for k in (1, 3, 4, 2):
        c = np.where((it == k) & (fc > 0) & (dc > 0))[0]
        if len(c):
            picks.append(int(c[0]))
    c = [int(i) for i in np.where((fc > 0) & (rc == 0) & (dc == 0))[0] if i not in picks]
    if c:
        picks.append(c[0])
    picks = picks[:n_show - 1]
    fail = np.where(fcol > 0)[0]
    if len(fail):
        picks.append(int(fail[0]))
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tag", default="s0")
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--fps-per-step", type=int, default=3)
    ap.add_argument("--out", default="assets/flocos_simulation.gif")
    ap.add_argument("--mp4", default="")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    te = get_data(cfg, "test")
    n = len(te["v0"])
    fm, fs, det, mtp, mc = predictors(cfg, f"flocos_set_{a.tag}.pt", a.tag)
    M, nfe = mc["M"], mc["nfe"]
    with torch.no_grad():
        noise = fixed_noise(n, M)
        cf, sf = [torch.cat(z) for z in zip(*[fs.predict_set(D.batch(te, slice(i, i + 500)), M, nfe, noise[i:i + 500])
                                              for i in range(0, n, 500)])]
        cd, sd = footprint_set(torch.cat([det(D.batch(te, slice(i, i + 500))) for i in range(0, n, 500)]))
        cr, ex = D.physics_reach_box(te)
        sr = G.rect_support(None, ex=ex[..., 0], ey=ex[..., 1])
    det_head = traj_heading(cd[:, 0]).numpy()                                   # (N, H)
    sets = dict(flo=(cf, sf), det=(cd, sd), rch=(cr, sr))
    R, COV, AREA = {}, {}, {}
    for k, (c, s) in sets.items():
        _, COV[k], AREA[k], R[k] = metrics(c, s, te)
    picks = pick_scenes(te, R, a.scenes)
    print("scenes", picks, [W.INTENT_NAMES[int(te["intent"][i])] for i in picks])
    full = json.load(open(CK / f"metrics_{a.tag}.json")) if (CK / f"metrics_{a.tag}.json").exists() else None

    names = dict(flo=("FloCOS", C_FLO), det=("DetReg", C_DET), rch=("Reach set", C_RCH))
    tally = {k: [0, 0] for k in names}
    frames, steps = [], W.H * a.fps_per_step
    for si, i in enumerate(picks):
        hist = te["hist"][i, :, :2].numpy()
        fut, head = te["fut"][i].numpy(), te["head"][i].numpy()
        lane_y = float(te["lane_y"][i])
        ego = te["ego"][i].numpy()
        ego_start = np.array([ego[1], lane_y + ego[0] * W.LANE_W])
        plans = {k: R[k]["traj"][i].numpy() for k in names}
        merged = {k: bool(R[k]["merged"][i]) for k in names}
        coll = {k: bool(R[k]["collision"][i]) for k in names}
        # first collision time per method (true footprint vs ego, step resolution)
        t_hit = {}
        for k in names:
            if coll[k]:
                for h in range(W.H):
                    sub = PL.collides_true(torch.tensor(plans[k][None, :h + 1]), torch.tensor(fut[None, :h + 1]),
                                           torch.tensor(head[None, :h + 1]))
                    if bool(sub[0]):
                        t_hit[k] = W.T_FUT[h]
                        break
        for k in names:
            tally[k][0] += int(merged[k] and not coll[k]); tally[k][1] += int(coll[k])
        intent = W.INTENT_NAMES[int(te["intent"][i])]
        for f in range(steps + 10):
            t = min(f, steps) / steps * W.T_FUT[-1]
            h = step_at(t) if t > 0 else 0
            cv = np.full((720, 1280, 3), 22, np.uint8)
            bev = np.full((BH, BW, 3), 34, np.uint8)
            # road: target lane centred at lane_y, one lane each side
            for dy in (-1.5, -0.5, 0.5, 1.5):
                yy = lane_y + dy * W.LANE_W
                if abs(dy) == 1.5:
                    cv2.line(bev, tuple(px([X0, yy])), tuple(px([X1, yy])), (200, 200, 200), 2)
                else:
                    dashed(bev, np.array([[X0, yy], [X1, yy]]), (190, 190, 190), 1, 16)
            # physics reach set at the current step (grey) and the FloCOS tube outline (all steps, faint)
            poly(bev, cr[i, 0, h].numpy(), sr[i, 0, h].numpy(), C_RCH, alpha=0.10, th=3)
            for hh in range(W.H):
                for m in range(M):
                    poly(bev, cf[i, m, hh].numpy(), sf[i, m, hh].numpy(), (110, 95, 20), th=1)
            for m in range(M):                                                  # current-step union, bright
                poly(bev, cf[i, m, h].numpy(), sf[i, m, h].numpy(), C_FLO, alpha=0.22, th=2)
            # input: history + present
            for p_ in hist[:-1]:
                cv2.circle(bev, tuple(px(p_)), 3, (150, 150, 150), -1)
            rect(bev, np.zeros(2), 0.0, (120, 120, 120), 1)
            dashed(bev, np.concatenate([[[0, 0]], fut]), C_GT, 2)
            # DetReg single footprint
            psi_d = np.interp(t, np.concatenate([[0], W.T_FUT]), np.concatenate([[0], det_head[i]]))
            rect(bev, interp(cd[i, 0].numpy(), t), psi_d, C_DET, 2)
            # ego ghosts
            for k, (lab, col) in names.items():
                pe = interp(plans[k], t, ego_start)
                nxt = interp(plans[k], min(t + 0.1, 3.0), ego_start)
                psi_e = float(np.arctan2(nxt[1] - pe[1], max(nxt[0] - pe[0], 1e-2)))
                rect(bev, pe, psi_e, tuple(int(0.55 * v) for v in col), 2, fill=True, L=W.EGO_L, Wd=W.EGO_W)
                put(bev, f"ego|{lab}", px(pe) + [-28, -14], 0.34, col)
                if not merged[k] and t > 1.0:
                    put(bev, f"{lab}: STUCK (no feasible merge)", px(pe) + [-60, 30 + 16 * list(names).index(k)], 0.4, col)
                if k in t_hit and t >= t_hit[k]:
                    ph = px(interp(plans[k], t_hit[k], ego_start))
                    cv2.drawMarker(bev, tuple(ph), C_RED, cv2.MARKER_TILTED_CROSS, 30, 3)
                    put(bev, f"{lab} ego COLLIDES at t = {t_hit[k]:.2f} s", ph + [-90, -22], 0.45, C_RED, 1)
            # ground truth on top: where the target REALLY is now
            pt = interp(fut, t)
            psi_t = np.interp(t, np.concatenate([[0], W.T_FUT]), np.concatenate([[0], head]))
            rect(bev, pt, psi_t, C_GT, 4)
            put(bev, "TARGET (truth)", px(pt) + [-45, 34], 0.42, C_GT)
            put(bev, f"physics reach set @t: {AREA['rch'][i, h]:.0f} m2 (grey, wider than the road)",
                (BW - 430, 44), 0.42, C_RCH)
            put(bev, f"scene {si + 1}/{len(picks)}   t = {t:4.2f} s   hidden intention: {intent}   "
                     f"ego task: merge into the target's lane", (8, 18), 0.5)
            put(bev, "grey dots = 1 s history | green = ground-truth future | cyan = FloCOS union of 6 convex polygons "
                     "(faint = full 3 s tube) | orange outline = DetReg footprint | filled = ego plans",
                (8, BH - 10), 0.36)
            cv[:BH] = bev
            # telemetry: coverage over time
            ts = W.T_FUT[:h + 1] if t > 0 else W.T_FUT[:1]
            pnl = cv[BH + 8:716, 6:426]
            plot(pnl, 2, 2, 414, 160, [(ts, COV[k][i, :len(ts)].numpy(), names[k][1], names[k][0]) for k in names],
                 0, 1.05, "coverage of TRUE footprint", 3.0, hlines=((1.0, (90, 160, 90), "100%"),))
            plot(pnl, 2, 168, 414, 158, [(ts, AREA[k][i, :len(ts)].numpy(), names[k][1], names[k][0]) for k in names],
                 5, 400, "occupied area (m^2, log)", 3.0, log=True, hlines=((8.74, (90, 160, 90), "1 car"),))
            # scene status
            st = cv[BH + 8:716, 432:862]
            put(st, "planner outcome in this scene (truth)", (6, 16), 0.45)
            for r_, (k, (lab, col)) in enumerate(names.items()):
                msg = "COLLISION" if coll[k] else ("merge completed" if merged[k] else "stuck: never merged")
                put(st, f"{lab:10s} {msg}", (6, 44 + 24 * r_), 0.45, C_RED if coll[k] else col)
            put(st, f"area @t: FloCOS {AREA['flo'][i, h]:.0f} m2 | reach {AREA['rch'][i, h]:.0f} m2", (6, 130), 0.42)
            put(st, f"coverage @t: FloCOS {COV['flo'][i, h]:.2f} | DetReg {COV['det'][i, h]:.2f}", (6, 154), 0.42)
            put(st, "planner: candidate merges filtered by", (6, 196), 0.4, (180, 180, 180))
            put(st, "max_h n_h.(p - c_i(t)) - s_ih(t) > margin  for all i, t, p", (6, 216), 0.4, (180, 180, 180))
            put(st, "SYNTHETIC toy world - not WOMD, not the authors' code", (6, 320), 0.38, (150, 150, 150))
            # tally
            tb_ = cv[BH + 8:716, 868:1274]
            put(tb_, f"running tally, {si + 1} scene(s) shown", (4, 16), 0.42)
            for r_, (k, (lab, col)) in enumerate(names.items()):
                y_ = 34 + r_ * 42
                cv2.rectangle(tb_, (110, y_), (110 + int(250 * tally[k][0] / len(picks)), y_ + 20), col, -1)
                cv2.rectangle(tb_, (110, y_), (360, y_ + 20), (90, 90, 90), 1)
                put(tb_, lab, (4, y_ + 16), 0.42, col)
                put(tb_, f"completed {tally[k][0]} | collisions {tally[k][1]}", (114, y_ + 36), 0.36, col)
            if full:
                rr = full["results"]
                put(tb_, f"full test split ({full['n_test']} agents):", (4, 182), 0.4)
                put(tb_, "method      cov    area   compl  coll", (4, 204), 0.38, (180, 180, 180))
                for r_, (lab, key, col) in enumerate((("DetReg", "DetReg (1 footprint)", C_DET),
                                                       ("MTP-6", "MTP-6 (6 footprints)", (200, 120, 255)),
                                                       ("Reach", "Physics reach set", C_RCH),
                                                       ("FloCOS", f"FloCOS full (M={M})", C_FLO))):
                    q = rr[key]
                    put(tb_, f"{lab:9s} {q['coverage']:.3f} {q['area']:6.1f} {q['completion']:.3f} {q['collision']:.3f}",
                        (4, 226 + 22 * r_), 0.38, col)
            frames.append(cv)
            if a.live:
                cv2.imshow("FloCOS", cv)
                if cv2.waitKey(30) == 27:
                    return
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rgb = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in frames]
    imageio.mimsave(a.out, [cv2.resize(fr, (960, 540), interpolation=cv2.INTER_AREA) for fr in rgb[::2]],
                    duration=0.16, loop=0)
    if a.mp4:
        imageio.mimsave(a.mp4, rgb, fps=12, codec="libx264", quality=7, macro_block_size=8)
    cv2.imwrite(str(Path(a.out).with_name("still.png")), frames[min(len(frames) - 1, int(steps * 0.8))])
    print(f"{len(frames)} frames -> {a.out} {a.mp4}")


if __name__ == "__main__":
    main()
