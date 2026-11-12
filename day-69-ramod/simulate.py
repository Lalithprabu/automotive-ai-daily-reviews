"""Real-time synthetic "AI in action" simulation for RaMOD (OpenCV).

Layout (1280 x 720, dark UI)
  top-left    INPUT: raw range-azimuth power (max over Doppler) + ground-truth boxes (green)
  top-middle  azimuth-Doppler map (mean over range) + RaMOD's predicted static-Doppler curve (cyan)
              -> watch the curve fold over the +-v_max seam when the ego goes faster than v_max
  top-right   AFTER static suppression + RaMOD detections (class colour, score, unwrapped radial velocity)
  bottom-left bird's-eye view: radar fan, static clutter (grey), GT boxes (green), RaMOD detections with
              radial-velocity arrows (cyan = unwrapped, orange = raw-Doppler single-frame RAD-CNN baseline)
  bottom-right telemetry: ego speed (true vs RaMOD), per-frame radial-velocity error (RaMOD vs RAD-CNN),
              wrap probability of every RaMOD detection

    python simulate.py --seed 5000 --frames 150 --out assets/ramod_simulation.gif --mp4 assets/ramod_simulation.mp4
SYNTHETIC toy radar world - not BRIDGE-Liv, not the authors' code.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from ramod import world as W
from ramod.data import CUBE_MEAN, CUBE_STD
from ramod.losses import decode
from train import build

CLS_COL = {0: (80, 220, 255), 1: (255, 120, 220), 2: (255, 200, 60)}          # BGR: ped yellow, cyc pink, car blue
C_GT, C_RAMOD, C_BASE = (90, 230, 90), (255, 230, 0), (0, 150, 255)
CLS_SHORT = ("PED", "CYC", "CAR")


def put(img, txt, org, scale=0.42, col=(235, 235, 235), th=1):
    (w, h), b = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = org
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + b), (18, 18, 18), -1)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def heat_img(a: np.ndarray, size: tuple[int, int], lo=None, hi=None, cmap=cv2.COLORMAP_MAGMA):
    lo = np.percentile(a, 5) if lo is None else lo
    hi = np.percentile(a, 99.7) if hi is None else hi
    u8 = (np.clip((a - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(cv2.resize(u8, size, interpolation=cv2.INTER_NEAREST), cmap)


def ra_panel(ra: np.ndarray, w: int, h: int):
    """(R, A) map -> image with range increasing UPWARDS and azimuth left(+y) -> right(-y)."""
    return heat_img(ra[::-1, ::-1], (w, h))


def box_to_px(box, w, h):
    """RA box [r0, a0, r1, a1] (bins) -> panel rectangle (flipped like ra_panel)."""
    sx, sy = w / W.A_BINS, h / W.R_BINS
    x0, x1 = (W.A_BINS - box[3]) * sx, (W.A_BINS - box[1]) * sx
    y0, y1 = (W.R_BINS - box[2]) * sy, (W.R_BINS - box[0]) * sy
    return (int(x0), int(y0)), (int(x1), int(y1))


def ra_to_xy(rb, ab):
    r = rb * W.R_RES
    s = np.clip(ab * W.S_RES - W.S_MAX, -0.999, 0.999)
    return np.array([r * np.sqrt(1 - s ** 2), r * s])


def bev_px(xy, origin=(320, 385), scale=12.5):
    xy = np.asarray(xy, float)
    return np.stack([origin[0] - xy[..., 1] * scale, origin[1] - xy[..., 0] * scale], -1).astype(np.int32)


def line_plot(p, x0, y0, w, h, series, ymin, ymax, title):
    cv2.rectangle(p, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
    put(p, title, (x0 + 4, y0 + 14))
    for k, (vals, col, lab) in enumerate(series):
        v = np.asarray(vals, float)
        if len(v) < 2:
            continue
        ys = y0 + h - (np.clip(v, ymin, ymax) - ymin) / (ymax - ymin) * (h - 22)
        xs = x0 + np.linspace(0, w, len(v))
        ok = np.isfinite(ys)
        if ok.sum() > 1:
            cv2.polylines(p, [np.stack([xs[ok], ys[ok]], -1).astype(np.int32)], False, col, 2, cv2.LINE_AA)
        last = v[ok][-1] if ok.any() else float("nan")
        put(p, f"{lab}: {last:.2f}", (x0 + w - 175, y0 + 16 + 15 * k), 0.38, col)


def pick_episode(seed: int, steps: int, min_ego: float):
    """First episode at/after `seed` whose ego starts faster than min_ego (so static clutter wraps)."""
    s = seed
    while True:
        rng = np.random.default_rng(s)
        sim = W.simulate_episode(rng, steps, waves=max(1, steps // 30))
        if sim["ego"][0, 3] >= min_ego and len(sim["movers"]) >= 2:
            return sim, rng, s
        s += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--min-ego", type=float, default=5.0)
    ap.add_argument("--out", default="assets/ramod_simulation.gif")
    ap.add_argument("--mp4", default="")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))
    torch.set_num_threads(1)
    ramod = build("ramod", cfg).eval(); ramod.load_state_dict(torch.load("checkpoints/ramod_s0.pt"))
    base = build("rad_cnn", cfg).eval(); base.load_state_dict(torch.load("checkpoints/rad_cnn_s0.pt"))
    T = ramod.T
    sim, rng, used_seed = pick_episode(a.seed, a.frames + T, a.min_ego)
    cubes, states = [], []
    for k in range(a.frames + T - 1):
        st = W.frame_state(sim, k)
        states.append(st)
        cubes.append((W.render_cube(st, rng) - CUBE_MEAN) / CUBE_STD)
    cubes = np.stack(cubes).astype(np.float32)

    hist = dict(v_true=[], v_hat=[], err_r=[], err_b=[], pw=[])
    frames = []
    for f in range(a.frames):
        k = f + T - 1
        st = states[k]
        gts = W.labels(st)
        x = torch.from_numpy(cubes[k - T + 1:k + 1])[None]
        with torch.no_grad():
            t0 = time.perf_counter(); o = ramod(x); ms = 1000 * (time.perf_counter() - t0)
            ob = base(x[:, -1:])
        dr = decode(o, ramod.vmax, thresh=0.3)[0]
        db = decode(ob, base.vmax, thresh=0.3, wrap_head=False)[0]
        v_hat = float(o["v_e"][0, -1])
        raw = cubes[k] * CUBE_STD + CUBE_MEAN                                     # (R, A, D) log power
        M = o["mask"][0, -1].numpy()                                              # (A, D)
        g = torch.sigmoid(ramod.sup.gamma_logit).item()
        sup = np.log1p(np.expm1(raw) * (1 - g * M[None]))

        # ---- match detections to GT (nearest centre within 2.5 bins) for the telemetry error traces
        def match(dets):
            errs = []
            for gt in gts:
                best = min(dets, key=lambda d: np.hypot(d["center"][0] - gt["rb"], d["center"][1] - gt["ab"]),
                           default=None)
                if best and np.hypot(best["center"][0] - gt["rb"], best["center"][1] - gt["ab"]) < 2.5:
                    errs.append(abs(best["v_rad"] - gt["v_rad"]))
            return float(np.mean(errs)) if errs else np.nan
        hist["v_true"].append(st["v_ego"]); hist["v_hat"].append(v_hat)
        hist["err_r"].append(match(dr)); hist["err_b"].append(match(db))
        hist["pw"].append(max([d["p_wrap"] for d in dr], default=0.0))

        canvas = np.full((720, 1280, 3), 22, np.uint8)
        PW, PH = 420, 300
        # (1) raw RA + GT
        p1 = ra_panel(raw.max(-1), PW, PH)
        for gt in gts:
            c0, c1 = box_to_px(gt["box"], PW, PH)
            cv2.rectangle(p1, c0, c1, C_GT, 2)
        put(p1, "INPUT raw range-azimuth (max over Doppler) + GT", (6, 16))
        put(p1, "range up / azimuth  L <-> R", (6, PH - 8), 0.36, (170, 170, 170))
        canvas[10:10 + PH, 6:6 + PW] = p1
        # (2) azimuth-Doppler + predicted static curve
        ad = raw.mean(0)                                                          # (A, D)
        p2 = heat_img(ad[::-1].T[::-1], (PW, PH))                                 # Doppler up, azimuth L->R
        vs = W.wrap_v(-v_hat * np.sqrt(1 - W.S_CENTERS[::-1] ** 2))
        pts = np.stack([(np.arange(W.A_BINS) + 0.5) * PW / W.A_BINS,
                        (W.V_MAX - vs) / (2 * W.V_MAX) * PH], 1)
        for i in range(len(pts) - 1):
            if abs(vs[i] - vs[i + 1]) < W.V_MAX:                                  # don't draw across the wrap seam
                cv2.line(p2, tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int)), (255, 255, 0), 2, cv2.LINE_AA)
        put(p2, "azimuth-Doppler + predicted STATIC curve (cyan)", (6, 16))
        put(p2, f"ego v true {st['v_ego']:.2f}  est {v_hat:.2f} m/s   v_max = {W.V_MAX:.0f}", (6, 34), 0.4,
            (255, 255, 0) if st["v_ego"] > W.V_MAX else (235, 235, 235))
        if st["v_ego"] > W.V_MAX:
            put(p2, "static clutter is WRAPPED", (6, 52), 0.4, (60, 60, 255))
        put(p2, "+v_max", (PW - 52, 14), 0.36); put(p2, "-v_max", (PW - 52, PH - 6), 0.36)
        canvas[10:10 + PH, 432:432 + PW] = p2
        # (3) suppressed RA + detections
        p3 = ra_panel(sup.max(-1), PW, PH)
        for gt in gts:
            c0, c1 = box_to_px(gt["box"], PW, PH)
            cv2.rectangle(p3, c0, c1, C_GT, 1)
        for d in dr:
            c0, c1 = box_to_px(d["box"], PW, PH)
            cv2.rectangle(p3, c0, c1, CLS_COL[d["cls"]], 2)
            put(p3, f"{CLS_SHORT[d['cls']]} {d['score']:.2f} v_r {d['v_rad']:+.1f} pw {d['p_wrap']:.2f}",
                (max(2, c0[0]), max(30, c0[1] - 4)), 0.36, CLS_COL[d["cls"]])
        put(p3, "AFTER static suppression + RaMOD detections", (6, 16))
        canvas[10:10 + PH, 858:858 + PW] = p3

        # (4) BEV
        bev = np.full((400, 640, 3), 30, np.uint8)
        fan = [bev_px([0, 0])] + [bev_px([W.R_MAX * np.cos(t), W.R_MAX * np.sin(t)]) for t in np.linspace(-1.047, 1.047, 30)]
        cv2.fillPoly(bev, [np.array(fan)], (45, 45, 52))
        sp = st["p"][:len(sim["static_xy"])]
        for q in bev_px(sp[(sp[:, 0] > 0) & (np.linalg.norm(sp, axis=1) < W.R_MAX)]):
            cv2.circle(bev, tuple(q), 1, (120, 120, 120), -1)
        for o_ in st["objs"]:
            hull = cv2.convexHull(bev_px(o_["pts"]))
            cv2.polylines(bev, [hull], True, C_GT, 2)
        def arrow(d, col, dy=0):
            c = ra_to_xy(*d["center"])
            u = c / (np.linalg.norm(c) + 1e-6)
            p0 = bev_px(c); p1_ = bev_px(c + u * d["v_rad"] * 0.6)
            cv2.arrowedLine(bev, tuple(p0 + [0, dy]), tuple(p1_ + [0, dy]), col, 2, tipLength=0.25)
        for d in db:
            arrow(d, C_BASE, 3)
        for d in dr:
            c = ra_to_xy(*d["center"])
            cv2.circle(bev, tuple(bev_px(c)), 5, CLS_COL[d["cls"]], -1)
            arrow(d, C_RAMOD)
        cv2.rectangle(bev, (314, 385), (326, 399), (230, 160, 60), -1)
        put(bev, "BEV  green=GT  dots=RaMOD  arrows=radial vel: cyan RaMOD (unwrapped) / orange RAD-CNN (raw)", (6, 16), 0.38)
        put(bev, f"t = {k * W.DT:4.1f} s   RaMOD CPU {ms:4.1f} ms (1 thread)", (6, 392), 0.4)
        canvas[315:715, 6:646] = bev

        # (5) telemetry
        tel = canvas[315:715, 652:1274]
        line_plot(tel, 4, 4, 614, 120, [(hist["v_true"], (200, 200, 200), "ego v true"),
                                        (hist["v_hat"], C_RAMOD, "ego v RaMOD")], 0, 9, "ego speed (m/s)  -- dashed = v_max")
        yv = int(4 + 120 - (W.V_MAX / 9) * 98)
        for xx in range(4, 618, 12):
            cv2.line(tel, (xx, yv), (xx + 6, yv), (90, 90, 255), 1)
        line_plot(tel, 4, 132, 614, 120, [(hist["err_r"], C_RAMOD, "RaMOD"), (hist["err_b"], C_BASE, "RAD-CNN")],
                  0, 10, "|radial velocity error| per frame (m/s)")
        line_plot(tel, 4, 260, 614, 100, [(hist["pw"], (60, 60, 255), "max p_wrap")], 0, 1,
                  "wrap probability (highest among detections)")
        put(tel, "SYNTHETIC toy radar world - not BRIDGE-Liv, not the authors' code", (4, 392), 0.4, (160, 160, 160))
        frames.append(canvas)
        if a.live:
            cv2.imshow("RaMOD", canvas)
            if cv2.waitKey(1) == 27:
                break

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rgb = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in frames]
    imageio.mimsave(a.out, [cv2.resize(fr, (960, 540), interpolation=cv2.INTER_AREA) for fr in rgb[::2]],
                    duration=0.2, loop=0)
    if a.mp4:
        imageio.mimsave(a.mp4, rgb, fps=10, codec="libx264", quality=7, macro_block_size=8)
    er, eb = np.nanmean(hist["err_r"]), np.nanmean(hist["err_b"])
    print(f"episode seed {used_seed}: {len(frames)} frames -> {a.out} {a.mp4}; mean |v_r err| RaMOD {er:.2f} "
          f"RAD-CNN {eb:.2f} m/s; ego MAE {np.mean(np.abs(np.array(hist['v_true']) - hist['v_hat'])):.2f} m/s")


if __name__ == "__main__":
    main()
