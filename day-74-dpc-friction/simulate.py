"""Real-time synthetic "AI in action" simulation for DPC under varying friction (OpenCV).

Layout 1280 x 720, dark UI, mixed-friction track (blue = wet sectors, mu = 0.5), same held-out seed for all cars.
  top-left      track overview: three cars, wet sectors, the zoom window
  main panel    "unrolled" chase view in Frenet coordinates (arc length s horizontal, lateral offset e_y vertical,
                lateral axis exaggerated x12 so centimetre errors are visible):
      INPUT         grey dots = the T_ini = 5 measured samples DPC conditions on (initial trajectory)
      GROUND TRUTH  green dashed = where the pooled-DPC car REALLY goes over the next 0.75 s (hindsight)
      PREDICTION    from the SAME initial trajectory and the SAME planned steering, three Hankel predictors:
                    cyan = matched data (knows the friction), orange = dry-only data, magenta = pooled data
      TRACES        solid lines = driven paths of the three closed-loop cars
                    (orange DPC dry-only data, magenta DPC pooled data, cyan DPC oracle-matched data)
  right column  (1) live multi-step prediction error per horizon step for the three predictors
                (2) |e_y| over time for the three cars, friction band underneath   (3) steering
  bottom strip  legend + held-out study numbers

    python simulate.py --out assets/dpc_simulation.gif --mp4 assets/dpc_simulation.mp4
SYNTHETIC toy world - not the authors' simulator, vehicles or code.
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

from dpcf import vehicle as V
from dpcf.control import BoxQP, DPCController
from dpcf.episode import run
from evaluate import build, tracks

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
BG, TXT = (22, 22, 22), (235, 235, 235)
C_GT, C_HI, C_POOL, C_MATCH, C_WET = (90, 220, 90), (0, 150, 255), (255, 80, 220), (255, 215, 0), (68, 42, 22)
CARS = [("dpc_high", "DPC, dry-only data", C_HI), ("dpc_pool", "DPC, pooled data", C_POOL),
        ("dpc_oracle", "DPC, matched data (oracle)", C_MATCH)]


def put(img, txt, org, scale=0.45, col=TXT, th=1, bg=True):
    (w, h), b = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = int(org[0]), int(org[1])
    if bg:
        cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + b), BG, -1)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def dashed(img, P, col, th=1, seg=7):
    P = np.asarray(P, float)
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        n = max(1, int(np.linalg.norm(b - a) / seg))
        for j in range(0, n, 2):
            u, v = a + (b - a) * j / n, a + (b - a) * min(n, j + 1) / n
            cv2.line(img, tuple(u.astype(int)), tuple(v.astype(int)), col, th, cv2.LINE_AA)


class Overview:
    def __init__(self, tr, W=420, H=250):
        self.tr, self.W, self.H = tr, W, H
        xs, ys = tr.x, tr.y
        self.sc = min((W - 30) / np.ptp(xs), (H - 40) / np.ptp(ys))
        self.cx, self.cy = xs.mean(), ys.mean()

    def px(self, x, y):
        return np.stack([self.W / 2 + (np.asarray(x) - self.cx) * self.sc,
                         self.H / 2 + 10 - (np.asarray(y) - self.cy) * self.sc], -1).astype(np.int32)

    def draw(self, cars_xy, s_now):
        img = np.full((self.H, self.W, 3), 30, np.uint8)
        tr = self.tr
        P = self.px(tr.x, tr.y)
        wet = tr.mu < 0.5 * (tr.mu_high + tr.mu_low)
        for i in range(0, len(P) - 1, 3):
            cv2.line(img, tuple(P[i]), tuple(P[i + 3 if i + 3 < len(P) else 0]),
                     (230, 140, 60) if wet[i] else (150, 150, 150), 9 if wet[i] else 7)
        i0 = tr.idx(s_now)
        win = [tr.idx(s_now + d) for d in np.linspace(-1.5, 3.0, 30)]
        cv2.polylines(img, [P[win]], False, (255, 255, 255), 2)
        for (x, y), col in cars_xy:
            cv2.circle(img, tuple(self.px(x, y)), 5, col, -1, cv2.LINE_AA)
        put(img, "track overview (blue = wet, mu 0.5)", (6, 16), 0.42)
        put(img, "white = zoom window", (6, self.H - 8), 0.36, (180, 180, 180))
        return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", default="assets/dpc_simulation.gif")
    ap.add_argument("--mp4", default="")
    ap.add_argument("--laps", type=float, default=2.0)
    a = ap.parse_args()
    P, _ = build(a.seed)
    tr = tracks()["mixed"]
    q = CFG["qp"]; N = CFG["dpc"]["N"]; Ti = CFG["dpc"]["T_ini"]
    qp = BoxQP(N, q=(q["q_ey"], q["q_epsi"]), r=q["r"], r_d=q["r_d"])
    thr = 0.5 * (tr.mu_high + tr.mu_low)
    logs = {}
    for name, _, _ in CARS:
        sel = (lambda mu: "high" if mu > thr else "low") if name == "dpc_oracle" else name.split("_")[1]
        logs[name] = run(tr, DPCController(P, sel, qp), "dpc", seed=a.seed, laps=a.laps, log=True, Ti=Ti, N=N)
        print(name, round(logs[name]["rms_ey"] * 100, 2), "cm")
    pool = logs["dpc_pool"]["frames"]
    summ = json.load(open(ROOT / "results/summary.json"))
    ov = Overview(tr)
    # main panel geometry
    MW, MH = 836, 330
    EY = 12.0 * 100                                     # px per m lateral (x12 exaggeration vs longitudinal)
    SXp = MW / 4.5                                      # px per m along s (window -1.5 .. +3.0 m)
    errs_hist = {n: [] for n, _, _ in CARS}
    frames = []
    L = min(len(logs[n]["frames"]) for n, _, _ in CARS)
    still = None
    for k in range(Ti + 1, L - N - 1):
        f = pool[k]
        s0 = f["s"]
        cv = np.full((720, 1280, 3), 22, np.uint8)
        # ---------------- overview
        cars_xy = [((logs[n]["frames"][k]["state"][0], logs[n]["frames"][k]["state"][1]), c) for n, _, c in CARS]
        cv[4:254, 4:424] = ov.draw(cars_xy, s0)
        # ---------------- main Frenet panel
        M = np.full((MH, MW, 3), 34, np.uint8)
        X = lambda s: (MW * 1.5 / 4.5 + (np.asarray(s) - s0) * SXp)
        Y = lambda e: (MH / 2 - np.asarray(e) * EY)
        for ss in np.arange(np.floor(s0 - 1.5), s0 + 3.1, 0.05):    # wet tint
            if tr.mu[tr.idx(ss)] < thr:
                x0 = int(X(ss)); cv2.rectangle(M, (x0, 0), (x0 + int(0.05 * SXp) + 1, MH), C_WET, -1)
        for e in (-0.10, -0.05, 0.05, 0.10):
            cv2.line(M, (0, int(Y(e))), (MW, int(Y(e))), (60, 60, 60), 1)
            put(M, f"{int(e * 100):+d} cm", (4, int(Y(e)) - 3), 0.34, (150, 150, 150), bg=False)
        dashed(M, [(0, MH / 2), (MW, MH / 2)], (200, 200, 200), 1, 10)
        put(M, "racing line", (MW - 90, int(MH / 2) - 5), 0.36, (200, 200, 200), bg=False)
        # traces of the three cars (past 1.5 m)
        for n, _, col in CARS:
            fr = logs[n]["frames"]
            pts = [(X(g["s"]), Y(g["ey"])) for g in fr[max(0, k - 60):k + 1] if g["s"] > s0 - 1.6]
            if len(pts) > 1:
                cv2.polylines(M, [np.array(pts, np.int32)], False, col, 2, cv2.LINE_AA)
            g = fr[k]
            cv2.circle(M, (int(X(s0)), int(Y(g["ey"]))), 6, col, -1, cv2.LINE_AA)
        # INPUT: T_ini samples (pooled car)
        for j in range(Ti):
            g = pool[k - Ti + j + 1]
            cv2.circle(M, (int(X(g["s"])), int(Y(g["ey"]))), 4, (170, 170, 170), -1, cv2.LINE_AA)
        # GROUND TRUTH future (hindsight)
        fut = [(X(pool[k + j]["s"]), Y(pool[k + j]["ey"])) for j in range(1, N + 1)]
        dashed(M, [(X(s0), Y(f["ey"]))] + fut, C_GT, 2)
        # PREDICTIONS from the same u_ini / y_ini / planned steering
        u_ini = torch.tensor(f["u_ini"]); y_ini = torch.tensor(f["y_ini"])
        u_f = torch.tensor(np.stack([f["plan"], f["k_in"]], -1))
        match_key = "high" if f["mu"] > thr else "low"
        preds = {}
        for key, col in (("pool", C_POOL), ("high", C_HI), (match_key + "_m", C_MATCH)):
            pk = key.replace("_m", "")
            yp = P[pk](u_ini[None], y_ini[None], u_f[None])[0, :, 0].numpy()
            preds[key] = yp
            pts = [(X(s0), Y(f["ey"]))] + [(X(pool[k + j + 1]["s"]), Y(yp[j])) for j in range(N)]
            cv2.polylines(M, [np.array(pts, np.int32)], False, col, 1 if key != "pool" else 2, cv2.LINE_AA)
            for j in (4, 9, 14):
                cv2.circle(M, tuple(np.array(pts[j + 1], int)), 3, col, -1, cv2.LINE_AA)
        put(M, "unrolled chase view: s along track (-1.5 .. +3 m), lateral offset x12", (6, 16), 0.45)
        mu_txt = f"mu here = {f['mu']:.1f} ({'WET' if f['mu'] < thr else 'dry'})"
        put(M, mu_txt, (MW - 200, 16), 0.45, (255, 170, 90) if f["mu"] < thr else TXT)
        cv[258:258 + MH, 4:4 + MW] = M
        # ---------------- right telemetry
        R = cv[4:600, 846:1276]
        x0, y0, w, h = 2, 2, 426, 190
        cv2.rectangle(R, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
        put(R, "live prediction error |y_hat - y_true| per step (cm)", (x0 + 4, y0 + 15), 0.4)
        truth = np.array([pool[k + j + 1]["ey"] for j in range(N)])
        bw = (w - 30) / N
        for j in range(N):
            for q_, (key, col) in enumerate((("high", C_HI), ("pool", C_POOL), (match_key + "_m", C_MATCH))):
                e = min(abs(preds[key][j] - truth[j]) * 100, 8.0)
                hb = int(e / 8.0 * (h - 50))
                xb = int(x0 + 14 + j * bw + q_ * bw / 3.3)
                cv2.rectangle(R, (xb, y0 + h - 10 - hb), (int(xb + bw / 3.6), y0 + h - 10), col, -1)
        put(R, "orange dry-only   magenta pooled   cyan matched", (x0 + 4, y0 + 32), 0.36, (200, 200, 200))
        put(R, "0 s", (x0 + 6, y0 + h - 12), 0.32, (150, 150, 150)); put(R, "0.75 s", (x0 + w - 44, y0 + h - 12), 0.32, (150, 150, 150))
        put(R, "8 cm", (x0 + w - 36, y0 + 46), 0.32, (150, 150, 150))
        # |e_y| over time
        y1 = 200
        cv2.rectangle(R, (x0, y1), (x0 + w, y1 + h), (70, 70, 70), 1)
        put(R, "|lateral error| over time (cm)", (x0 + 4, y1 + 15), 0.4)
        tmax = L * V.DT
        for t in range(0, k, 2):                                   # friction band
            if pool[t]["mu"] < thr:
                xt = int(x0 + t * V.DT / tmax * w)
                cv2.line(R, (xt, y1 + h - 8), (xt, y1 + h - 2), (255, 170, 90), 2)
        for n, lab, col in CARS:
            ev = np.array([abs(g["ey"]) * 100 for g in logs[n]["frames"][:k + 1]])
            pts = np.stack([x0 + np.arange(len(ev)) * V.DT / tmax * w,
                            y1 + h - 12 - np.clip(ev, 0, 6) / 6 * (h - 50)], -1).astype(np.int32)
            cv2.polylines(R, [pts], False, col, 1, cv2.LINE_AA)
            rms = np.sqrt((ev[20:] ** 2).mean()) if len(ev) > 20 else np.nan
            put(R, f"{lab}: RMS {rms:.2f} cm", (x0 + 6, y1 + 32 + 14 * CARS.index((n, lab, col))), 0.36, col)
        put(R, "blue ticks = wet sector", (x0 + w - 150, y1 + h - 14), 0.32, (255, 170, 90))
        # steering
        y2 = 398
        cv2.rectangle(R, (x0, y2), (x0 + w, y2 + h), (70, 70, 70), 1)
        put(R, "steering command (rad)", (x0 + 4, y2 + 15), 0.4)
        for n, lab, col in CARS:
            dv = np.array([g["delta"] for g in logs[n]["frames"][:k + 1]])
            pts = np.stack([x0 + np.arange(len(dv)) * V.DT / tmax * w,
                            y2 + h / 2 + 10 - dv / 0.4 * (h / 2 - 25)], -1).astype(np.int32)
            cv2.polylines(R, [pts], False, col, 1, cv2.LINE_AA)
        # ---------------- bottom strip
        sp, sc = summ["prediction"], summ["closedloop"]
        put(cv, f"t = {f['t']:4.1f} s   v = {f['state'][3]:.2f} m/s   seed {a.seed} (held-out)   lap {f['s'] / tr.L:.2f}",
            (430, 22), 0.5)
        put(cv, "grey = 5-sample initial trajectory (input) | green dashed = true next 0.75 s | thin/thick = Hankel "
                "predictions | solid = driven paths", (8, 616), 0.4, (200, 200, 200))
        put(cv, f"prediction RMS (20 held-out seeds): on mu 0.5 matched {sp['test_low']['matched']:.2f} cm | dry-only "
                f"{sp['test_low']['mismatched']:.2f} cm ({sp['test_low']['r_mm']:.1f}x) | pooled {sp['test_low']['pooled']:.2f} cm "
                f"({sp['test_low']['r_p']:.2f}x)", (8, 642), 0.43, C_POOL)
        m = sc["mixed"]
        put(cv, f"mixed track tracking RMS: dry-only {m['dpc_high']['rms']:.2f} | wet-only {m['dpc_low']['rms']:.2f} | pooled "
                f"{m['dpc_pool']['rms']:.2f} | matched {m['dpc_oracle']['rms']:.2f} | model MPC (true mu) "
                f"{m['model_mpc_true_mu']['rms']:.2f} cm", (8, 668), 0.43, C_MATCH)
        put(cv, "SYNTHETIC 1:10 toy world, my re-implementation from the abstract of arXiv 2611.21378 - not the authors' "
                "simulator, CPM Lab cars or code", (8, 694), 0.4, (150, 150, 150))
        if still is None and f["mu"] < thr and k > 40:
            still = len(frames)
        frames.append(cv)
    title = np.full((720, 1280, 3), 22, np.uint8)
    put(title, "Day 74 - Data-driven predictive control under varying friction", (150, 300), 1.0, TXT, 2)
    put(title, "one Hankel matrix of dry + wet driving data, no friction estimator", (230, 350), 0.8, C_POOL, 2)
    put(title, "SYNTHETIC demo - toy re-implementation of arXiv 2611.21378", (300, 400), 0.7, (160, 160, 160), 1)
    frames = [title] * 30 + frames
    still = None if still is None else still + 30
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rgb = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in frames]
    imageio.mimsave(a.out, [cv2.resize(fr, (960, 540), interpolation=cv2.INTER_AREA) for fr in rgb[::2]],
                    duration=0.1, loop=0)
    if a.mp4:
        imageio.mimsave(a.mp4, rgb, fps=20, codec="libx264", quality=7, macro_block_size=8)
    cv2.imwrite(str(Path(a.out).with_name("still.png")), frames[(still or 0) + 25])
    print(f"{len(frames)} frames -> {a.out} {a.mp4}")


if __name__ == "__main__":
    main()
