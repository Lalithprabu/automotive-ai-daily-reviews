"""Real-time synthetic "AI in action" simulation: DecayFormer forecasting a cell's whole life from its first cycles.

Two held-out cells (their aging condition was never seen in training) are streamed cycle by cycle:
    cycles 1-100 : the cycler feed arrives (V vs SOC curve, SOC-interval dQ/dV bars); from cycle 20 on, every 5 cycles
                   DecayFormer, the ~5x larger Transformer and a sqrt-law fit re-forecast the FULL SOH trajectory
    fast-forward : the real cell keeps aging to EOL; the true curve (green) is revealed against the frozen
                   cycle-100 forecasts
Cell choice is not cherry-picked: cell A = the knee-dominated test cell with the MEDIAN DecayFormer lifetime error,
cell B = the non-knee test cell with the median error.

Layout 1280 x 720, dark UI:
  top-left  (836 x 300)  CYCLER FEED: discharge V(SOC) curves (current bright, earlier faded) + dQ/dV per SOC bin
  low-left  (836 x 300)  SOH TRAJECTORY: grey = measured capacity (input), green dashed = truth,
                         blue = DecayFormer, red = Transformer (free head), yellow dotted = sqrt fit; EOL markers
  right     (430 px)     (1) master curve in normalised lifetime u = n / L  (2) attention over cycles + retention
                         gate  (3) held-out study numbers
  bottom                 legend + SYNTHETIC disclaimer

    python simulate.py --out assets/decayformer_simulation.gif --mp4 assets/decayformer_simulation.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from dfm.baselines import sqrt_fit
from dfm.data import CRATES, N_MAX, NB, conditions, cycle_features, make_cell, soh_curve
from evaluate import load_model, predict
from train import CFG, load_data, split

ROOT = Path(__file__).resolve().parent
TXT, GRID = (235, 235, 235), (52, 52, 52)
C = dict(gt=(90, 220, 90), df=(255, 160, 40), tf=(60, 60, 235), fit=(60, 220, 230), obs=(175, 175, 175))  # BGR
K0, STEP, KMAX, FF, HOLD = 20, 5, 100, 40, 22


def put(img, txt, org, scale=0.45, col=TXT, th=1):
    cv2.putText(img, txt, (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def poly(img, pts, col, th=2):
    pts = np.asarray(pts)
    pts = pts[np.isfinite(pts).all(1)]
    if len(pts) > 1:
        cv2.polylines(img, [pts.astype(np.int32)], False, col, th, cv2.LINE_AA)


def dashed(img, pts, col, th=1, seg=7, gap=True):
    pts = np.asarray(pts, float)
    pts = pts[np.isfinite(pts).all(1)]
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        n = max(1, int(np.linalg.norm(b - a) / seg))
        for j in range(0, n, 2 if gap else 1):
            u, v = a + (b - a) * j / n, a + (b - a) * min(n, j + 1) / n
            cv2.line(img, tuple(u.astype(int)), tuple(v.astype(int)), col, th, cv2.LINE_AA)


def dotted(img, pts, col, every=6):
    pts = np.asarray(pts)
    pts = pts[np.isfinite(pts).all(1)]
    for q in pts[::every]:
        cv2.circle(img, tuple(q.astype(int)), 1, col, -1, cv2.LINE_AA)


def pick_cells(d: dict, P: np.lib.npyio.NpzFile) -> list[int]:
    err = np.abs(P["decayformer_life"] - d["life"]) / d["life"]
    kd = P["knee_frac"] > 0.25
    out = []
    for mask in (kd, ~kd):
        idx = np.nonzero(mask)[0]
        out.append(int(idx[np.argsort(err[idx])[len(idx) // 2]]))
    return out


def run_cell(i: int, d: dict, S: dict, label: str, frames: list):
    fold = int([f for f in range(CFG["data"]["n_folds"]) if i in split(d, f)[1]][0])
    ci = int(d["cond"][i]); kk = i % CFG["data"]["cells_per_cond"]
    cs = CFG["data"]["seed"] * 100_003 + ci * 101 + kk
    cell = make_cell(conditions()[ci], cs)
    feats, curves = cycle_features(cell, KMAX, np.random.default_rng(cs + 7))
    T, Cr, D = cell["cond"]
    life_t = float(d["life"][i])
    grid = np.arange(0, N_MAX + 1, 10, dtype=np.float32)
    truth = soh_curve(cell, grid.astype(float))
    truth[grid > life_t] = np.nan
    m_df, m_tf = load_model("decayformer", fold), load_model("transformer", fold)
    x = d["x"][i:i + 1]
    F = {}
    for k in range(K0, KMAX + 1, STEP):
        p, L, out = predict(m_df, "decayformer", x, k, grid)
        pt, Lt, _ = predict(m_tf, "transformer", x, k, grid)
        pf, Lf = sqrt_fit(x, k, grid)
        F[k] = dict(df=p[0], dfL=float(L[0]), tf=pt[0], tfL=float(Lt[0]), fit=pf[0], fitL=float(Lf[0]),
                    G=out["G"][0].numpy(), attn=out["attn"][0].mean(0).numpy(), gate=out["gates"][0].mean(-1).numpy())
    soc = np.linspace(0, 1, 50)
    n_frames = KMAX + FF + HOLD
    meas = feats[:, 0]
    for fr in range(n_frames):
        k = min(fr + 1, KMAX)                                          # cycles observed so far
        ff = max(0, fr + 1 - KMAX)                                     # fast-forward progress
        kf = (k // STEP) * STEP if k >= K0 else None
        Fk = F.get(kf) if kf else None
        cvs = np.full((720, 1280, 3), 22, np.uint8)
        # ======================================================== cycler feed
        W_, H_ = 836, 300
        A = np.full((H_, W_, 3), 30, np.uint8)
        x0, x1, y0, y1 = 50, 560, 30, 200
        vx = lambda s: x0 + s * (x1 - x0)
        vy = lambda v: y1 - (v - 2.7) / (4.3 - 2.7) * (y1 - y0)
        for b in range(NB + 1):
            cv2.line(A, (int(vx(b / NB)), y0), (int(vx(b / NB)), y1 + 70), GRID, 1)
        for vv in (3.0, 3.5, 4.0):
            put(A, f"{vv:.1f}V", (8, vy(vv) + 4), 0.35, (140, 140, 140))
        for j in range(0, k - 1, 10):
            poly(A, np.stack([vx(soc), vy(curves[j])], -1), (70, 70, 70), 1)
        poly(A, np.stack([vx(soc), vy(curves[k - 1])], -1), (240, 240, 240), 2)
        dq = d["x"][i, k - 1, 5:5 + NB] * 0.6 + 1.2
        dq0 = d["x"][i, 0, 5:5 + NB] * 0.6 + 1.2
        for b in range(NB):
            bx0, bx1 = int(vx(b / NB)) + 6, int(vx((b + 1) / NB)) - 6
            h = int(np.clip(dq[b] / 3.0, 0, 1) * 60)
            h0 = int(np.clip(dq0[b] / 3.0, 0, 1) * 60)
            cv2.rectangle(A, (bx0, y1 + 70 - h), (bx1, y1 + 70), (150, 110, 60), -1)
            cv2.line(A, (bx0, y1 + 70 - h0), (bx1, y1 + 70 - h0), (200, 200, 200), 1)
        put(A, "SOC-interval dQ/dV (bars; white tick = cycle 1)", (x0, y1 + 86), 0.38, (170, 170, 170))
        put(A, "SOC 0 -> 1", (x1 - 70, y0 - 6), 0.35, (140, 140, 140))
        put(A, f"CYCLER FEED - {label}", (8, 18), 0.48)
        rx = 590
        put(A, f"condition  {T:.0f} C | charge {Cr:g}C | DoD {D:.0%}", (rx, 50), 0.42)
        put(A, "(never seen in training)", (rx, 70), 0.38, (150, 150, 150))
        put(A, f"cycle      {k if ff == 0 else KMAX + int(ff / FF * (life_t - KMAX)) if ff <= FF else int(life_t)}",
            (rx, 105), 0.5, TXT, 2)
        if ff == 0:
            put(A, f"capacity   {meas[k - 1] * 100:.2f} %", (rx, 130), 0.45)
        else:
            n_ff = KMAX + min(ff / FF, 1.0) * (life_t - KMAX)
            put(A, f"true SOH   {soh_curve(cell, np.array([n_ff]))[0] * 100:.2f} %", (rx, 130), 0.45, C["gt"])
        put(A, f"resistance {feats[k - 1, 1] * 1000:.1f} mOhm" + ("  (cycle 100)" if ff else ""), (rx, 152), 0.45)
        put(A, f"temp rise  {feats[k - 1, 4]:.1f} C" + ("  (cycle 100)" if ff else ""), (rx, 174), 0.45)
        if ff > 0:
            put(A, "FAST-FORWARD to end of life", (rx, 210), 0.5, C["gt"], 2)
            put(A, "(model input frozen at 100 cycles)", (rx, 232), 0.38, (150, 150, 150))
        cvs[4:4 + H_, 4:4 + W_] = A
        # ======================================================== SOH trajectory
        B_ = np.full((H_, W_, 3), 30, np.uint8)
        px0, px1, py0, py1 = 60, 820, 28, 270
        X = lambda n: px0 + n / N_MAX * (px1 - px0)
        Y = lambda s: py1 - (s - 0.76) / (1.005 - 0.76) * (py1 - py0)
        for s_ in (0.8, 0.9, 1.0):
            cv2.line(B_, (px0, int(Y(s_))), (px1, int(Y(s_))), (90, 90, 90) if s_ == 0.8 else GRID, 1)
            put(B_, f"{s_:.1f}", (22, Y(s_) + 4), 0.38, (150, 150, 150))
        put(B_, "EOL", (px1 - 34, Y(0.8) - 5), 0.38, (150, 150, 150))
        for n_ in range(0, N_MAX + 1, 500):
            put(B_, f"{n_}", (X(n_) - 10, py1 + 18), 0.35, (130, 130, 130))
        cv2.rectangle(B_, (int(X(0)), py0), (int(X(k)), py1), (45, 45, 45), -1)
        tp = np.stack([X(grid), Y(truth)], -1)
        if ff == 0:
            dashed(B_, tp, (60, 110, 60), 1, 6)
        else:
            reveal = KMAX + min(ff / FF, 1.0) * (life_t - KMAX)
            dashed(B_, tp[grid <= reveal], C["gt"], 2, 7)
            dashed(B_, tp[grid > reveal], (60, 110, 60), 1, 6)
        if Fk is not None:
            cut = lambda key: grid <= Fk[key] + 10                     # forecasts are drawn up to their own EOL
            dotted(B_, np.stack([X(grid), Y(Fk["fit"])], -1)[cut("fitL")], C["fit"], 3)
            poly(B_, np.stack([X(grid), Y(Fk["tf"])], -1)[cut("tfL")], C["tf"], 2)
            poly(B_, np.stack([X(grid), Y(Fk["df"])], -1)[cut("dfL")], C["df"], 3)
            for key, col, dy in (("fitL", C["fit"], 0), ("tfL", C["tf"], 8), ("dfL", C["df"], 16)):
                xx = int(X(min(Fk[key], N_MAX)))
                cv2.drawMarker(B_, (xx, int(Y(0.8)) + 10 + dy), col, cv2.MARKER_TRIANGLE_UP, 9, 2)
        ns = np.arange(1, k + 1)
        for n_, q in zip(ns[::2], meas[:k:2]):                         # input on top of the forecasts
            cv2.circle(B_, (int(X(n_)), int(Y(q))), 2, C["obs"], -1)
        if ff > 0:
            cv2.drawMarker(B_, (int(X(life_t)), int(Y(0.8))), C["gt"], cv2.MARKER_DIAMOND, 12, 2)
        put(B_, "SOH TRAJECTORY (cycles 0-3000) - shaded = observed window", (8, 18), 0.48)
        if Fk is not None:
            yy = 44
            for key, nm, col in (("dfL", "DecayFormer", C["df"]), ("tfL", "Transformer", C["tf"]), ("fitL", "sqrt fit", C["fit"])):
                txt = f"{nm:<12} EOL {Fk[key]:5.0f}"
                if ff >= FF:
                    txt += f"  ({(Fk[key] - life_t) / life_t * 100:+.0f}%)"
                put(B_, txt, (560, yy), 0.42, col, 1); yy += 18
            if ff >= FF:
                put(B_, f"{'truth':<12} EOL {life_t:5.0f}", (560, yy), 0.42, C["gt"], 2)
        else:
            put(B_, f"waiting for {K0} cycles before the first forecast", (300, 150), 0.5, (170, 170, 170))
        cvs[310:310 + H_, 4:4 + W_] = B_
        # ======================================================== right column
        R = cvs[4:610, 846:1276]
        w = 428
        h1 = 200
        cv2.rectangle(R, (1, 1), (w, h1), (70, 70, 70), 1)
        put(R, "master curve: SOH vs normalised life u = n / L", (6, 16), 0.42)
        ux = lambda u: 30 + u / 1.2 * (w - 50)
        uy = lambda s: h1 - 20 - (s - 0.76) / (1.005 - 0.76) * (h1 - 50)
        cv2.line(R, (int(ux(1.0)), 26), (int(ux(1.0)), h1 - 18), (90, 90, 90), 1)
        cv2.line(R, (24, int(uy(0.8))), (w - 10, int(uy(0.8))), (90, 90, 90), 1)
        put(R, "u=1", (ux(1.0) - 12, h1 - 6), 0.35, (150, 150, 150))
        put(R, "0.8", (2, uy(0.8) + 4), 0.35, (150, 150, 150))
        ut = grid / life_t
        dashed(R, np.stack([ux(ut[ut <= 1.2]), uy(truth[ut <= 1.2])], -1), C["gt"], 2, 6)
        if Fk is not None:
            uk = np.linspace(0, 1, len(Fk["G"]))
            pts = np.stack([ux(uk), uy(1 - 0.2 * Fk["G"])], -1)
            poly(R, pts, C["df"], 2)
            for q in pts:
                cv2.circle(R, tuple(q.astype(int)), 2, C["df"], -1, cv2.LINE_AA)
            put(R, "16 knots, softmax steps -> monotone by construction", (6, 34), 0.36, (160, 160, 160))
        # attention + gate
        y2, h2 = 206, 170
        cv2.rectangle(R, (1, y2), (w, y2 + h2), (70, 70, 70), 1)
        put(R, "where it looks: state-query attention over cycles", (6, y2 + 16), 0.42)
        if Fk is not None:
            a = Fk["attn"]
            am = a.max() + 1e-9
            for n_ in range(KMAX):
                xx = 10 + n_ * (w - 20) / KMAX
                hh = a[n_] / am * 80
                cv2.rectangle(R, (int(xx), int(y2 + 120 - hh)), (int(xx + 2), y2 + 120), C["df"], -1)
            g = Fk["gate"]
            poly(R, [(10 + n_ * (w - 20) / KMAX, y2 + 160 - g[n_] * 34) for n_ in range(kf)], (200, 200, 200), 1)
            put(R, "bars = attention | white = mean retention gate a_t", (6, y2 + 34), 0.36, (160, 160, 160))
        put(R, "cycle 1", (6, y2 + 136), 0.33, (130, 130, 130))
        put(R, "100", (w - 30, y2 + 136), 0.33, (130, 130, 130))
        # study
        y3 = 382
        cv2.rectangle(R, (1, y3), (w, 604), (70, 70, 70), 1)
        mt = S["metrics"]
        put(R, f"held-out study: {S['n_cells']} cells, {S['n_conditions']} conditions, 6 folds", (6, y3 + 16), 0.42)
        put(R, "                  params  traj MAPE  life MAPE", (6, y3 + 38), 0.38, (170, 170, 170))
        yy = y3 + 56
        for v, nm, col in (("decayformer", "DecayFormer", C["df"]), ("df_freehead", "  free head", (200, 200, 200)),
                           ("df_nosoc", "  no SOC feats", (200, 200, 200)), ("df_noattn", "  no attention", (200, 200, 200)),
                           ("transformer", "Transformer", C["tf"]), ("gru", "GRU", (200, 200, 200)),
                           ("sqrt_fit", "sqrt fit", C["fit"])):
            p = f"{S['params'][v] / 1000:.0f}k" if v in S["params"] else "-"
            put(R, f"{nm:<16} {p:>6}   {mt[v]['traj']:5.2f}%   {mt[v]['life']:5.1f}%", (6, yy), 0.38, col); yy += 18
        put(R, "paper (BatteryLife Li-ion): 2.25% MAPE", (6, yy + 6), 0.38, (150, 150, 150))
        # bottom
        put(cvs, "grey dots = measured capacity (input) | green dashed = true SOH | blue = DecayFormer | "
                 "red = Transformer (5x params, free head) | yellow dots = sqrt-law fit | triangles = predicted EOL",
            (8, 636), 0.38, (200, 200, 200))
        put(cvs, f"{label}: test cell #{i}, fold {fold}; picked as the MEDIAN-error cell of its group, not the best",
            (8, 660), 0.4, (180, 180, 180))
        put(cvs, "SYNTHETIC aging simulator, my re-implementation from the abstract of arXiv 2611.19116 - "
                 "not BatteryLife, not the authors' model or code", (8, 702), 0.4, (150, 150, 150))
        frames.append(cvs)
    return dict(cell=i, fold=fold, life=life_t, df=F[KMAX]["dfL"], tf=F[KMAX]["tfL"], fit=F[KMAX]["fitL"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/decayformer_simulation.gif")
    ap.add_argument("--mp4", default="")
    ap.add_argument("--gif_every", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(2)
    d = load_data()
    S = json.load(open(ROOT / "results/summary.json"))
    P = np.load(ROOT / "results/predictions_k100.npz")
    ca, cb = pick_cells(d, P)
    frames = []
    title = np.full((720, 1280, 3), 22, np.uint8)
    put(title, "Day 77 - A battery's whole life from its first 100 cycles", (210, 290), 1.0, TXT, 2)
    put(title, "DecayFormer: selective recurrence + state-query attention + master-curve head", (175, 340), 0.72, C["df"], 2)
    put(title, "SYNTHETIC demo - toy re-implementation of arXiv 2611.19116", (330, 390), 0.7, (160, 160, 160), 1)
    frames += [title] * 15
    info = [run_cell(ca, d, S, "cell A (knee-dominated)", frames), run_cell(cb, d, S, "cell B (no strong knee)", frames)]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave(out, [cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA) for f in rgb[::a.gif_every]],
                    duration=0.1 * a.gif_every, loop=0)
    if a.mp4:
        imageio.mimsave(ROOT / a.mp4, rgb, fps=10, codec="libx264", quality=7, macro_block_size=8)
    cv2.imwrite(str(out.with_name("still.png")), frames[15 + KMAX + FF + 5])
    print(f"{len(frames)} frames -> {out}")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
