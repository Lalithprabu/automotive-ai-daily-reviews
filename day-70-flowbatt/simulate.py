"""Real-time synthetic "AI in action" simulation for FlowBatt (OpenCV).

Per featured test cell, three acts:
  ACT 1  INPUT      the early-life capacity matrix fills in, cycle 10 -> 100 (Q(V) curves + amplified Delta-Q(V))
  ACT 2  GENERATE   FlowBatt integrates its learned velocity field: 100 noise vectors flow (t: 0 -> 1) into
                    100 plausible SOH futures; the 5-95 % band and the median form live
  ACT 3  REVEAL     a time cursor sweeps through the cell's real future: ground truth (green) vs FlowBatt
                    band + median (cyan) vs a deterministic Transformer point forecast (orange)

Layout (1280 x 720, dark UI)
  top-left   INPUT capacity curves        top-middle  flow-matching generation      top-right  fleet RUL board
  bottom-left  SOH: prefix | GT | FlowBatt band+median | Transformer      bottom-right  telemetry

    python simulate.py --out assets/flowbatt_simulation.gif --mp4 assets/flowbatt_simulation.mp4
SYNTHETIC toy cell fleet - not MATR / HUST / MIX data, not the authors' code or numbers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from flowbatt import world as W
from flowbatt.data import Normaliser, load_splits, tensors
from train import build

C_GT, C_FB, C_TR, C_PRE, C_EOL = (90, 230, 90), (255, 220, 0), (0, 150, 255), (240, 240, 240), (80, 80, 255)
BG = 22


def put(img, txt, org, scale=0.42, col=(235, 235, 235), th=1):
    (w, h), b = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = org
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + b), (18, 18, 18), -1)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


class Axes:
    """Tiny data->pixel mapper for a rectangle inside an image."""

    def __init__(self, img, x0, y0, w, h, xlim, ylim, pad=(38, 10, 22, 26)):
        self.img, self.xlim, self.ylim = img, xlim, ylim
        l, r, t, b = pad
        self.X0, self.Y0, self.W, self.H = x0 + l, y0 + t, w - l - r, h - t - b
        cv2.rectangle(img, (self.X0, self.Y0), (self.X0 + self.W, self.Y0 + self.H), (70, 70, 70), 1)

    def px(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        u = self.X0 + (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.W
        v = self.Y0 + self.H - (np.clip(y, *self.ylim) - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.H
        return np.stack([u, v], -1).astype(np.int32)

    def line(self, x, y, col, th=1, dashed=False):
        p = self.px(x, y)
        if dashed:
            for i in range(0, len(p) - 1, 2):
                cv2.line(self.img, tuple(p[i]), tuple(p[i + 1]), col, th, cv2.LINE_AA)
        else:
            cv2.polylines(self.img, [p], False, col, th, cv2.LINE_AA)

    def band(self, x, lo, hi, col, alpha=0.35):
        poly = np.concatenate([self.px(x, hi), self.px(x[::-1], lo[::-1])])
        ov = self.img.copy()
        cv2.fillPoly(ov, [poly], col)
        cv2.addWeighted(ov, alpha, self.img, 1 - alpha, 0, self.img)

    def yticks(self, vals, fmt="{:.2f}"):
        for v in vals:
            p = self.px(self.xlim[0], v)
            cv2.line(self.img, tuple(p), (p[0] - 4, p[1]), (120, 120, 120), 1)
            put(self.img, fmt.format(v), (p[0] - 36, p[1] + 4), 0.32, (170, 170, 170))

    def xticks(self, vals, fmt="{:.0f}"):
        for v in vals:
            p = self.px(v, self.ylim[0])
            cv2.line(self.img, tuple(p), (p[0], p[1] + 4), (120, 120, 120), 1)
            put(self.img, fmt.format(v), (p[0] - 12, p[1] + 16), 0.32, (170, 170, 170))


def viridis(f):
    u8 = np.uint8(np.clip(f, 0, 1) * 255)
    return tuple(int(c) for c in cv2.applyColorMap(np.array([[u8]]), cv2.COLORMAP_VIRIDIS)[0, 0])


def pick_cells(te):
    """typical / hidden-defect early failure / long-life, deterministic choice."""
    eol = W.eol_cycle(te["soh_clean"])
    typ = int(np.argmin(np.abs(eol - 900) + 2000 * (np.abs(te["hidden"]) > 0.5)))
    bad = int(np.argmin(te["hidden"] + 5 * (te["chem"] != 1)))                 # NMC with the worst hidden defect
    lng = int(np.argmin(np.abs(eol - 2000) + 2000 * (np.abs(te["hidden"]) > 0.7)))
    return [typ, bad, lng]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="assets/flowbatt_simulation.gif")
    ap.add_argument("--mp4", default="")
    ap.add_argument("--still", default="assets/flowbatt_still.png")
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))
    torch.set_num_threads(2)
    sp = load_splits(cfg)
    norm = Normaliser(sp["train"]["soh"])
    te = sp["test"]
    cap, pre, _ = tensors(te, norm)
    fb = build("flowbatt", cfg).eval(); fb.load_state_dict(torch.load(f"checkpoints/flowbatt_s{a.seed}.pt"))
    tr = build("transformer", cfg).eval(); tr.load_state_dict(torch.load(f"checkpoints/transformer_s{a.seed}.pt"))
    steps = cfg["sample"]["flow_steps"]
    feat = pick_cells(te)
    board = feat + [i for i in range(len(te["soh"])) if i not in feat][:5]       # 8 cells on the RUL board

    # ---- precompute forecasts
    g = torch.Generator().manual_seed(2026)
    fc = {}
    for i in board:
        s, path = fb.sample(cap[i:i + 1], pre[i:i + 1], 100, steps=steps, return_path=True, generator=g)
        soh_s = norm.inv(s[0].numpy())                                          # (100, L)
        with torch.no_grad():
            det = norm.inv(tr.predict(cap[i:i + 1], pre[i:i + 1])[0].numpy())
        eols = W.eol_cycle(soh_s)
        fc[i] = dict(s=soh_s, path=[norm.inv(p[0, :24].numpy()) for p in path], det=det,
                     med=np.median(soh_s, 0), lo=np.percentile(soh_s, 5, 0), hi=np.percentile(soh_s, 95, 0),
                     eols=eols, eol_true=float(W.eol_cycle(te["soh_clean"][i:i + 1])[0]),
                     eol_med=float(W.eol_cycle(np.median(soh_s, 0)[None])[0]),
                     eol_det=float(W.eol_cycle(det[None])[0]))

    frames, tele = [], dict(e_fb=[], e_tr=[], inband=[])
    revealed = set()
    G = W.GRID
    for ci, i in enumerate(feat):
        f = fc[i]
        capm = te["cap"][i]
        for fr in range(50):
            act = 1 if fr < 10 else (2 if fr < 26 else 3)
            k_cyc = min(100, 10 * (fr + 1)) if act == 1 else 100
            t_idx = 0 if act == 1 else (min(steps, 2 * (fr - 10 + 1)) if act == 2 else steps)
            cursor = 100 + (fr - 26) / 23 * (G[-1] - 100) if act == 3 else 100
            cursor = min(cursor, max(f["eol_true"] * 1.25, 600)) if act == 3 else cursor
            if act == 3 and fr == 49:
                revealed.add(i)
            canvas = np.full((720, 1280, 3), BG, np.uint8)

            # ---------------- (1) INPUT capacity curves
            p1 = np.full((300, 420, 3), 28, np.uint8)
            ax = Axes(p1, 0, 0, 420, 300, (0, 1), (0, 1.0))
            for n in range(0, k_cyc, 3):
                ax.line(W.V, capm[n] / capm[0].max(), viridis(n / 100), 1)
            dq = (capm[k_cyc - 1] - capm[0])
            lo_ = min(-0.01, float((capm[99] - capm[0]).min()) * 1.2)
            ax2 = Axes(p1, 230, 150, 190, 130, (0, 1), (lo_, 0.005), pad=(30, 6, 18, 16))
            ax2.line(W.V, dq, (255, 160, 255), 2)
            ax2.line([0, 1], [0, 0], (100, 100, 100), 1)
            put(p1, f"dQ(V)=Q{k_cyc}-Q1", (ax2.X0 + 2, ax2.Y0 + 12), 0.34, (255, 160, 255))
            put(p1, f"INPUT  Q(V), cycles 1..{k_cyc} (viridis = cycle)", (6, 16))
            put(p1, f"cell #{i}  chem {W.CHEM_NAMES[int(te['chem'][i])]}   prefix SOH {te['prefix'][i][-1]:.3f}",
                (6, 292), 0.36, (170, 170, 170))
            canvas[8:308, 6:426] = p1

            # ---------------- (2) flow generation
            p2 = np.full((300, 420, 3), 28, np.uint8)
            ax = Axes(p2, 0, 0, 420, 300, (0, G[-1]), (0.5, 1.05))
            ax.yticks([0.6, 0.8, 1.0]); ax.xticks([0, 1000, 2000])
            if act >= 2:
                xt = f["path"][t_idx]
                for j in range(len(xt)):
                    ax.line(G, xt[j], (200, 170, 60) if t_idx < steps else C_FB, 1)
            ax.line([0, G[-1]], [W.EOL, W.EOL], C_EOL, 1, dashed=True)
            put(p2, "FlowBatt GENERATE: noise -> SOH futures (24 of 100 shown)", (6, 16))
            tcol = C_FB if act >= 2 else (150, 150, 150)
            put(p2, f"flow time t = {t_idx / steps:.2f}   Euler step {t_idx}/{steps}", (44, 40), 0.42, tcol)
            canvas[8:308, 432:852] = p2

            # ---------------- (3) fleet RUL board
            p3 = np.full((300, 420, 3), 28, np.uint8)
            ax = Axes(p3, 0, 0, 420, 300, (0, 2600), (-0.5, len(board) - 0.5), pad=(46, 10, 26, 40))
            ax.xticks([0, 800, 1600, 2400])
            for r, j in enumerate(board):
                y = len(board) - 1 - r
                fj = fc[j]
                show = (j in revealed) or (j == i and act == 3) or (j not in feat)
                gen_ok = (j in revealed) or (j == i and act >= 2 and t_idx == steps) or (j not in feat)
                if j == i:
                    pp = ax.px(0, y); cv2.rectangle(p3, (ax.X0 + 1, pp[1] - 13), (ax.X0 + ax.W - 1, pp[1] + 13), (55, 55, 40), -1)
                put(p3, f"#{j}", (4, int(ax.px(0, y)[1]) + 4), 0.34, C_FB if j == i else (170, 170, 170))
                if gen_ok:
                    elo, ehi = np.percentile(fj["eols"], [5, 95])
                    a0, a1 = ax.px(elo, y), ax.px(ehi, y)
                    cv2.line(p3, tuple(a0), tuple(a1), C_FB, 3)
                    cv2.circle(p3, tuple(ax.px(fj["eol_med"], y)), 5, C_FB, -1)
                    cv2.circle(p3, tuple(ax.px(fj["eol_det"], y + 0.28)), 4, C_TR, -1)
                if show:
                    q = ax.px(fj["eol_true"], y)
                    cv2.line(p3, (q[0], q[1] - 10), (q[0], q[1] + 10), C_GT, 3)
            put(p3, "FLEET end-of-life (cycles @ 80% SOH)", (6, 16))
            put(p3, "green=true  cyan=FlowBatt median+5-95%  orange=Transformer", (6, 296), 0.33, (190, 190, 190))
            canvas[8:308, 858:1278] = p3

            # ---------------- (4) SOH: GT vs prediction
            p4 = np.full((400, 640, 3), 28, np.uint8)
            ax = Axes(p4, 0, 0, 640, 400, (0, G[-1]), (0.55, 1.02), pad=(44, 12, 26, 30))
            ax.yticks([0.6, 0.7, 0.8, 0.9, 1.0]); ax.xticks([0, 500, 1000, 1500, 2000, 2500])
            ax.line([0, G[-1]], [W.EOL, W.EOL], C_EOL, 1, dashed=True)
            put(p4, "80% EOL", (ax.X0 + ax.W - 64, int(ax.px(0, W.EOL)[1]) - 6), 0.34, C_EOL)
            if act >= 2 and t_idx == steps:
                ax.band(G, f["lo"], f["hi"], C_FB, 0.28)
                ax.line(G, f["med"], C_FB, 2)
                ax.line(G, f["det"], C_TR, 2, dashed=True)
                for e, c in ((f["eol_med"], C_FB), (f["eol_det"], C_TR)):
                    q = ax.px(e, W.EOL); cv2.drawMarker(p4, tuple(q), c, cv2.MARKER_TRIANGLE_DOWN, 12, 2)
            if act == 3:
                m = G <= cursor
                ax.line(G[m], te["soh"][i][m], C_GT, 2)
                q = ax.px(cursor, 0.55); cv2.line(p4, (q[0], ax.Y0), (q[0], ax.Y0 + ax.H), (150, 150, 150), 1)
                if cursor >= f["eol_true"]:
                    q = ax.px(f["eol_true"], W.EOL); cv2.drawMarker(p4, tuple(q), C_GT, cv2.MARKER_DIAMOND, 12, 2)
            ax.line(G[:W.PREFIX], te["prefix"][i], C_PRE, 4)
            put(p4, "SOH vs cycle   white = observed prefix (cycles 0-90)   green = GROUND TRUTH (revealed)", (6, 16), 0.38)
            put(p4, "cyan = FlowBatt median + 5-95% band   orange dashed = deterministic Transformer", (6, 34), 0.38)
            if act == 3 or (act == 2 and t_idx == steps):
                put(p4, f"EOL true {f['eol_true']:.0f} | FlowBatt {f['eol_med']:.0f} "
                        f"[{np.percentile(f['eols'], 5):.0f}-{np.percentile(f['eols'], 95):.0f}] | Transformer {f['eol_det']:.0f}",
                    (ax.X0 + 6, ax.Y0 + ax.H - 10), 0.4, (235, 235, 235))
            banner = {1: "ACT 1  reading the first 100 cycles", 2: "ACT 2  flow matching: integrating dx/dt = v(x,t|c)",
                      3: "ACT 3  the real future unfolds"}[act]
            put(p4, banner, (ax.X0 + 6, ax.Y0 + 34), 0.5, C_FB)
            canvas[314:714, 6:646] = p4

            # ---------------- (5) telemetry
            if act == 3:
                j = min(int(np.searchsorted(G, cursor)), W.L - 1)
                if G[j] <= f["eol_true"] * 1.05:
                    tele["e_fb"].append(abs(f["med"][j] - te["soh"][i][j]) * 100)
                    tele["e_tr"].append(abs(f["det"][j] - te["soh"][i][j]) * 100)
                    tele["inband"].append(float(f["lo"][j] <= te["soh"][i][j] <= f["hi"][j]))
            p5 = np.full((400, 626, 3), 28, np.uint8)
            ax = Axes(p5, 0, 0, 626, 150, (0, max(20, len(tele["e_fb"]))), (0, 8), pad=(40, 10, 24, 18))
            ax.yticks([0, 4, 8], "{:.0f}")
            if len(tele["e_fb"]) > 1:
                xs = np.arange(len(tele["e_fb"]))
                ax.line(xs, tele["e_fb"], C_FB, 2); ax.line(xs, tele["e_tr"], C_TR, 2)
            put(p5, "|SOH error| at cursor (%-pts) over all reveals", (6, 16), 0.4)
            if tele["e_fb"]:
                put(p5, f"FlowBatt {np.mean(tele['e_fb']):.2f}  Transformer {np.mean(tele['e_tr']):.2f}  (running mean)",
                    (300, 16), 0.38, (220, 220, 220))
            # RUL histogram of per-sample EOL
            ax = Axes(p5, 0, 150, 626, 160, (0, 2600), (0, 40), pad=(40, 10, 22, 22))
            ax.xticks([0, 500, 1000, 1500, 2000, 2500])
            if act >= 2 and t_idx == steps:
                h, edges = np.histogram(f["eols"], bins=40, range=(0, 2600))
                for b in range(40):
                    if h[b]:
                        u0, v0 = ax.px(edges[b], h[b]); u1, v1 = ax.px(edges[b + 1], 0)
                        cv2.rectangle(p5, (int(u0) + 1, int(v0)), (int(u1) - 1, int(v1)), C_FB, -1)
                q = ax.px(f["eol_det"], 0); cv2.line(p5, (q[0], ax.Y0), (q[0], ax.Y0 + ax.H), C_TR, 2)
            if act == 3 and cursor >= f["eol_true"]:
                q = ax.px(f["eol_true"], 0); cv2.line(p5, (q[0], ax.Y0), (q[0], ax.Y0 + ax.H), C_GT, 2)
            put(p5, "end-of-life distribution: 100 FlowBatt samples (cyan), Transformer (orange), truth (green)",
                (6, 166), 0.36)
            cov = np.mean(tele["inband"]) if tele["inband"] else float("nan")
            put(p5, ("truth inside 5-95% band: --  (waiting for the reveal)" if cov != cov else
                     f"truth inside 5-95% band: {cov * 100:5.1f}% of revealed points (nominal 90%)"), (6, 334), 0.42,
                C_GT if cov == cov and cov >= 0.85 else (0, 200, 255))
            put(p5, f"featured cell {ci + 1}/3   hidden defect z = {te['hidden'][i]:+.2f} (NOT visible in input)",
                (6, 358), 0.42, (210, 210, 210))
            put(p5, "SYNTHETIC toy cell fleet - not MATR/HUST/MIX, not the authors' code", (6, 390), 0.38, (150, 150, 150))
            canvas[314:714, 652:1278] = p5
            frames.append(canvas)
            if a.live:
                cv2.imshow("FlowBatt", canvas)
                if cv2.waitKey(1) == 27:
                    return

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rgb = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in frames]
    imageio.mimsave(a.out, [cv2.resize(fr, (960, 540), interpolation=cv2.INTER_AREA) for fr in rgb[::2]],
                    duration=0.2, loop=0)
    if a.mp4:
        imageio.mimsave(a.mp4, rgb, fps=10, codec="libx264", quality=7, macro_block_size=8)
    if a.still:
        cv2.imwrite(a.still, frames[149 if len(frames) > 149 else -1])
    print(f"{len(frames)} frames -> {a.out} {a.mp4}; featured cells {feat}; running |err| FlowBatt "
          f"{np.mean(tele['e_fb']):.2f} vs Transformer {np.mean(tele['e_tr']):.2f}; in-band {np.mean(tele['inband']):.2f}")
    for i in feat:
        f = fc[i]
        print(f"  cell {i}: EOL true {f['eol_true']:.0f}  FB median {f['eol_med']:.0f} "
              f"[{np.percentile(f['eols'], 5):.0f}, {np.percentile(f['eols'], 95):.0f}]  Transformer {f['eol_det']:.0f}  "
              f"hidden {te['hidden'][i]:+.2f}")


if __name__ == "__main__":
    main()
