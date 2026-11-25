"""Real-time synthetic "AI in action" clip: a clean-trained vs a noise-augmented interaction-aware Transformer
predicting highway traffic while the sensing / V2X noise level is ramped up.

Layout 1280 x 720 (BGR, 10 fps):
    top panel     bird's-eye view, CLEAN-TRAINED predictor (curated data, the usual practice)    - red
    middle panel  bird's-eye view, NOISE-AUGMENTED RETRAIN (same architecture)                   - blue
                  both panels get the SAME noisy input every frame:
                    grey dots   = sensor-tracked history, orange dots = V2X-received history,
                    hollow dots = packet lost -> stale value held, white outline box = TRUE current position,
                    green dashed = TRUE future (3 s), thin lines = the 6 predicted modes, thick = top-scoring mode
    bottom-left   held-out minADE6 vs noise level for every pipeline, cursor at the current level
    bottom-centre this frame: minADE6 of the focus agent and of all 8 agents, for both predictors (live bars)
    bottom-right  degradation factors (held-out)
Acts per scene: noise level 0 -> 1 -> 2 -> 4 (2 s each, a new noise draw every frame, predictions recomputed live)
-> REVEAL (3 s): traffic plays forward along the true future, predictions stay frozen.

    python simulate.py --out assets/noisytraj_simulation.gif --mp4 assets/noisytraj_simulation.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from evaluate import LABEL, load
from nr.noise import add_noise
from nr.scene import CAR_L, LANE_W, T_H
from train import load_split, to_batch

ROOT = Path(__file__).resolve().parent
TXT, DIM = (235, 235, 235), (150, 150, 150)
COL = dict(clean=(70, 70, 235), aug=(255, 150, 40), gt=(90, 220, 90), sens=(190, 190, 190), v2x=(0, 165, 255),
           clean_kf=(180, 120, 220), clean_dn=(200, 80, 200), cv=(120, 120, 120), cv_kf=(160, 160, 100))
PX, PW, PH = 20, 1240, 205                       # BEV panels
PANELS = [(28, "clean"), (258, "aug")]
SX, SY = 9.5, 1.6 * 9.5                          # px per metre (lateral exaggerated x1.6)


def put(img, txt, org, s=0.45, col=TXT, th=1):
    cv2.putText(img, txt, (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)


def bev_panel(img, y0, name, view_x0, st, hist, fresh, v2x, pred, logit, t, focus, err_focus):
    """Draw one bird's-eye panel.  st: clean states (N, 50, 4); hist noisy (N, 20, 4); pred (N, K, 30, 2)."""
    P = lambda X, Y: (int(PX + (X - view_x0) * SX), int(y0 + PH / 2 + 8 - Y * SY))
    full = img
    img = full.copy()                                    # draw on a copy, then paste only the panel rectangle (clipping)
    cv2.rectangle(img, (PX, y0), (PX + PW, y0 + PH), (48, 48, 50), -1)
    top, bot = P(view_x0, 1.5 * LANE_W)[1], P(view_x0, -1.5 * LANE_W)[1]
    cv2.rectangle(img, (PX, top), (PX + PW, bot), (78, 78, 82), -1)
    for yl in (-0.5 * LANE_W, 0.5 * LANE_W):
        for xx in np.arange(np.floor(view_x0 / 8) * 8, view_x0 + PW / SX, 8):
            a, b = P(xx, yl), P(xx + 4, yl)
            if a[0] >= PX and b[0] <= PX + PW:
                cv2.line(img, a, b, (210, 210, 210), 1)
    for yl in (-1.5 * LANE_W, 1.5 * LANE_W):
        cv2.line(img, (PX, P(0, yl)[1]), (PX + PW, P(0, yl)[1]), (235, 235, 235), 2)
    clip = lambda p: PX <= p[0] <= PX + PW
    N = st.shape[0]
    tt = T_H - 1 + t                                                                  # index of the "now" state
    for i in range(N):
        # true past (faint) and true future (green dashed)
        q = [P(*st[i, k, :2]) for k in range(0, T_H)]
        cv2.polylines(img, [np.array(q, np.int32)], False, (110, 110, 110), 1, cv2.LINE_AA)
        fut = [P(*st[i, k, :2]) for k in range(T_H - 1, st.shape[1])]
        for k in range(0, len(fut) - 1, 2):
            cv2.line(img, fut[k], fut[k + 1], COL["gt"], 1 if i != focus else 2, cv2.LINE_AA)
        # predictions: 6 thin modes + thick top mode
        top_k = int(np.argmax(logit[i]))
        for k in range(pred.shape[1]):
            pts = np.array([P(*hist[i, -1, :2])] + [P(*p) for p in pred[i, k]], np.int32)
            cv2.polylines(img, [pts], False, COL[name], 3 if k == top_k else 1, cv2.LINE_AA)
        # noisy observed history
        c = COL["v2x"] if v2x[i] else COL["sens"]
        for k in range(hist.shape[1]):
            p = P(*hist[i, k, :2])
            if clip(p):
                cv2.circle(img, p, 3 if k == hist.shape[1] - 1 else 2, c, -1 if fresh[i, k] > 0 else 1, cv2.LINE_AA)
        # true vehicle box at "now"
        x, y, h, _ = st[i, tt]
        a, b = P(x - CAR_L / 2, y + 0.9), P(x + CAR_L / 2, y - 0.9)
        cv2.rectangle(img, a, b, (255, 255, 255), 2 if i == focus else 1)
        if i == focus:
            put(img, "focus", (a[0], a[1] - 4), 0.4, (255, 255, 255))
    full[y0:y0 + PH, PX:PX + PW] = img[y0:y0 + PH, PX:PX + PW]
    img = full
    head = "CLEAN-TRAINED predictor (curated, noise-free training data)" if name == "clean" else \
        "NOISE-AUGMENTED RETRAIN (same Transformer, trained on noisy histories)"
    cv2.rectangle(img, (PX - 1, y0 - 1), (PX + PW, y0 + PH), COL[name], 2)
    put(img, head, (PX + 8, y0 + 18), 0.5, COL[name], 1)
    put(img, f"focus agent minADE6 {err_focus:.2f} m", (PX + PW - 250, y0 + 18), 0.5, (255, 255, 255), 1)


def curves(img, x0, y0, w, h, S, lvl):
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
    put(img, "held-out minADE6 (m) vs noise level", (x0 + 6, y0 + 16), 0.42)
    L = S["levels"]
    M = S["metrics"]
    ymax = max(M[n][str(L[-1])]["ade"] for n in ("clean", "aug", "clean_dn", "clean_kf")) * 1.1
    gx0, gy0, gw, gh = x0 + 40, y0 + 30, w - 60, h - 70
    for v in np.linspace(0, ymax, 5):
        yy = int(gy0 + gh - v / ymax * gh)
        cv2.line(img, (gx0, yy), (gx0 + gw, yy), (50, 50, 50), 1)
        put(img, f"{v:.1f}", (x0 + 6, yy + 4), 0.35, DIM)
    X = lambda l: int(gx0 + l / L[-1] * gw)
    for l in L:
        put(img, f"{l:g}", (X(l) - 6, gy0 + gh + 16), 0.35, DIM)
    for n in ("clean", "clean_kf", "clean_dn", "aug"):
        pts = np.array([(X(l), int(gy0 + gh - M[n][str(l)]["ade"] / ymax * gh)) for l in L], np.int32)
        cv2.polylines(img, [pts], False, COL[n], 2, cv2.LINE_AA)
    cv2.line(img, (X(lvl), gy0), (X(lvl), gy0 + gh), (255, 255, 255), 1)
    yy = y0 + h - 14
    for j, (n, lab) in enumerate((("clean", "clean"), ("clean_kf", "+Kalman"), ("clean_dn", "+learned denoiser"),
                                  ("aug", "noise-aug"))):
        xx = x0 + 8 + j * (w - 10) // 4
        cv2.line(img, (xx, yy - 4), (xx + 14, yy - 4), COL[n], 2)
        put(img, lab, (xx + 18, yy), 0.33, COL[n])


def live_bars(img, x0, y0, w, h, e):
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
    put(img, "this frame: minADE6 (m)", (x0 + 6, y0 + 16), 0.42)
    vmax = 6.0
    rows = [("focus, clean-trained", e["clean_f"], COL["clean"]), ("focus, noise-aug", e["aug_f"], COL["aug"]),
            ("all 8, clean-trained", e["clean_a"], COL["clean"]), ("all 8, noise-aug", e["aug_a"], COL["aug"])]
    for j, (lab, v, c) in enumerate(rows):
        yy = y0 + 34 + j * 34
        put(img, lab, (x0 + 8, yy + 12), 0.38, DIM)
        bw = int(min(v, vmax) / vmax * (w - 210))
        cv2.rectangle(img, (x0 + 160, yy), (x0 + 160 + bw, yy + 16), c, -1)
        put(img, f"{v:.2f}", (x0 + 166 + bw, yy + 13), 0.4, TXT)
    put(img, f"t0 position error of the input: {e['herr']:.2f} m (mean over agents)", (x0 + 8, y0 + h - 12), 0.37,
        (200, 200, 200))


def table(img, x0, y0, w, h, S):
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
    M = S["metrics"]
    put(img, f"held-out {S['n_scenes']} scenes: minADE6 at level 0 / 2 / 4", (x0 + 6, y0 + 16), 0.4)
    yy = y0 + 38
    for n in ("cv", "clean", "clean_kf", "clean_dn", "aug"):
        r = M[n]
        put(img, f"{LABEL[n][:30]:<30}", (x0 + 6, yy), 0.35, COL.get(n, TXT))
        put(img, f"{r['0']['ade']:.2f} / {r['2']['ade']:.2f} / {r['4']['ade']:.2f}   {r['4']['degradation']:.1f}x",
            (x0 + 232, yy), 0.35, COL.get(n, TXT))
        yy += 22
    put(img, "last column: degradation at level 4", (x0 + 6, yy + 6), 0.33, DIM)
    put(img, "(vs the same predictor on clean input)", (x0 + 6, yy + 22), 0.33, DIM)


def scene_frames(si, d, models, S, label, note, frames, levels=(0, 1, 2, 4), per=20, seed=0):
    b = to_batch(d, np.array([si]))
    st = d["states"][si]
    v2x = b["v2x"]
    # focus = the V2X-received agent whose true future deviates most from constant velocity (an interesting one)
    cand = np.where(d["v2x"][si])[0]
    cand = cand if len(cand) else np.arange(8)
    p0, vel = st[:, T_H - 1, :2], st[:, T_H - 1, :2] - st[:, T_H - 2, :2]
    dev = np.linalg.norm(st[:, -1, :2] - (p0 + vel * 30), axis=-1)
    focus = int(cand[np.argmax(dev[cand])])
    view_x0 = float(st[focus, T_H - 1, 0] - 40)
    g = torch.Generator().manual_seed(4000 + si + seed)
    last = None
    seq = [(l, f) for l in levels for f in range(per)] + [(levels[-1], -1 - k) for k in range(30)]
    for lvl, f in seq:
        reveal = f < 0
        if not reveal or last is None:
            hist, fresh = add_noise(b["hist"], v2x, float(lvl), g)
            out = {n: models[n](hist, fresh, v2x) for n in ("clean", "aug")}
            pr = {n: out[n]["pred"][0].detach().numpy() for n in out}
            lg = {n: out[n]["logit"][0].detach().numpy() for n in out}
            fut = b["fut"][0].numpy()
            err = {n: np.linalg.norm(pr[n] - fut[:, None], axis=-1).mean(-1).min(-1) for n in pr}  # (N,)
            e = dict(clean_f=err["clean"][focus], aug_f=err["aug"][focus], clean_a=err["clean"].mean(),
                     aug_a=err["aug"].mean(),
                     herr=float((hist[0, :, -1, :2] - b["hist"][0, :, -1, :2]).norm(dim=-1).mean()))
            last = (hist[0].numpy(), fresh[0].numpy(), pr, lg, e)
        h_, fr_, pr, lg, e = last
        t = (-f - 1) if reveal else 0
        img = np.full((720, 1280, 3), 22, np.uint8)
        for y0, n in PANELS:
            bev_panel(img, y0, n, view_x0, st, h_, fr_, d["v2x"][si], pr[n], lg[n], t, focus, e[f"{n}_f"])
        curves(img, 20, 474, 420, 214, S, lvl)
        live_bars(img, 452, 474, 400, 214, e)
        table(img, 864, 474, 396, 214, S)
        act = f"REVEAL t = +{(t + 1) * 0.1:.1f} s (predictions frozen)" if reveal else f"noise level {lvl:g}"
        put(img, f"{label}: held-out scene #{si} ({note})   [{act}]", (20, 18), 0.5, (120, 220, 255))
        put(img, "grey = sensor-tracked history | orange = V2X-received | hollow = packet lost (stale value held) | "
                 "white box = TRUE position | green dashed = TRUE future | lateral scale x1.6", (20, 700), 0.37,
            (200, 200, 200))
        put(img, "SYNTHETIC highway traffic + synthetic noise model, my re-implementation from the abstract of arXiv "
                 "2611.28128 - not the authors' models, data or noise", (20, 715), 0.37, (150, 150, 150))
        frames.append(img)
    return dict(scene=int(si), focus=focus, note=note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/noisytraj_simulation.gif")
    ap.add_argument("--mp4", default="")
    ap.add_argument("--gif_every", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(2)
    d = load_split("test")
    S = json.load(open(ROOT / "results" / "summary.json"))
    models = {"clean": load("clean", 0), "aug": load("aug", 0)}
    frames = []
    title = np.full((720, 1280, 3), 22, np.uint8)
    put(title, "Day 81 - Trajectory prediction under sensing and V2X noise", (230, 280), 1.0, TXT, 2)
    put(title, "same Transformer, trained clean vs trained on noise-augmented histories", (250, 330), 0.75, COL["aug"], 2)
    put(title, "SYNTHETIC demo - toy re-implementation of arXiv 2611.28128", (340, 380), 0.7, (160, 160, 160), 1)
    frames += [title] * 15
    C = S["clips"]
    info = [scene_frames(C["random"], d, models, S, "scene 1", "RANDOM scene", frames)]
    if C.get("win") is not None:
        info.append(scene_frames(C["win"], d, models, S, "scene 2",
                                 f"SELECTED: noise-aug much better at level 4 ({C['n_win']} such scenes)", frames))
    if C.get("loss") is not None:
        info.append(scene_frames(C["loss"], d, models, S, "scene 3",
                                 f"SELECTED: noise-aug worse than clean-trained on CLEAN input ({C['n_loss']} such scenes)", frames))
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave(out, [cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA) for f in rgb[::a.gif_every]],
                    duration=0.1 * a.gif_every, loop=0)
    if a.mp4:
        imageio.mimsave(ROOT / a.mp4, rgb, fps=10, codec="libx264", quality=7, macro_block_size=8)
    cv2.imwrite(str(out.with_name("still.png")), frames[15 + 70])
    print(f"{len(frames)} frames -> {out}")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
