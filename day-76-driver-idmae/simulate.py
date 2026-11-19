"""Real-time synthetic "AI in action" simulation: ID-MAE watching a driver whose state changes mid-drive (OpenCV).

Story: one held-out (unseen) driver drives a curvy road for 48 s.  The induced condition changes every 12 s:
    0-12 s none  ->  12-24 s visual (phone glances)  ->  24-36 s cognitive (n-back-style load)  ->  36-48 s emotional
Every 0.5 s (one patch) ID-MAE runs twice on the signals received so far:
    classify  on the last 10 s           -> impairment posterior (label token un-masked)
    forecast  on the last 8 s + 2 s of masked future patches -> predicted gaze and vehicle state for the next 2 s

Layout 1280 x 720, dark UI:
  top-left  (836 x 330)  STRAIGHTENED ROAD VIEW (road frame, lateral x20): lane edges, grey = observed path (input),
                         green dashed = true next 2 s, blue = ID-MAE forecast, red = constant-velocity forecast
  mid-left  (836 x 260)  DRIVER'S VIEW: windshield schematic with areas of interest (mirrors, speedo, phone);
                         grey trail = observed gaze (last 2 s), green = true next 2 s, blue = ID-MAE gaze forecast
  right     (430 px)     (1) impairment posterior over time with the true condition band
                         (2) lateral offset + steering telemetry, observed / true future / forecast
                         (3) held-out study numbers
  bottom                 legend + SYNTHETIC disclaimer

    python simulate.py --out assets/idmae_simulation.gif --mp4 assets/idmae_simulation.mp4
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

from evaluate import load_model
from idm.baselines import kinematic_forecast
from idm.data import AOI, CONDS, DT, make_driver, simulate_drive, to_features
from idm.model import P

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
TXT, GRID = (235, 235, 235), (52, 52, 52)
C = dict(gt=(90, 220, 90), pred=(255, 160, 40), kin=(60, 60, 235), obs=(170, 170, 170))   # BGR
CC = [(200, 200, 200), (230, 120, 200), (0, 190, 255), (70, 70, 240)]   # none / cognitive / visual / emotional
SEG, WIN, FUT = 12.0, 100, 20                                          # segment s, window steps, future steps
ORDER = [0, 2, 1, 3]                                                   # none -> visual -> cognitive -> emotional


def put(img, txt, org, scale=0.45, col=TXT, th=1):
    cv2.putText(img, txt, (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def dashed(img, pts, col, th=1, seg=7):
    pts = np.asarray(pts, float)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        n = max(1, int(np.linalg.norm(b - a) / seg))
        for j in range(0, n, 2):
            u, v = a + (b - a) * j / n, a + (b - a) * min(n, j + 1) / n
            cv2.line(img, tuple(u.astype(int)), tuple(v.astype(int)), col, th, cv2.LINE_AA)


def poly(img, pts, col, th=2):
    cv2.polylines(img, [np.asarray(pts).astype(np.int32)], False, col, th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", type=int, default=117)
    ap.add_argument("--loc", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="model seed")
    ap.add_argument("--run_seed", type=int, default=4243, help="simulator noise seed (4242 = first try, see README)")
    ap.add_argument("--out", default="assets/idmae_simulation.gif")
    ap.add_argument("--mp4", default="")
    ap.add_argument("--gif_every", type=int, default=2, help="keep every n-th frame in the gif")
    a = ap.parse_args()
    torch.set_num_threads(2)
    model = load_model("idmae", 1.0, a.seed)
    S = json.load(open(ROOT / "results/summary.json"))
    drv = make_driver(a.driver)
    pre = 130                                                          # 3 s warm-up + 10 s baseline pre-roll
    n_show = int(4 * SEG / DT)
    cond = np.concatenate([np.zeros(pre, int)] + [np.full(int(SEG / DT), c) for c in ORDER] + [np.full(FUT, 3)])
    d = simulate_drive(drv, cond, loc_seed=900_000 + a.loc, run_seed=a.run_seed, init_seed=a.run_seed)
    gaze, veh, road = to_features(d)
    # ------------------------------------------------------------------ model outputs every 0.5 s
    post, fc = {}, {}
    for k in range(0, n_show + 1, P):
        i = pre + k                                                    # index of "now" (first unseen step)
        g = torch.from_numpy(gaze[None, i - WIN:i]); v = torch.from_numpy(veh[None, i - WIN:i])
        r = torch.from_numpy(road[None, i - WIN:i])
        post[k] = model.classify(g, v, r)[0][0].numpy()
        g2 = torch.from_numpy(gaze[None, i - WIN + FUT:i + FUT]); v2 = torch.from_numpy(veh[None, i - WIN + FUT:i + FUT])
        r2 = torch.from_numpy(road[None, i - WIN + FUT:i + FUT])
        pg, pv = model.forecast(g2, v2, r2, FUT // P, CFG["eval"]["sampler_steps"])
        kg, kv = kinematic_forecast(gaze[None, i - WIN + FUT:i + FUT], veh[None, i - WIN + FUT:i + FUT], FUT)
        fc[k] = dict(pg=pg[0, -FUT:].numpy(), pv=pv[0, -FUT:].numpy(), kv=kv[0, -FUT:])
    # per-segment mean posterior over the last 6 s of each segment (window fully inside the segment)
    seg_post = {CONDS[c]: np.mean([post[k] for k in post if j * SEG + 6 <= k * DT < (j + 1) * SEG], 0)
                for j, c in enumerate(ORDER)}
    s_all = d["s"]
    frames = []
    hist_t, hist_p = [], []
    for k in range(n_show):
        i = pre + k
        kk = (k // P) * P                                              # last model update
        F_ = fc[kk]
        off = k - kk                                                   # steps since the forecast was made
        cvs = np.full((720, 1280, 3), 22, np.uint8)
        c_now = int(cond[i])
        # ============================================================ straightened road view
        W_, H_ = 836, 330
        M = np.full((H_, W_, 3), 30, np.uint8)
        s0 = s_all[i]
        back, ahead = 200.0, 60.0
        sx = lambda s: (s - s0 + back) / (back + ahead) * W_
        LAT = 20.0 * 3.4                                               # px per m (lateral x20 vs ~3.2 px/m)
        sy = lambda y: H_ / 2 - y * LAT
        for yy in (-1.75, 1.75):
            cv2.line(M, (0, int(sy(yy))), (W_, int(sy(yy))), (200, 200, 200), 2)
        dashed(M, [(0, sy(0)), (W_, sy(0))], (80, 80, 80), 1, 12)
        for m_ in range(int((s0 - back) // 25) * 25, int(s0 + ahead) + 25, 25):
            x = int(sx(m_)); cv2.line(M, (x, int(sy(1.75)) + 2), (x, int(sy(-1.75)) - 2), GRID, 1)
        j0 = max(0, i - 80)
        obs = np.stack([sx(s_all[j0:i + 1]), sy(d["y"][j0:i + 1])], -1)
        for q in obs[::2]:
            cv2.circle(M, tuple(q.astype(int)), 2, C["obs"], -1, cv2.LINE_AA)
        # forecast made at step kk: positions from integrating speed from s at kk
        base_s = s_all[pre + kk - 1]
        t_i = pre + kk
        true_xy = np.stack([sx(s_all[t_i:t_i + FUT]), sy(d["y"][t_i:t_i + FUT])], -1)
        sp_pred = F_["pv"][:, 1] * 5 + 25
        pred_xy = np.stack([sx(base_s + np.cumsum(sp_pred * DT)), sy(F_["pv"][:, 0])], -1)
        sp_kin = F_["kv"][:, 1] * 5 + 25
        kin_xy = np.stack([sx(base_s + np.cumsum(sp_kin * DT)), sy(F_["kv"][:, 0])], -1)
        poly(M, kin_xy, C["kin"], 2)
        poly(M, pred_xy, C["pred"], 3)
        dashed(M, true_xy, C["gt"], 3, 8)
        cx, cy = int(sx(s0)), int(sy(d["y"][i]))
        ang = np.arctan(np.tan(d["psi"][i]) * LAT / (W_ / (back + ahead)))
        box = cv2.boxPoints(((cx, cy), (26, 12), np.degrees(-ang)))
        cv2.fillPoly(M, [box.astype(np.int32)], (240, 240, 240))
        put(M, "STRAIGHTENED ROAD VIEW (road frame, lateral exaggerated x20) - 200 m behind, 60 m ahead", (8, 18), 0.45)
        put(M, f"lateral offset {d['y'][i]:+.2f} m   speed {d['v'][i] * 3.6:5.1f} km/h   "
               f"forecast age {off * DT:.1f} s", (8, H_ - 10), 0.45, (200, 200, 200))
        if d["phone"][i]:
            put(M, "EYES ON PHONE - steering frozen", (W_ - 300, 18), 0.5, CC[2], 2)
        cvs[4:4 + H_, 4:4 + W_] = M
        # ============================================================ driver's view (gaze)
        y0 = 340
        Dv = np.full((260, W_, 3), 28, np.uint8)
        gx = lambda yaw: W_ / 2 + yaw / 60.0 * (W_ / 2 - 20)
        gy = lambda pit: 120 - pit / 40.0 * 120
        cv2.rectangle(Dv, (int(gx(-30)), int(gy(10))), (int(gx(30)), int(gy(-10))), (60, 60, 60), 1)
        put(Dv, "road ahead", (gx(-30) + 4, gy(10) + 14), 0.38, (150, 150, 150))
        for name, (yw, pt) in AOI.items():
            col = CC[2] if name == "phone" else (120, 120, 120)
            cv2.rectangle(Dv, (int(gx(yw - 7)), int(gy(pt + 5))), (int(gx(yw + 7)), int(gy(pt - 5))), col, 1)
            put(Dv, name.replace("_", " "), (gx(yw - 7), gy(pt - 5) + 13), 0.34, col)
        gtr = np.stack([gx(d["gyaw"][max(0, i - 20):i + 1]), gy(d["gpitch"][max(0, i - 20):i + 1])], -1)
        poly(Dv, gtr, C["obs"], 1)
        cv2.circle(Dv, tuple(gtr[-1].astype(int)), 7, (255, 255, 255), 2, cv2.LINE_AA)
        tg = np.stack([gx(d["gyaw"][t_i:t_i + FUT]), gy(d["gpitch"][t_i:t_i + FUT])], -1)
        pgx = np.stack([gx(F_["pg"][:, 0] * 30), gy(F_["pg"][:, 1] * 30)], -1)
        dashed(Dv, tg, C["gt"], 2, 6)
        poly(Dv, pgx, C["pred"], 2)
        cv2.circle(Dv, tuple(pgx[min(off, FUT - 1)].astype(int)), 5, C["pred"], -1, cv2.LINE_AA)
        put(Dv, "DRIVER'S VIEW - gaze (yaw -60..60 deg, pitch -40..20 deg)", (8, 16), 0.45)
        cvs[y0:y0 + 260, 4:4 + W_] = Dv
        # ============================================================ right column
        R = cvs[4:604, 846:1276]
        w = 428
        # (1) posterior
        h1 = 210
        cv2.rectangle(R, (1, 1), (w, h1), (70, 70, 70), 1)
        put(R, "ID-MAE impairment posterior (last 10 s window)", (6, 16), 0.42)
        T_all = n_show * DT
        X = lambda t: 8 + t / T_all * (w - 16)
        Yp = lambda p: h1 - 12 - p * (h1 - 84)
        for j, c in enumerate(ORDER):                                  # true condition band
            cv2.rectangle(R, (int(X(j * SEG)), 26), (int(X((j + 1) * SEG)), 40), CC[c], -1)
            put(R, CONDS[c], (X(j * SEG) + 3, 37), 0.36, (20, 20, 20))
        if kk not in hist_t:
            hist_t.append(kk); hist_p.append(post[kk])
        if len(hist_t) > 1:
            for c in range(4):
                poly(R, [(X(t * DT), Yp(p[c])) for t, p in zip(hist_t, hist_p)], CC[c], 2)
        cv2.line(R, (int(X(k * DT)), 42), (int(X(k * DT)), h1 - 6), (90, 220, 90), 1)
        yy = 56
        pn = post[kk]
        for c in range(4):
            put(R, f"{CONDS[c]} {pn[c]:.2f}", (8 + c * 104, yy), 0.38, CC[c], 2 if c == pn.argmax() else 1)
        # (2) telemetry
        y2, h2 = 216, 180
        cv2.rectangle(R, (1, y2), (w, y2 + h2), (70, 70, 70), 1)
        put(R, "lateral offset (m) and steering: last 8 s + 2 s forecast", (6, y2 + 16), 0.42)
        for row, (ch, scl, lab) in enumerate(((0, 1.2, "lat"), (2, 1.6, "steer"))):
            yc = y2 + 62 + row * 72
            cv2.line(R, (8, yc), (w - 8, yc), (50, 50, 50), 1)
            put(R, lab, (10, yc - 22), 0.36, (170, 170, 170))
            xs = lambda n: 8 + n / 100 * (w - 16)
            ob = veh[t_i - 80:t_i, ch]
            poly(R, [(xs(n), yc - np.clip(ob[n] / scl, -1, 1) * 28) for n in range(80)], C["obs"], 1)
            dashed(R, [(xs(80 + n), yc - np.clip(veh[t_i + n, ch] / scl, -1, 1) * 28) for n in range(FUT)], C["gt"], 2, 5)
            poly(R, [(xs(80 + n), yc - np.clip(F_["pv"][n, ch] / scl, -1, 1) * 28) for n in range(FUT)], C["pred"], 2)
            poly(R, [(xs(80 + n), yc - np.clip(F_["kv"][n, ch] / scl, -1, 1) * 28) for n in range(FUT)], C["kin"], 1)
            cv2.line(R, (int(xs(80)), yc - 30), (int(xs(80)), yc + 30), (90, 90, 90), 1)
        # (3) study
        y3 = 402
        cv2.rectangle(R, (1, y3), (w, 596), (70, 70, 70), 1)
        put(R, f"held-out study ({S['n_test']} windows, {S['n_test_drivers']} unseen drivers)", (6, y3 + 16), 0.42)
        cl = S["cls"]
        rows = [("paired AUC, 100% labels", "f1.0"), ("paired AUC, 10% labels", "f0.1")]
        yy = y3 + 38
        put(R, "                         ID-MAE  no-recon  GRU", (6, yy), 0.38, (170, 170, 170)); yy += 18
        for lab, f in rows:
            put(R, f"{lab:<24} {cl['idmae_' + f]['paired_auc']['mean']:.3f}   {cl['norecon_' + f]['paired_auc']['mean']:.3f}"
                   f"    {cl['gru_' + f]['paired_auc']['mean']:.3f}", (6, yy), 0.38, TXT); yy += 18
        put(R, f"visual vs cognitive AUC (100%): {cl['idmae_f1.0']['vis_vs_cog_auc']['mean']:.3f}", (6, yy), 0.38,
            CC[2]); yy += 18
        fk = S["forecast"]
        put(R, f"2 s position ADE: ID-MAE {fk['idmae_iterative']['pos_ade']:.2f} m vs CV {fk['kinematic']['pos_ade']:.2f} m",
            (6, yy), 0.38, C["pred"]); yy += 18
        put(R, f"2 s gaze error: ID-MAE {fk['idmae_iterative']['gaze_deg']:.1f} deg vs hold {fk['kinematic']['gaze_deg']:.1f} deg",
            (6, yy), 0.38, C["pred"]); yy += 18
        put(R, "paper (DIVERT): 0.706 with recon vs 0.590 without", (6, yy), 0.38, (150, 150, 150))
        # ============================================================ bottom
        put(cvs, f"t = {k * DT:4.1f} s   true condition: {CONDS[c_now]}", (8, 626), 0.5, CC[c_now], 2)
        put(cvs, "grey = observed (input) | green dashed = true next 2 s | blue = ID-MAE masked-patch forecast | "
                 "red = constant-velocity / hold-gaze baseline", (8, 652), 0.4, (200, 200, 200))
        put(cvs, f"driver #{a.driver} never seen in training (susceptibility {drv['suscept']:.2f}); model seed {a.seed}; "
                 f"sim seed {a.run_seed} (2nd tried: the 1st had no phone glance at all in the visual segment)",
            (8, 676), 0.4, (180, 180, 180))
        put(cvs, "SYNTHETIC toy simulator, my re-implementation from the abstract of arXiv 2611.23211 - not DIVERT, "
                 "not the authors' model or code", (8, 702), 0.4, (150, 150, 150))
        frames.append(cvs)
    title = np.full((720, 1280, 3), 22, np.uint8)
    put(title, "Day 76 - Is the driver impaired? Masked-token inference + gaze/trajectory forecast", (110, 290), 0.9, TXT, 2)
    put(title, "ID-MAE: one transformer predicts the impairment label as a masked token", (240, 340), 0.7, C["pred"], 2)
    put(title, "SYNTHETIC demo - toy re-implementation of arXiv 2611.23211", (330, 390), 0.7, (160, 160, 160), 1)
    frames = [title] * 15 + frames
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave(out, [cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA) for f in rgb[::a.gif_every]],
                    duration=0.1 * a.gif_every, loop=0)
    if a.mp4:
        imageio.mimsave(ROOT / a.mp4, rgb, fps=10, codec="libx264", quality=7, macro_block_size=8)
    still = 15 + int((SEG + 6) / DT)                                   # mid visual segment
    cv2.imwrite(str(out.with_name("still.png")), frames[still])
    print(f"{len(frames)} frames -> {out}")
    print("segment-mean posterior (last 6 s of each segment):",
          {k: [round(float(x), 2) for x in v] for k, v in seg_post.items()})
    ade = [np.abs(fc[k]["pv"][:, 0] - veh[pre + k:pre + k + FUT, 0]).mean() for k in fc if k < n_show]
    kade = [np.abs(fc[k]["kv"][:, 0] - veh[pre + k:pre + k + FUT, 0]).mean() for k in fc if k < n_show]
    print(f"clip lateral ADE: ID-MAE {np.mean(ade):.3f} m, CV {np.mean(kade):.3f} m")


if __name__ == "__main__":
    main()
