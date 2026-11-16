"""Real-time synthetic "AI in action" simulation for PET-MPC (OpenCV).

Layout 1280 x 720, dark UI. The SAME seed (same obstacle / crowd motion) is driven by two controllers:
  left-top     BEV  "Koopman MPC, nominal (no margin)"
  left-bottom  BEV  "PET-MPC (Koopman + Markov residual margins)"
      INPUT         grey dots = last 1 s of each agent's observed track
      GROUND TRUTH  green disc = where the agent REALLY is; green dashed = its real next 2 s
      PREDICTION    cyan polyline = 2 s Koopman forecast; cyan rings = agent radius + Markov margin m_i at
                    0.5 / 1.0 / 1.5 / 2.0 s (the tube widens with the measured prediction error)
      PLANNING      magenta = MPC plan of the look-ahead point; yellow segments = active separating half-planes
                    (one per aggregated group); red X = physical contact
  right column  telemetry: (1) realised i-step error (hindsight) vs Markov margin for the closest agent,
                (2) clearance over time for both controllers, (3) robot speed for both controllers
  bottom strip  legend + held-out paired-seed results

    python simulate.py --out assets/petmpc_simulation.gif --mp4 assets/petmpc_simulation.mp4
SYNTHETIC toy world - not the authors' simulator or code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import yaml

from petmpc import world as Wd
from petmpc.episode import run_episode

ROOT = Path(__file__).resolve().parent
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
BG, TXT = (22, 22, 22), (235, 235, 235)
C_GT, C_PRED, C_PLAN, C_CON, C_RED = (90, 220, 90), (255, 215, 0), (255, 80, 220), (0, 220, 255), (60, 60, 255)
C_NOM, C_PET, C_ROB = (0, 150, 255), (255, 215, 0), (245, 245, 245)
BW, BH = 860, 290
X0, X1 = -1.0, 21.0
SX = BW / (X1 - X0)
YC = BH // 2 + 6


def put(img, txt, org, scale=0.45, col=TXT, th=1):
    (w, h), b = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + b), BG, -1)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def px(xy):
    xy = np.asarray(xy, float)
    return np.stack([(xy[..., 0] - X0) * SX, YC - xy[..., 1] * SX], -1).astype(np.int32)


def dashed(img, pts, col, th=1, seg=8):
    p = px(pts)
    for i in range(len(p) - 1):
        a, b = p[i].astype(float), p[i + 1].astype(float)
        n = max(1, int(np.linalg.norm(b - a) / seg))
        for j in range(0, n, 2):
            u, v = a + (b - a) * j / n, a + (b - a) * min(n, j + 1) / n
            cv2.line(img, tuple(u.astype(int)), tuple(v.astype(int)), col, th, cv2.LINE_AA)


def ring(img, c, r, col, th=1, alpha=None):
    if alpha is not None:
        ov = img.copy()
        cv2.circle(ov, tuple(px(c)), max(1, int(r * SX)), col, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, dst=img)
    cv2.circle(img, tuple(px(c)), max(1, int(r * SX)), col, th, cv2.LINE_AA)


def contact_time(run):
    fr = run["frames"]
    radii = None
    for k in range(1, len(fr)):
        d = np.linalg.norm(fr[k]["agents"] - fr[k]["state"][:2], axis=1)
        if radii is None:
            radii = np.full(len(d), Wd.R_OBS if run["scenario"] == "obstacle" else Wd.R_PED)
        if np.any(d - Wd.R_ROBOT - radii < 0):
            return k
    return None


def draw_bev(run, k, title, col_title, show_margin, tally):
    fr = run["frames"]
    kk = min(k, len(fr) - 1)
    f = fr[kk]
    sc = run["scenario"]
    r_ag = Wd.R_OBS if sc == "obstacle" else Wd.R_PED
    img = np.full((BH, BW, 3), 34, np.uint8)
    for yw in (-Wd.HALF_W, Wd.HALF_W):
        cv2.line(img, tuple(px([X0, yw])), tuple(px([X1, yw])), (200, 200, 200), 2)
    dashed(img, np.array([[X0, 0], [X1, 0]]), (80, 80, 80), 1, 14)
    cv2.line(img, tuple(px([Wd.GOAL_X, -Wd.HALF_W])), tuple(px([Wd.GOAL_X, Wd.HALF_W])), (90, 200, 90), 2)
    put(img, "GOAL", px([Wd.GOAL_X - 0.9, Wd.HALF_W - 0.35]), 0.4, C_GT)
    N = f["pred"].shape[1]
    for j in range(len(f["agents"])):
        # input: last 1 s of history
        for b in range(max(0, kk - 10), kk, 2):
            cv2.circle(img, tuple(px(fr[b]["agents"][j])), 2, (140, 140, 140), -1)
        # ground truth future (hindsight)
        fut = [fr[b]["agents"][j] for b in range(kk, min(len(fr), kk + N + 1))]
        if len(fut) > 1:
            dashed(img, np.array(fut), C_GT, 1)
        # prediction + margin tube
        P = f["pred"][j]
        cv2.polylines(img, [px(P)], False, C_PRED, 1, cv2.LINE_AA)
        if show_margin:
            for i in (4, 9, 14, 19):
                if i < N:
                    ring(img, P[i], r_ag + f["marg"][j, i], C_PRED, 1, alpha=0.07)
        else:
            for i in (9, 19):
                if i < N:
                    ring(img, P[i], r_ag, C_PRED, 1)
        if sc == "crowd":
            ring(img, f["agents"][j], Wd.PS_PED, (0, 190, 255), 1)
        ring(img, f["agents"][j], r_ag, C_GT, 2, alpha=0.55)
    # active separating half-planes (earliest step per row group, drawn as short segments)
    seen = set()
    for i, h, ell in f["rows"]:
        key = (round(h[0], 1), round(h[1], 1))
        if i > 6 or key in seen or len(seen) >= 2:                          # at most 2 lines, near-term only
            continue
        seen.add(key)
        t = np.array([-h[1], h[0]])
        foot = f["plan"][i] - (f["plan"][i] @ h - ell) * h
        seg = np.array([foot - 1.0 * t, foot + 1.0 * t])
        cv2.line(img, tuple(px(seg[0])), tuple(px(seg[1])), C_CON, 2, cv2.LINE_AA)
        cv2.arrowedLine(img, tuple(px(foot)), tuple(px(foot + 0.45 * h)), C_CON, 1, cv2.LINE_AA, tipLength=0.4)
    # robot
    cv2.polylines(img, [px(f["plan"])], False, C_PLAN, 2, cv2.LINE_AA)
    st = f["state"]
    ring(img, st[:2], Wd.R_ROBOT, C_ROB, 2, alpha=0.8)
    hd = st[:2] + 0.45 * np.array([np.cos(st[2]), np.sin(st[2])])
    cv2.line(img, tuple(px(st[:2])), tuple(px(hd)), (40, 40, 40), 2)
    kc = tally["contact_k"]
    if kc is not None and k >= kc:
        pc = fr[kc]["state"][:2]
        cv2.drawMarker(img, tuple(px(pc)), C_RED, cv2.MARKER_TILTED_CROSS, 34, 3)
        put(img, f"CONTACT at t = {fr[kc]['t']:.1f} s", px(pc) + [-70, -26], 0.5, C_RED)
    put(img, title, (8, 18), 0.52, col_title)
    status = f"t = {f['t']:4.1f} s   v = {f['v']:.2f} m/s"
    if k >= len(fr) - 1 and run["reached"]:
        status += f"   GOAL reached in {run['time']:.1f} s"
    put(img, status, (BW - 330, 18), 0.45)
    return img


def plot(img, x0, y0, w, h, series, ymin, ymax, title, tmax, hl=None):
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
    put(img, title, (x0 + 4, y0 + 15), 0.4)
    Y = lambda v: (y0 + h - 6 - (np.clip(v, ymin, ymax) - ymin) / (ymax - ymin) * (h - 30)).astype(int)
    if hl is not None:
        yy = int(Y(np.array(hl[0])))
        cv2.line(img, (x0, yy), (x0 + w, yy), hl[1], 1)
        put(img, hl[2], (x0 + 4, yy - 3), 0.33, hl[1])
    xl = x0 + w - 4
    for xs, ys, col, lab in series:
        if len(xs) > 1:
            P = np.stack([x0 + np.asarray(xs) / tmax * w, Y(np.asarray(ys))], -1).astype(np.int32)
            cv2.polylines(img, [P], False, col, 2, cv2.LINE_AA)
        tw = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)[0][0]
        xl -= tw + 8
        put(img, lab, (xl, y0 + 30), 0.34, col)


def err_vs_margin(img, x0, y0, w, h, run, k):
    """Bars: realised i-step error (hindsight) of the agent closest to the robot vs its Markov margin."""
    fr = run["frames"]
    kk = min(k, len(fr) - 1)
    f = fr[kk]
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
    put(img, "closest agent: realised error vs Markov margin", (x0 + 4, y0 + 15), 0.4)
    j = int(np.argmin(np.linalg.norm(f["agents"] - f["state"][:2], axis=1)))
    N = f["pred"].shape[1]
    ymax = 1.2
    bw = (w - 20) / N
    for i in range(N):
        xb = int(x0 + 10 + i * bw)
        if kk + i + 1 < len(fr):
            e = np.linalg.norm(fr[kk + i + 1]["agents"][j] - f["pred"][j, i])
            hb = int(min(e, ymax) / ymax * (h - 40))
            cv2.rectangle(img, (xb, y0 + h - 8 - hb), (int(xb + bw - 2), y0 + h - 8), C_GT, -1)
        m = f["markov"][j, i]
        ym = int(y0 + h - 8 - min(m, ymax) / ymax * (h - 40))
        cv2.line(img, (xb, ym), (int(xb + bw - 2), ym), C_PET, 2)
    put(img, "green = |error| at step i (hindsight)   cyan = margin m_i", (x0 + 4, y0 + 32), 0.34, (200, 200, 200))
    put(img, "0 s", (x0 + 6, y0 + h - 10), 0.32, (160, 160, 160))
    put(img, "2 s", (x0 + w - 26, y0 + h - 10), 0.32, (160, 160, 160))


def clearance(run):
    fr = run["frames"]
    r = Wd.R_OBS if run["scenario"] == "obstacle" else Wd.R_PED
    return np.array([np.min(np.linalg.norm(f["agents"] - f["state"][:2], axis=1)) - Wd.R_ROBOT - r for f in fr])


def pick(study):
    """Deterministic, honest choice: first held-out seeds where nominal has contact and PET-MPC has none,
    plus the first held-out seed where PET-MPC itself has contact (if any)."""
    d = json.load(open(ROOT / "results" / f"{study}.json"))
    ps = d.get("per_seed")
    s0 = d["seeds"][0]
    if ps is None:
        return [], None
    nom, pet = np.array(ps["koopman_nominal"]), np.array(ps["pet_mpc"])
    wins = [s0 + int(i) for i in np.where((nom == 1) & (pet == 0))[0]]
    fails = [s0 + int(i) for i in np.where(pet == 1)[0]]
    return wins, (fails[0] if fails else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/petmpc_simulation.gif")
    ap.add_argument("--mp4", default="")
    ap.add_argument("--n-wins", type=int, default=2)
    ap.add_argument("--crowd-seed", type=int, default=1013)
    ap.add_argument("--crowd-fail", type=int, default=1005)
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    wins, fail = pick("obstacle")
    vignettes = [("obstacle", s, "nominal collides, PET-MPC avoids") for s in wins[:a.n_wins]]
    if a.crowd_seed >= 0:
        vignettes.append(("crowd", a.crowd_seed, "10 pedestrians: PET-MPC keeps out of personal space (orange rings)"))
    if fail is not None:
        vignettes.append(("obstacle", fail, "honest failure: PET-MPC also makes contact"))
    if a.crowd_fail >= 0:
        vignettes.append(("crowd", a.crowd_fail, "honest failure: a 1.7 m/s pedestrian brushes the slowed robot"))
    res = json.load(open(ROOT / "results" / "obstacle.json"))["summary"]
    cres = json.load(open(ROOT / "results" / "crowd.json"))["summary"]
    still_idx = None
    print("vignettes", vignettes)
    frames = []
    for vi, (sc, seed, caption) in enumerate(vignettes):
        npd = CFG["eval"]["n_ped"]
        nom = run_episode(sc, seed, "koopman_nominal", CFG, log=True, n_ped=npd)
        pet = run_episode(sc, seed, "pet_mpc", CFG, log=True, n_ped=npd)
        tn, tp = dict(contact_k=contact_time(nom)), dict(contact_k=contact_time(pet))
        cn, cp = clearance(nom), clearance(pet)
        L = max(len(nom["frames"]), len(pet["frames"]))
        tmax = L * Wd.DT
        stride = a.stride if sc == "obstacle" else 2 * a.stride
        for k in list(range(0, L, stride)) + [L - 1] * 8:
            cv = np.full((720, 1280, 3), 22, np.uint8)
            cv[4:4 + BH, 4:4 + BW] = draw_bev(nom, k, "Koopman MPC, nominal (no margin)", C_NOM, False, tn)
            cv[8 + BH:8 + 2 * BH, 4:4 + BW] = draw_bev(pet, k, "PET-MPC: Koopman + Markov residual margins", C_PET, True, tp)
            rc = cv[4:600, 872:1276]
            err_vs_margin(rc, 2, 2, 400, 190, pet, k)
            ts = np.arange(len(cn)) * Wd.DT
            kn, kp = min(k, len(cn) - 1), min(k, len(cp) - 1)
            plot(rc, 2, 200, 400, 190, [(ts[:kn + 1], cn[:kn + 1], C_NOM, "nominal"),
                                        (np.arange(kp + 1) * Wd.DT, cp[:kp + 1], C_PET, "PET-MPC")],
                 -0.3, 3.0, "clearance to nearest agent (m)", tmax, hl=(0.0, C_RED, "contact"))
            vn = [f["v"] for f in nom["frames"]][:kn + 1]
            vp = [f["v"] for f in pet["frames"]][:kp + 1]
            plot(rc, 2, 398, 400, 190, [(np.arange(len(vn)) * Wd.DT, vn, C_NOM, "nominal"),
                                        (np.arange(len(vp)) * Wd.DT, vp, C_PET, "PET-MPC")],
                 0, 1.3, "robot speed (m/s)", tmax)
            # bottom strip
            put(cv, f"vignette {vi + 1}/{len(vignettes)}  seed {seed} ({sc}): {caption}", (8, 618), 0.5)
            put(cv, "grey dots = 1 s history | green = truth (dashed: real next 2 s) | cyan = 2 s Koopman forecast + "
                    "margin rings | magenta = MPC plan | yellow = active half-planes", (8, 642), 0.4, (200, 200, 200))
            if sc == "obstacle":
                sn, sp_ = res["koopman_nominal"], res["pet_mpc"]
                put(cv, f"held-out paired study ({sp_['n']} seeds, 1 moving obstacle): runs with contact  nominal "
                        f"{100*sn['contact_rate']:.1f}%  ->  PET-MPC {100*sp_['contact_rate']:.1f}%   |   mean time "
                        f"{sn['mean_time']:.2f} s -> {sp_['mean_time']:.2f} s", (8, 668), 0.45, C_PET)
            else:
                sn, sp_, sg = cres["koopman_nominal"], cres["pet_mpc"], cres["pet_mpc_generic"]
                put(cv, f"crowd study ({sp_['n']} seeds, 10 peds): contact {100*sn['contact_rate']:.0f}% -> "
                        f"{100*sp_['contact_rate']:.0f}% | personal-space time {100*sn['ps_overlap_frac']:.1f}% -> "
                        f"{100*sp_['ps_overlap_frac']:.1f}% (generic {100*sg['ps_overlap_frac']:.1f}%) | time "
                        f"{sn['mean_time']:.1f} -> {sp_['mean_time']:.1f} s", (8, 668), 0.45, C_PET)
            put(cv, "SYNTHETIC toy world, my re-implementation from the abstract - not the authors' simulator or code "
                    "(paper: 10.0% -> 1.33%, +0.68 s)", (8, 694), 0.4, (150, 150, 150))
            if vi == 0 and still_idx is None and tn["contact_k"] is not None and k >= tn["contact_k"]:
                still_idx = len(frames)
            frames.append(cv)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rgb = [cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) for fr in frames]
    imageio.mimsave(a.out, [cv2.resize(fr, (960, 540), interpolation=cv2.INTER_AREA) for fr in rgb[::2]],
                    duration=0.2, loop=0)
    if a.mp4:
        imageio.mimsave(a.mp4, rgb, fps=10, codec="libx264", quality=7, macro_block_size=8)
    still = frames[still_idx if still_idx is not None else int(len(frames) * 0.3)]
    cv2.imwrite(str(Path(a.out).with_name("still.png")), still)
    print(f"{len(frames)} frames -> {a.out} {a.mp4}")


if __name__ == "__main__":
    main()
