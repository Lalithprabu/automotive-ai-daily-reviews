"""Real-time synthetic "AI in action" clip: imaging-radar-only baseline vs RadarOSV-style dual-radar network on a
moving street scene. Every frame is rendered from the simulator and both networks run live on it.

Layout 1280 x 720 (BGR, 10 fps):
  top row     A  INPUT 1: imaging radar (fine range/azimuth, Doppler aliases every 5 m/s)  - colour = strongest Doppler
              B  INPUT 2: single-chip radar (coarse, wide +-24 m/s Doppler)              - same colour scale
              C  BASELINE prediction: imaging radar + external ego velocity              - classes + radial-velocity arrows
              D  RadarOSV-style prediction: both radars, radar-estimated ego velocity
  bottom row  E  GROUND TRUTH: occupancy classes + true radial velocity (green arrows)
              F  live telemetry: dynamic-cell radial-velocity MAE per frame, both models
              G  live telemetry: ego speed, true vs radar-estimated
              H  held-out table (1,500 scenes)
In C and D: white boxes = TRUE objects, green arrow = TRUE radial velocity, coloured arrow = predicted radial velocity
(mean over the object's cells), label "gt / pred" in m/s.

    python simulate.py --out assets/radarosv_simulation.gif --mp4 assets/radarosv_simulation.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from evaluate import load
from osv.radar import AI, AS, DI, DS, RI, RS, VI, VS, render
from osv.scene import CLASSES, DYN, FOV, FOV_MASK, G, GTH, R_MAX, make_scene, rasterize, step_scene

ROOT = Path(__file__).resolve().parent
P = 304                                                  # panel size (px)
CLS_COL = {0: (34, 34, 34), 1: (120, 120, 120), 2: (60, 170, 255), 3: (40, 110, 230), 4: (230, 120, 255),
           5: (120, 255, 255)}
COL = dict(base=(80, 80, 240), osv=(255, 170, 40), gt=(90, 230, 90), txt=(235, 235, 235), dim=(150, 150, 150))

# pixel -> (x, y) for the BEV panels: forward is up, left is left
_u, _v = np.meshgrid(np.arange(P), np.arange(P))
PX_Y = 12.8 - (_u + 0.5) / P * 25.6
PX_X = 25.6 - (_v + 0.5) / P * 25.6
PX_R, PX_T = np.hypot(PX_X, PX_Y), np.arctan2(PX_Y, PX_X)
PX_IN = (np.abs(PX_T) < FOV) & (PX_R < R_MAX)


def put(img, t, org, s=0.42, col=COL["txt"], th=1):
    cv2.putText(img, t, (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)


def to_px(x, y):
    return int((12.8 - y) / 25.6 * P), int((25.6 - x) / 25.6 * P)


def vel_color(v, vmax=12.0):
    """Diverging colour: approaching (negative) = blue, receding (positive) = red, BGR."""
    t = np.clip(v / vmax, -1, 1)
    b = np.where(t < 0, 255, 255 * (1 - t))
    r = np.where(t > 0, 255, 255 * (1 + t))
    g = 255 * (1 - np.abs(t)) * 0.85
    return np.stack([b, g, r], -1)


def fan(cube, nr, na, nd, vmax):
    """Polar cube (D, R, A) -> BEV fan image: brightness = max power over Doppler, hue = Doppler of that peak."""
    c = cube.astype(np.float32)
    pw, di = c.max(0), c.argmax(0)
    v = (di + 0.5) / nd * 2 * vmax - vmax
    ri = np.clip((PX_R / (R_MAX / nr)).astype(int), 0, nr - 1)
    ai = np.clip(((PX_T + FOV) / (2 * FOV) * na).astype(int), 0, na - 1)
    floor = np.percentile(pw, 75)                                          # suppress the noise floor
    p = np.clip((pw[ri, ai] - floor) / max(pw.max() - floor, 1e-3), 0, 1) ** 0.7
    img = vel_color(v[ri, ai]) * p[..., None]
    img[~PX_IN] = (18, 18, 18)
    return img.astype(np.uint8)


def bev(sem):
    img = np.array([CLS_COL[c] for c in range(6)], np.uint8)[sem[::-1, ::-1]]
    img = cv2.resize(img, (P, P), interpolation=cv2.INTER_NEAREST)
    img[~PX_IN] = (14, 14, 14)
    return img


def obj_masks(scene):
    out = []
    for o in scene["objs"]:
        if o["cls"] in DYN:
            m = rasterize(dict(v_ego=scene["v_ego"], objs=[o]))["sem"] > 0
            m &= FOV_MASK
            if m.sum() > 0:
                out.append((o, m))
    return out


def arrows(img, objs, vgt, vpred, col):
    for o, m in objs:
        cx, cy = o["cx"], o["cy"]
        th = np.arctan2(cy, cx)
        u = np.array([np.cos(th), np.sin(th)])
        p0 = to_px(cx, cy)
        g = float(vgt[m].mean())
        p1 = to_px(cx + 0.35 * g * u[0], cy + 0.35 * g * u[1])
        cv2.arrowedLine(img, p0, p1, COL["gt"], 2, cv2.LINE_AA, tipLength=0.3)
        if vpred is not None:
            pr = float(vpred[m].mean())
            p2 = to_px(cx + 0.35 * pr * u[0], cy + 0.35 * pr * u[1])
            cv2.arrowedLine(img, (p0[0] + 2, p0[1] + 2), (p2[0] + 2, p2[1] + 2), col, 2, cv2.LINE_AA, tipLength=0.3)
            put(img, f"{g:+.1f}/{pr:+.1f}", (p0[0] + 6, p0[1] - 6), 0.34, (255, 255, 255))


def boxes(img, scene):
    for o in scene["objs"]:
        if o["cls"] in DYN:
            from osv.scene import box_corners
            c = np.array([to_px(x, y) for x, y in box_corners(o)], np.int32)
            cv2.polylines(img, [c], True, (255, 255, 255), 1, cv2.LINE_AA)


def frame_panel(canvas, img, x0, y0, title, col=COL["txt"], sub=None):
    canvas[y0:y0 + P, x0:x0 + P] = img
    cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + P, y0 + P), col, 1)
    put(canvas, title, (x0 + 2, y0 - 6), 0.42, col)
    if sub:
        put(canvas, sub, (x0 + 4, y0 + P - 6), 0.36, (210, 210, 210))


def graph(canvas, x0, y0, w, h, series, ymax, title, labels):
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
    put(canvas, title, (x0 + 6, y0 + 16), 0.42)
    gx, gy, gw, gh = x0 + 34, y0 + 26, w - 44, h - 62
    for v in np.linspace(0, ymax, 5):
        yy = int(gy + gh - v / ymax * gh)
        cv2.line(canvas, (gx, yy), (gx + gw, yy), (48, 48, 48), 1)
        put(canvas, f"{v:.0f}", (x0 + 6, yy + 4), 0.33, COL["dim"])
    n = 60
    for (vals, col, dashed) in series:
        pts = [(int(gx + i / (n - 1) * gw), int(gy + gh - min(v, ymax) / ymax * gh)) for i, v in enumerate(vals)
               if np.isfinite(v)]
        for i in range(len(pts) - 1):
            if not dashed or i % 2 == 0:
                cv2.line(canvas, pts[i], pts[i + 1], col, 2, cv2.LINE_AA)
    for j, (lab, col) in enumerate(labels):
        put(canvas, lab, (x0 + 8 + j * (w // len(labels)), y0 + h - 10), 0.36, col)


def table(canvas, x0, y0, w, h, S):
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
    M = S["metrics"]
    put(canvas, f"held-out {S['n_scenes']} synthetic scenes", (x0 + 6, y0 + 16), 0.42)
    rows = [("", "base", "osv"), ("occ mIoU <=12.8 m", "occ_miou_near", None), ("dyn class acc", "dyn_macro_acc", None),
            ("vel MAE all (m/s)", "vel_mae_all", None), ("  aliased cells", "vel_mae_ali", None),
            ("  unaliased cells", "vel_mae_una", None), ("ego speed MAE", "ego_mae", None)]
    put(canvas, "baseline", (x0 + 150, y0 + 40), 0.38, COL["base"])
    put(canvas, "RadarOSV", (x0 + 226, y0 + 40), 0.38, COL["osv"])
    for i, (lab, key, _) in enumerate(rows[1:]):
        yy = y0 + 62 + i * 24
        put(canvas, lab, (x0 + 6, yy), 0.36, COL["dim"])
        b = "ext." if key == "ego_mae" else f"{M['base'][key]:.2f}"
        put(canvas, b, (x0 + 160, yy), 0.4, COL["base"])
        put(canvas, f"{M['osv'][key]:.2f}", (x0 + 236, yy), 0.4, COL["osv"])
    put(canvas, "baseline is GIVEN the true ego speed", (x0 + 6, y0 + h - 30), 0.34, COL["dim"])
    put(canvas, "RadarOSV estimates it from radar", (x0 + 6, y0 + h - 12), 0.34, COL["dim"])


@torch.no_grad()
def run_frame(scene, rng, models):
    img, sc = render(scene, rng)
    gt = rasterize(scene)
    ti, ts = torch.from_numpy(img.astype(np.float32))[None], torch.from_numpy(sc.astype(np.float32))[None]
    ve = torch.tensor([scene["v_ego"]], dtype=torch.float32)
    pr = {n: {k: v[0].numpy() for k, v in m.predict(ti, ts, ve).items()} for n, m in models.items()}
    dyn = np.isin(gt["sem"], DYN) & FOV_MASK
    err = {n: float(np.abs(pr[n]["v_gr"] - gt["vgr"])[dyn].mean()) if dyn.any() else np.nan for n in pr}
    return img, sc, gt, pr, err


def score_seed(seed, models, n=8):
    rng = np.random.default_rng(seed)
    s = make_scene(rng)
    e = []
    for _ in range(n):
        e.append(run_frame(s, rng, models)[4])
        s = step_scene(s, 0.1, rng)
    return {k: float(np.nanmean([x[k] for x in e])) for k in models}


def scene_clip(seed, models, S, label, note, frames, n=60):
    rng = np.random.default_rng(seed)
    s = make_scene(rng)
    hist = dict(base=[], osv=[], ego=[], ego_hat=[])
    for f in range(n):
        img, sc, gt, pr, err = run_frame(s, rng, models)
        hist["base"].append(err["base"]); hist["osv"].append(err["osv"])
        hist["ego"].append(s["v_ego"]); hist["ego_hat"].append(float(pr["osv"]["v_ego"]))
        objs = obj_masks(s)
        cv = np.full((720, 1280, 3), 22, np.uint8)
        X = [12, 328, 644, 960]
        frame_panel(cv, fan(img, RI, AI, DI, VI), X[0], 40, "INPUT 1: imaging radar (aliases > 2.5 m/s)",
                    sub="fine range/azimuth, 16 Doppler bins")
        frame_panel(cv, fan(sc, RS, AS, DS, VS), X[1], 40, "INPUT 2: single-chip radar (+-24 m/s)",
                    sub="coarse, sparse, unaliased Doppler")
        for x0, name, title in ((X[2], "base", "BASELINE: imaging + external ego"),
                                (X[3], "osv", "RadarOSV-style: both radars, radar ego")):
            b = bev(pr[name]["sem"])
            boxes(b, s)
            arrows(b, objs, gt["vgr"], pr[name]["v_gr"], COL[name])
            frame_panel(cv, b, x0, 40, title, COL[name], sub=f"dynamic-cell vel MAE {err[name]:.2f} m/s")
        g = bev(gt["sem"])
        boxes(g, s)
        arrows(g, objs, gt["vgr"], None, None)
        frame_panel(cv, g, X[0], 384, "GROUND TRUTH: occupancy + radial velocity", COL["gt"],
                    sub=f"ego speed {s['v_ego']:.1f} m/s (truth)")
        graph(cv, X[1], 372, 304, 316, [(hist["base"], COL["base"], False), (hist["osv"], COL["osv"], False)],
              8.0, "radial-velocity MAE on dynamic cells (m/s)", [("baseline", COL["base"]), ("RadarOSV", COL["osv"])])
        graph(cv, X[2], 372, 304, 316, [(hist["ego"], COL["gt"], False), (hist["ego_hat"], COL["osv"], True)],
              12.0, "ego speed (m/s): truth vs radar estimate", [("truth", COL["gt"]), ("RadarOSV est.", COL["osv"])])
        table(cv, X[3], 372, 304, 316, S)
        put(cv, f"{label}: {note}   t = {f * 0.1:.1f} s", (12, 18), 0.5, (120, 220, 255))
        leg = "  ".join(f"{c}" for c in CLASSES[1:])
        put(cv, "cells: grey static | orange car | dark orange truck | pink cyclist | yellow pedestrian    "
                "white box = TRUE object | green arrow = TRUE radial velocity | label = gt/pred m/s", (12, 702), 0.36,
            (200, 200, 200))
        put(cv, "SYNTHETIC radar scenes, my re-implementation from the abstract of arXiv 2611.30078 - "
                "not the authors' model, data or radars", (12, 716), 0.36, COL["dim"])
        frames.append(cv)
        s = step_scene(s, 0.1, rng)
    return dict(seed=seed, base=float(np.nanmean(hist["base"])), osv=float(np.nanmean(hist["osv"])),
                ego_err=float(np.mean(np.abs(np.array(hist["ego"]) - hist["ego_hat"]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/radarosv_simulation.gif")
    ap.add_argument("--mp4", default="")
    ap.add_argument("--pool", type=int, default=120)
    a = ap.parse_args()
    torch.set_num_threads(2)
    S = json.load(open(ROOT / "results" / "summary.json"))
    models = {"base": load("base", 0), "osv": load("osv", 0)}
    # clip selection rules, fixed before rendering:
    #   scene 1 RANDOM = seed 82
    #   scene 2 WIN    = median (by baseline - RadarOSV error) of pool seeds where RadarOSV is better by > 1 m/s
    #   scene 3 LOSS   = median of pool seeds where RadarOSV is WORSE; if none, the seed with the smallest advantage
    pool = {sd: score_seed(sd, models) for sd in range(1000, 1000 + a.pool)}
    diff = {sd: v["base"] - v["osv"] for sd, v in pool.items() if np.isfinite(v["base"] - v["osv"])}
    wins = sorted([sd for sd, d in diff.items() if d > 1.0], key=diff.get)
    losses = sorted([sd for sd, d in diff.items() if d < 0], key=diff.get)
    win = wins[len(wins) // 2]
    if losses:
        loss, loss_note = losses[len(losses) // 2], f"SELECTED: RadarOSV WORSE than baseline ({len(losses)} of {len(diff)} pool scenes)"
    else:
        loss = min(diff, key=diff.get)
        loss_note = f"SELECTED: RadarOSV's SMALLEST advantage (no losses in {len(diff)} pool scenes)"
    frames = []
    title = np.full((720, 1280, 3), 22, np.uint8)
    put(title, "Day 82 - RadarOSV: one radar sees detail, the other sees speed", (210, 280), 0.95, COL["txt"], 2)
    put(title, "imaging radar alone vs imaging + single-chip radar with velocity-candidate selection", (190, 330), 0.68,
        COL["osv"], 2)
    put(title, "SYNTHETIC demo - toy re-implementation of arXiv 2611.30078", (340, 380), 0.68, COL["dim"], 1)
    frames += [title] * 15
    info = [scene_clip(82, models, S, "scene 1", "RANDOM scene (seed 82)", frames)]
    info.append(scene_clip(win, models, S, "scene 2", f"SELECTED: RadarOSV better by > 1 m/s ({len(wins)} of {len(diff)} pool scenes)", frames))
    info.append(scene_clip(loss, models, S, "scene 3", loss_note, frames))
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave(out, [cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA) for f in rgb[::2]], duration=0.2, loop=0)
    if a.mp4:
        imageio.mimsave(ROOT / a.mp4, rgb, fps=10, codec="libx264", quality=7, macro_block_size=8)
    cv2.imwrite(str(out.with_name("still.png")), frames[15 + 30])
    clips = dict(info=info, n_pool=len(diff), n_win=len(wins), n_loss=len(losses),
                 pool_mean_base=float(np.mean([v["base"] for v in pool.values()])),
                 pool_mean_osv=float(np.mean([v["osv"] for v in pool.values()])))
    json.dump(clips, open(ROOT / "results" / "clips.json", "w"), indent=1)
    print(f"{len(frames)} frames -> {out}")
    print(json.dumps(clips, indent=1))


if __name__ == "__main__":
    main()
