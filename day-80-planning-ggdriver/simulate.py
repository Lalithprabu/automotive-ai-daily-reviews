"""Real-time synthetic "AI in action" clip: glance-only planner vs. a GGDriver-style gaze planner, side by side.

Layout 1280 x 720 (BGR, 10 fps):
    top-left   camera, GLANCE-ONLY planner (old way): its 8 hypotheses, its pick (red), true objects (green boxes)
    top-right  camera, GAZE planner: its 8 hypotheses, the corridor patches each candidate gazes at (cyan cells),
               its pick (blue); both views show the best trajectory under the rule score (dotted green)
    bottom     BEV of the scene with both picks | predicted vs true score of each planner's 8 proposals |
               held-out table
Acts per scene: GLANCE (8 hypotheses) -> GAZE (corridor masks, one candidate at a time) -> SCORE (pick) ->
EXECUTE (4 s: the camera drives each planner's own trajectory through the static scene; crashes flash red) -> HOLD.

    python simulate.py --out assets/ggdriver_simulation.gif --mp4 assets/ggdriver_simulation.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from evaluate import LABEL, load_model
from ggd.model import corridor_mask
from ggd.scene import (CAM_H, CX, CY, EGO_FRONT, EGO_REAR, EGO_W, F_PX, GW, LANE, OBJ_NAMES, PATCH, SCENE_TYPES,
                       render, scene_from, score_traj, traj_points)
from train import load_split, to_batch

ROOT = Path(__file__).resolve().parent
S_CAM = 3.75                                        # 160 x 96 -> 600 x 360
CW, CH = int(160 * S_CAM), int(96 * S_CAM)
CAMS = [(20, 50), (660, 50)]
TXT, DIM = (235, 235, 235), (150, 150, 150)
COL = dict(glance=(60, 60, 235), gaze=(255, 160, 40), gt=(90, 220, 90), patch=(255, 255, 0), crash=(40, 40, 255))


def put(img, txt, org, s=0.45, col=TXT, th=1):
    cv2.putText(img, txt, (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)


def proj(X, Z, Y=0.0):
    """ego-frame ground point(s) -> camera pixel (scaled)."""
    Zc = np.maximum(Z, 0.5)
    return (CX + F_PX * X / Zc) * S_CAM, (CY + F_PX * (CAM_H - Y) / Zc) * S_CAM


def draw_path(img, pts, ex, ez, col, th=2, dotted=False, corridor=False):
    p = np.concatenate([[[ex, ez]], pts], 0)
    rel = p - [ex, ez]
    keep = rel[:, 1] > 1.2
    if keep.sum() < 2:
        return
    u, v = proj(rel[keep, 0], rel[keep, 1])
    q = np.stack([u, v], 1).astype(np.int32)
    if dotted:
        for a in q[::1]:
            cv2.circle(img, tuple(a), 2, col, -1, cv2.LINE_AA)
    else:
        cv2.polylines(img, [q], False, col, th, cv2.LINE_AA)
    if corridor:
        for s in (-EGO_W / 2, EGO_W / 2):
            u2, v2 = proj(rel[keep, 0] + s, rel[keep, 1])
            cv2.polylines(img, [np.stack([u2, v2], 1).astype(np.int32)], False, col, 1, cv2.LINE_AA)


def obj_boxes(sc, ex, ez):
    out = []
    for i in range(len(sc["typ"])):
        if sc["typ"][i] < 0:
            continue
        x, z, w, l = sc["obj"][i]
        x, z = x - ex, z - ez - l / 2
        if z < 1.0:
            continue
        from ggd.scene import OBJ
        h = OBJ[OBJ_NAMES[sc["typ"][i]]][2]
        u0, vb = proj(x - w / 2, z)
        u1, vt = proj(x + w / 2, z, h)
        out.append((i, int(u0) - 2, int(vt) - 2, int(u1) + 2, int(vb) + 2))
    return out


def interp(pts, t):
    tt = np.concatenate([[0], np.arange(1, 17) * 0.25])
    p = np.concatenate([[[0, 0]], pts], 0)
    return np.array([np.interp(t, tt, p[:, 0]), np.interp(t, tt, p[:, 1])])


def crash_time(pts, sc):
    """first time (s) the ego box overlaps an object, else None; 0.05 s resolution."""
    for t in np.arange(0, 4.001, 0.05):
        x, z = interp(pts, t)
        for i in range(len(sc["typ"])):
            if sc["typ"][i] < 0:
                continue
            ox, oz, w, l = sc["obj"][i]
            if abs(x - ox) < (EGO_W + w) / 2 and oz - l / 2 < z + EGO_FRONT and oz + l / 2 > z - EGO_REAR:
                return float(t)
    return None


def bev(img, x0, y0, w, h, sc, picks, t):
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
    sx, zr = w / 22.0, (h - 10) / 75.0
    P = lambda X, Z: (int(x0 + w / 2 + X * sx), int(y0 + h - 8 - (Z + 4) * zr))
    L, R = sc["edges"]
    a, b = P(L, -4), P(R, 72)
    cv2.rectangle(img, (a[0], b[1]), (b[0], a[1]), (60, 60, 62), -1)
    for i in range(1, sc["n"]):
        xx = L + i * LANE
        for zz in range(-4, 72, 8):
            cv2.line(img, P(xx, zz), P(xx, zz + 4), (200, 200, 200), 1)
    for xx in (L, R):
        cv2.line(img, P(xx, -4), P(xx, 72), (230, 230, 230), 1)
    for i in range(len(sc["typ"])):
        if sc["typ"][i] < 0:
            continue
        ox, oz, ow, ol = sc["obj"][i]
        nm = OBJ_NAMES[sc["typ"][i]]
        col = {"car": (200, 200, 200), "cone": (0, 140, 255), "debris": (60, 110, 170), "barrel": (50, 80, 230),
               "ped": (220, 120, 220)}[nm]
        cv2.rectangle(img, P(ox - max(ow, 0.5) / 2, oz - max(ol, 0.5) / 2), P(ox + max(ow, 0.5) / 2, oz + max(ol, 0.5) / 2),
                      col, -1)
    for nm, (pts, ct) in picks.items():
        p = np.concatenate([[[0, 0]], pts], 0)
        cv2.polylines(img, [np.array([P(*q) for q in p], np.int32)], False, COL[nm], 2, cv2.LINE_AA)
        tt = min(t, ct) if ct is not None else t
        ex, ez = interp(pts, tt)
        c = COL["crash"] if (ct is not None and t >= ct) else COL[nm]
        cv2.rectangle(img, P(ex - EGO_W / 2, ez - EGO_REAR), P(ex + EGO_W / 2, ez + EGO_FRONT), c, -1 if c == COL["crash"] else 2)
    put(img, "bird's-eye view (static scene)", (x0 + 6, y0 + 16), 0.4)


def render_scene(si, d, models, S, label, frames, note=""):
    sc = scene_from(d, si)
    b = to_batch(d, np.array([si]))
    P = {}
    for nm, m in models.items():
        p = m.plan(b["img"], b["v0"], 8)
        out = p["out"]
        gt = score_traj(out["pts"], out["dv"], b["v0"], b["obj"], b["valid"], b["edges"], b["zmax"])["score"][0].numpy()
        hyp = traj_points(out["anchors"], b["v0"][:, None])[0].numpy()
        P[nm] = dict(pts=p["pts"][0].numpy(), dv=p["dv"][0].numpy(), best=int(p["best"][0]), pred=out["score"][0].numpy(),
                     true=gt, hyp=hyp, ref=out["pts"][0].numpy(),
                     mask=corridor_mask(out["pts"])[0].numpy())                   # (8, 240) re-projected corridors
        P[nm]["crash"] = crash_time(P[nm]["pts"], sc)
        P[nm]["score"] = float(gt[P[nm]["best"]])
    # best under the rule score in the refined 5x5 grids (the dotted green "what a perfect scorer would pick")
    from ggd.scene import ANCHORS, REF_D, REF_V
    k = int(np.argmax(d["best_score"][si]))
    best_dv = ANCHORS[k] + d["best_delta"][si, k]
    best_pts = traj_points(torch.tensor(best_dv)[None, None], b["v0"][:, None])[0, 0].numpy()
    acts = [("GLANCE", 15), ("GAZE", 24), ("SCORE", 10), ("EXECUTE", 41), ("HOLD", 12)]
    wx = SCENE_TYPES[sc["stype"]]
    for act, n in acts:
        for f in range(n):
            img = np.full((720, 1280, 3), 22, np.uint8)
            t = f * 0.1 if act == "EXECUTE" else (4.0 if act == "HOLD" else 0.0)
            for ci, nm in enumerate(("glance", "gaze")):
                Q = P[nm]
                ct = Q["crash"]
                tt = min(t, ct) if ct is not None else t
                ex, ez = interp(Q["pts"], tt)
                cam = render(sc, ex, ez)
                cam = cv2.resize(cam, (CW, CH), interpolation=cv2.INTER_NEAREST)
                if act == "GAZE" and nm == "gaze":                                 # corridor patches of candidate j
                    j = min(7, f // 3)
                    over = cam.copy()
                    for pi in np.where(Q["mask"][j])[0]:
                        r, c = divmod(pi, GW)
                        cv2.rectangle(over, (int(c * PATCH * S_CAM), int(r * PATCH * S_CAM)),
                                      (int((c + 1) * PATCH * S_CAM) - 1, int((r + 1) * PATCH * S_CAM) - 1), COL["patch"], -1)
                    cam = cv2.addWeighted(over, 0.28, cam, 0.72, 0)
                    draw_path(cam, Q["ref"][j], 0, 0, COL["patch"], 2, corridor=True)
                    put(cam, f"candidate {j + 1}/8 gazes at {int(Q['mask'][j].sum())} of 240 patches -> "
                             f"predicted score {Q['pred'][j]:.2f}", (10, CH - 12), 0.5, COL["patch"])
                elif act == "GAZE":
                    over = cam.copy()
                    over[:] = (120, 120, 120)
                    cam = cv2.addWeighted(over, 0.15, cam, 0.85, 0)
                    put(cam, "all 8 candidates scored on the SAME 9 shared tokens", (10, CH - 12), 0.5, (200, 200, 255))
                if act == "GLANCE":
                    for j in range(8):
                        draw_path(cam, Q["hyp"][j], 0, 0, (180, 180, 180), 1)
                    put(cam, "glance: 8 hypotheses from shared scene features", (10, CH - 12), 0.5, DIM)
                for (i, u0, v0, u1, v1) in obj_boxes(sc, ex, ez):
                    cv2.rectangle(cam, (u0, v0), (u1, v1), COL["gt"], 1, cv2.LINE_AA)
                if act in ("SCORE", "EXECUTE", "HOLD"):
                    draw_path(cam, best_pts, ex, ez, COL["gt"], dotted=True)
                    draw_path(cam, Q["pts"], ex, ez, COL[nm], 3, corridor=True)
                if ct is not None and t >= ct:
                    over = cam.copy()
                    over[:] = COL["crash"]
                    cam = cv2.addWeighted(over, 0.35, cam, 0.65, 0)
                    put(cam, f"COLLISION at t = {ct:.2f} s", (CW // 2 - 150, CH // 2), 1.0, (255, 255, 255), 2)
                x0, y0 = CAMS[ci]
                img[y0:y0 + CH, x0:x0 + CW] = cam
                cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + CW, y0 + CH), COL[nm], 2)
                head = "GLANCE-ONLY planner (old way)" if nm == "glance" else "GAZE planner (GGDriver-style)"
                put(img, head, (x0 + 2, y0 - 8), 0.55, COL[nm], 1)
                dd = 0.0 if abs(Q['dv'][0]) < 0.005 else Q['dv'][0]
                pk = f"pick: lateral {dd:+.2f} m, target {Q['dv'][1]:.1f} m/s, true score {Q['score']:.2f}"
                (tw_, th_), _ = cv2.getTextSize(pk, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x0 + 4, y0 + 6), (x0 + 14 + tw_, y0 + 12 + th_ + 6), (25, 25, 25), -1)
                put(img, pk, (x0 + 9, y0 + 23), 0.5, (255, 255, 255), 1)
            # bottom panels
            bev(img, 20, 420, 300, 262, sc, {nm: (P[nm]["pts"], P[nm]["crash"]) for nm in P}, t)
            px, py, pw = 336, 420, 470
            cv2.rectangle(img, (px, py), (px + pw, py + 262), (70, 70, 70), 1)
            put(img, "8 refined proposals: predicted score (bar) vs true rule score (|)", (px + 6, py + 16), 0.4)
            for ci, nm in enumerate(("glance", "gaze")):
                Q = P[nm]
                cx0 = px + 8 + ci * (pw // 2)
                put(img, "glance-only" if nm == "glance" else "gaze", (cx0, py + 36), 0.42, COL[nm])
                for j in range(8):
                    yy = py + 46 + j * 26
                    bw = int(np.clip(Q["pred"][j], 0, 1) * (pw // 2 - 70))
                    cv2.rectangle(img, (cx0 + 22, yy), (cx0 + 22 + bw, yy + 14), COL[nm], -1)
                    tx = cx0 + 22 + int(np.clip(Q["true"][j], 0, 1) * (pw // 2 - 70))
                    cv2.line(img, (tx, yy - 3), (tx, yy + 17), COL["gt"], 2)
                    sel = act in ("SCORE", "EXECUTE", "HOLD") and j == Q["best"]
                    put(img, ">" if sel else f"{j + 1}", (cx0 + 4, yy + 12), 0.42, (255, 255, 255) if sel else DIM)
                    if sel:
                        cv2.rectangle(img, (cx0 + 20, yy - 2), (cx0 + pw // 2 - 46, yy + 16), (255, 255, 255), 1)
            tx0, ty0, tw = 822, 420, 438
            cv2.rectangle(img, (tx0, ty0), (tx0 + tw, ty0 + 262), (70, 70, 70), 1)
            M = S["metrics"]
            put(img, f"held-out {S['n_clips']} scenes", (tx0 + 6, ty0 + 16), 0.42)
            put(img, "planner                    score  crash  small-haz", (tx0 + 6, ty0 + 36), 0.38, (170, 170, 170))
            rows = [("keep lane at current speed", "keep_lane", DIM), ("glance only", "glance", COL["glance"]),
                    ("gaze: refine only", "gaze_refine", TXT), ("gaze: score only", "gaze_score", TXT),
                    ("gaze: refine + score", "gaze", COL["gaze"]), ("random patches", "random", TXT),
                    ("all 240 patches", "all", TXT), ("oracle (best refined)", "oracle_refined", COL["gt"])]
            yy = ty0 + 56
            for nm_, key, col in rows:
                if key not in M:
                    continue
                r = M[key]
                cr = r.get("collision_rate")
                put(img, f"{nm_:<26} {r['score']:.3f}  {'  -  ' if cr is None else f'{cr * 100:4.1f}%'}  "
                         f"{r['score_small']:.3f}", (tx0 + 6, yy), 0.38, col)
                yy += 20
            put(img, "score = NC x DAC x (0.75 progress + 0.25 comfort)", (tx0 + 6, yy + 8), 0.36, DIM)
            # header + footer
            put(img, f"{label}: held-out scene #{si} ({wx} - {note})   ego speed {float(b['v0'][0]):.1f} m/s   "
                     f"[{act}]  t = {t:.1f} s", (20, 20), 0.5, (120, 220, 255))
            put(img, "green boxes = true objects | dotted green = best trajectory under the rule score | cyan cells = "
                     "patches inside the candidate's projected corridor | camera only, no perception labels",
                (20, 698), 0.38, (200, 200, 200))
            put(img, "SYNTHETIC driving scenes, my re-implementation from the abstract of arXiv 2611.27330 - not NAVSIM / "
                     "WOD-E2E / Bench2Drive, not the authors' model or code", (20, 714), 0.38, (150, 150, 150))
            frames.append(img)
    return dict(scene=int(si), type=wx, note=note, **{nm: dict(dv=P[nm]["dv"].round(2).tolist(), score=round(P[nm]["score"], 3),
                                                                crash=P[nm]["crash"]) for nm in P})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/ggdriver_simulation.gif")
    ap.add_argument("--mp4", default="")
    ap.add_argument("--gif_every", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(2)
    d = load_split("test")
    S = json.load(open(ROOT / "results" / "summary.json"))
    models = {"glance": load_model("glance")[0], "gaze": load_model("gaze")[0]}
    frames = []
    title = np.full((720, 1280, 3), 22, np.uint8)
    put(title, "Day 80 - Glance at what is shared, gaze at what differs", (270, 280), 1.0, TXT, 2)
    put(title, "every candidate trajectory picks its own camera patches before it is scored", (230, 330), 0.75,
        COL["gaze"], 2)
    put(title, "SYNTHETIC demo - toy re-implementation of arXiv 2611.27330", (340, 380), 0.7, (160, 160, 160), 1)
    frames += [title] * 15
    C = S["clips"]
    info = [render_scene(C["random_small"], d, models, S, "scene 1", frames, "a RANDOM small-hazard scene")]
    if "gaze_win" in C:
        info.append(render_scene(C["gaze_win"], d, models, S, "scene 2", frames,
                                 f"SELECTED: glance crashes, gaze does not ({C['n_gaze_win']} such scenes)"))
    if "gaze_fail" in C:
        info.append(render_scene(C["gaze_fail"], d, models, S, "scene 3", frames, f"SELECTED: gaze crashes, glance does not ({C['n_gaze_fail_vs_glance_ok']} such scenes)"))
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave(out, [cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA) for f in rgb[::a.gif_every]],
                    duration=0.1 * a.gif_every, loop=0)
    if a.mp4:
        imageio.mimsave(ROOT / a.mp4, rgb, fps=10, codec="libx264", quality=7, macro_block_size=8)
    cv2.imwrite(str(out.with_name("still.png")), frames[15 + 15 + 12])
    print(f"{len(frames)} frames -> {out}")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
