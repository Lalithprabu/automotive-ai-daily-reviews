"""Real-time synthetic "AI in action" clip: one world model forecasting RGB, thermal and LiDAR 2 s ahead in fog and snow.

Layout 1280 x 720 (BGR, 10 fps):
    left 4 x 3 grid of sensor tiles (224 x 168 each)
        rows     RGB | thermal | LiDAR depth (16 beams, near = red, far = blue)
        columns  INPUT (observed, then frozen at t = 0) | GROUND TRUTH future | TRI-SENSOR forecast | SINGLE-SENSOR forecast
        overlays green boxes + dotted tracks = true cars / pedestrians; cyan boxes = hot blobs found in the forecast
                 thermal frame (a crude detector run on the model's own output)
    right panel  LiDAR error vs horizon for this clip (tri / LiDAR-only / persistence), camera error, held-out table
    bottom       legend + SYNTHETIC disclaimer

Acts per scene: OBSERVE (6 frames, 1.5 s) -> FORECAST (8 frames, 2 s, model sees nothing new) -> HOLD (errors).
Scene 3 replays the snow clip with the RGB tile blanked in the tri model's input.

    python simulate.py --out assets/trimodal_simulation.gif --mp4 assets/trimodal_simulation.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from evaluate import forecast, load_model
from train import load_split
from twm.data import CAR, D_MAX, DT, PED, T_CTX, T_FUT, WEATHERS
from twm.model import TILE_W

ROOT = Path(__file__).resolve().parent
TS = 3.5                                   # tile scale: 64 x 48 -> 224 x 168
TW, TH = int(64 * TS), int(48 * TS)
X0, Y0 = 8, 34
TXT, DIM = (235, 235, 235), (150, 150, 150)
C = dict(gt=(90, 220, 90), tri=(255, 170, 50), single=(70, 70, 235), per=(160, 160, 160), det=(255, 255, 0))
SUB = 3                                    # render frames per 0.25 s sensor frame


def put(img, txt, org, s=0.42, col=TXT, th=1):
    cv2.putText(img, txt, (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, s, col, th, cv2.LINE_AA)


def rgb_img(t3):                           # (3, 48, 64) [0,1] RGB -> BGR uint8 scaled
    a = (np.clip(np.moveaxis(t3, 0, -1), 0, 1) * 255).astype(np.uint8)[..., ::-1]
    return cv2.resize(a, (TW, TH), interpolation=cv2.INTER_NEAREST)


def th_img(t3):
    g = (np.clip(t3.mean(0), 0, 1) * 255).astype(np.uint8)
    return cv2.resize(cv2.applyColorMap(g, cv2.COLORMAP_INFERNO), (TW, TH), interpolation=cv2.INTER_NEAREST)


def lidar_img(t3, thr=0.5):
    d = np.clip(t3[:2].mean(0), 0, 1)
    v = t3[2] > thr
    col = cv2.applyColorMap((np.clip(d * D_MAX / 40.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    col = col[..., ::-1].copy()            # near = red, far = blue
    col[~v] = 18
    big = cv2.resize(col, (TW, TH), interpolation=cv2.INTER_NEAREST)
    return big


def hot_boxes(t3, thr=0.58):
    g = t3.mean(0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((g > thr).astype(np.uint8), 8)
    return [tuple(stats[i, :4]) for i in range(1, n) if stats[i, 4] >= 2]


def draw_boxes(img, boxes, col, th=1):
    for (x, y, w, h) in boxes:
        cv2.rectangle(img, (int(x * TS), int(y * TS)), (int((x + w) * TS), int((y + h) * TS)), col, th, cv2.LINE_AA)


def gt_boxes(d, si, t):
    out = []
    for k, c in enumerate(d["cls"][si]):
        if c in (CAR, PED):
            b = d["boxes"][si, t, k]
            if b[0] >= 0:
                out.append((b[0], b[1], b[2] - b[0], b[3] - b[1], k))
    return out


def tracks(d, si, t0, t1):
    """Box bottom-centre tracks (pixels, scaled) for frames t0..t1."""
    tr = {}
    for t in range(t0, t1 + 1):
        for (x, y, w, h, k) in gt_boxes(d, si, t):
            tr.setdefault(k, []).append(((x + w / 2) * TS, (y + h) * TS))
    return tr


def render_scene(si, d, mt, ms, S, label, frames, blank=None, note=""):
    wx = WEATHERS[d["weather"][si]]
    idx = np.array([si])
    tri = forecast(mt, d, [0, 1, 2], idx, blank_tile=blank)[0]                       # (8, 3, 48, 192)
    single = [forecast(ms[s], d, [s], idx)[0] for s in range(3)]                       # 3 x (8, 3, 48, 64)
    from train import batch_tiles
    obs = batch_tiles(d, idx, [0, 1, 2]).numpy()[0]                                    # (14, 3, 48, 192)
    # per-frame LiDAR error for this clip (measured returns)
    g = d["depth"][si, T_CTX:] / 100.0
    v = d["valid"][si, T_CTX:]
    vl = d["valid"][si, T_CTX - 1]
    vc = v & vl[None]                       # returns measured now AND in the last observed frame (persistence defined)
    def lerr(p):
        e = np.abs(np.clip(p[:, :2].mean(1), 0, 1) * D_MAX - g)
        return np.array([e[t][vc[t]].mean() for t in range(T_FUT)])
    e_tri, e_lid = lerr(tri[..., 2 * TILE_W:]), lerr(single[2])
    per = np.repeat(obs[T_CTX - 1:T_CTX, :, :, 2 * TILE_W:], T_FUT, 0)
    e_per = lerr(per)
    rg = lambda p: np.abs(np.clip(p, 0, 1) - obs[T_CTX:, :, :, :TILE_W]).mean((1, 2, 3)) * 255
    e_rgb_tri, e_rgb_s = rg(tri[..., :TILE_W]), rg(single[0])
    n_obs, n_fc, n_hold = T_CTX * SUB, T_FUT * SUB, 20
    for fr in range(n_obs + n_fc + n_hold):
        img = np.full((720, 1280, 3), 22, np.uint8)
        if fr < n_obs:
            phase, t = "OBSERVE", fr // SUB
        elif fr < n_obs + n_fc:
            phase, t = "FORECAST", T_CTX + (fr - n_obs) // SUB
        else:
            phase, t = "HOLD", T_CTX + T_FUT - 1
        k = t - T_CTX                                                                  # forecast index
        t_in = min(t, T_CTX - 1)
        heads = ["INPUT (observed)", "GROUND TRUTH", "TRI-SENSOR model", "SINGLE-SENSOR models"]
        for c, h in enumerate(heads):
            put(img, h, (X0 + c * (TW + 8) + 4, Y0 - 8), 0.45, [TXT, C["gt"], C["tri"], C["single"]][c])
        for r, (nm, fn) in enumerate((("RGB", rgb_img), ("thermal", th_img), ("LiDAR", lidar_img))):
            sl = slice(r * TILE_W, (r + 1) * TILE_W)
            tiles = [fn(obs[t_in, :, :, sl])]
            tiles.append(fn(obs[t, :, :, sl]) if phase != "OBSERVE" else np.full((TH, TW, 3), 30, np.uint8))
            if phase == "OBSERVE":
                tiles += [np.full((TH, TW, 3), 30, np.uint8)] * 2
            else:
                tp = tri[k, :, :, sl]
                if blank == r:
                    tp = np.zeros_like(tp)
                tiles += [fn(tp), fn(single[r][k])]
            for c, tl in enumerate(tiles):
                tl = tl.copy()
                if c == 0:
                    draw_boxes(tl, [b[:4] for b in gt_boxes(d, si, t_in)], C["gt"])
                if c == 1 and phase != "OBSERVE":
                    draw_boxes(tl, [b[:4] for b in gt_boxes(d, si, t)], C["gt"])
                    for pts in tracks(d, si, T_CTX, t).values():
                        for p in pts:
                            cv2.circle(tl, (int(p[0]), int(p[1])), 1, C["gt"], -1)
                if c >= 2 and phase != "OBSERVE" and r < 2:
                    src = tri[k, :, :, TILE_W:2 * TILE_W] if c == 2 else (single[1][k] if r == 1 else None)
                    if src is not None and not (c == 2 and blank == 1):
                        draw_boxes(tl, hot_boxes(src), C["det"])
                if phase == "OBSERVE" and c >= 1:
                    put(tl, "waiting for 1.5 s of input", (10, TH // 2), 0.45, DIM)
                if c == 2 and blank == r and phase != "OBSERVE":
                    put(tl, "BLANKED INPUT", (40, TH // 2), 0.6, (80, 80, 255), 2)
                y = Y0 + r * (TH + 8)
                x = X0 + c * (TW + 8)
                img[y:y + TH, x:x + TW] = tl
                cv2.rectangle(img, (x - 1, y - 1), (x + TW, y + TH), (70, 70, 70), 1)
            put(img, nm, (X0 + 4, Y0 + r * (TH + 8) + 16), 0.45, (255, 255, 255), 1)
        # --------------- right panel
        px = X0 + 4 * (TW + 8) + 4
        pw = 1280 - px - 8
        clock = (t - (T_CTX - 1)) * DT
        put(img, f"{label}  [{phase}]", (px, 22), 0.5)
        put(img, f"weather: {wx.upper()}   t = {clock:+.2f} s", (px, 44), 0.48, (120, 220, 255))
        if note:
            put(img, note, (px, 64), 0.4, (80, 80, 255))
        # LiDAR error plot
        gy, gh = 80, 170
        cv2.rectangle(img, (px, gy), (px + pw, gy + gh), (70, 70, 70), 1)
        put(img, "LiDAR error this clip [m], common returns", (px + 4, gy + 16), 0.4)
        ymax = max(0.5, float(np.nanmax([e_tri.max(), e_lid.max(), e_per.max()])) * 1.3)
        fx = lambda i: px + 30 + i * (pw - 40) / (T_FUT - 1)
        fy = lambda e: gy + gh - 14 - np.clip(e / ymax, 0, 1) * (gh - 62)
        put(img, f"{ymax:.1f}", (px + 2, fy(ymax) + 4), 0.32, DIM)
        put(img, "0", (px + 14, fy(0) + 4), 0.32, DIM)
        put(img, "+0.25 s", (px + 18, gy + gh - 2), 0.3, DIM)
        put(img, "+2 s", (px + pw - 34, gy + gh - 2), 0.3, DIM)
        n_show = (k + 1) if phase == "FORECAST" else (T_FUT if phase == "HOLD" else 0)
        for e, col in ((e_per, C["per"]), (e_lid, C["single"]), (e_tri, C["tri"])):
            pts = np.array([(fx(i), fy(e[i])) for i in range(n_show)], np.int32)
            if len(pts) > 1:
                cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, tuple(p), 2, col, -1)
        if n_show:
            j = n_show - 1
            put(img, f"tri {e_tri[j]:.2f}  LiDAR-only {e_lid[j]:.2f}  persist {e_per[j]:.2f}", (px + 4, gy + 32), 0.36)
        # RGB error
        ry, rh = gy + gh + 8, 104
        cv2.rectangle(img, (px, ry), (px + pw, ry + rh), (70, 70, 70), 1)
        put(img, "RGB error this clip [grey levels]", (px + 4, ry + 16), 0.4)
        if n_show:
            j = n_show - 1
            if blank == 0:
                put(img, "tri: RGB input blanked -> no RGB forecast", (px + 8, ry + 40), 0.4, (80, 80, 255))
            bars = list(enumerate((("tri-sensor", e_rgb_tri, C["tri"]), ("RGB-only", e_rgb_s, C["single"]))))
            for i, (nm, e, col) in bars[(1 if blank == 0 else 0):]:
                put(img, f"{nm:<11} {e[j]:5.1f}", (px + 8, ry + 40 + 20 * i), 0.42, col)
                bw = int(np.clip(e[j] / 40, 0, 1) * (pw - 150))
                cv2.rectangle(img, (px + 140, ry + 30 + 20 * i), (px + 140 + bw, ry + 42 + 20 * i), col, -1)
            put(img, "(lower is better; the price of joint training)", (px + 8, ry + 92), 0.33, DIM)
        # held-out table
        ty = ry + rh + 8
        cv2.rectangle(img, (px, ty), (px + pw, 604), (70, 70, 70), 1)
        M = S["metrics"]
        put(img, f"held-out {S['n_clips']} clips, error at +2 s", (px + 4, ty + 16), 0.4)
        put(img, "            LiDAR m   obj m  RGB  therm", (px + 4, ty + 36), 0.36, (170, 170, 170))
        rows = (("tri-sensor", "tri", C["tri"]), ("LiDAR-only", "lidar", C["single"]),
                ("RGB-only", "rgb", C["single"]), ("thermal-only", "thermal", C["single"]),
                ("persistence", None, C["per"]))
        yy = ty + 54
        for nm, key, col in rows:
            if key is None:
                lm = M["persist_lidar"]["lidar_all_2s"]; om = M["persist_lidar"]["lidar_obj_2s"]
                rm = M["persist_rgb"]["rgb_mae_2s"]; tm = M["persist_thermal"]["thermal_mae_2s"]
            else:
                r = M[key]
                lm, om = r.get("lidar_all_2s", np.nan), r.get("lidar_obj_2s", np.nan)
                rm, tm = r.get("rgb_mae_2s", np.nan), r.get("thermal_mae_2s", np.nan)
            f = lambda v, w=5, p=2: (" " * (w - 1) + "-") if not np.isfinite(v) else f"{v:{w}.{p}f}"
            put(img, f"{nm:<12}{f(lm)}  {f(om)} {f(rm, 5, 1)} {f(tm, 5, 1)}", (px + 4, yy), 0.36, col)
            yy += 18
        put(img, "persistence LiDAR = last frame, on returns", (px + 4, yy + 4), 0.32, DIM)
        put(img, "measured in both frames", (px + 4, yy + 18), 0.32, DIM)
        # bottom strip
        put(img, "green boxes + dots = true cars/pedestrians and their future track | cyan boxes = hot blobs detected in the "
                 "model's forecast thermal frame", (8, 636), 0.4, (200, 200, 200))
        put(img, "LiDAR tiles: 16 beams projected into the RGB camera, near = red, far = blue, black = no return | "
                 "the model sees nothing after t = 0", (8, 656), 0.4, (200, 200, 200))
        put(img, f"{label}: held-out clip #{si} ({wx}), picked as the MEDIAN tri-sensor LiDAR error of its weather, not the best",
            (8, 680), 0.4, (180, 180, 180))
        put(img, "SYNTHETIC 3-sensor driving simulator, my re-implementation from the abstract of arXiv 2611.22669 - "
                 "not DSERT-RoLL, not the authors' model or code", (8, 704), 0.4, (150, 150, 150))
        frames.append(img)
    return dict(clip=int(si), weather=wx, lidar_2s=dict(tri=float(e_tri[-1]), lidar=float(e_lid[-1]), persist=float(e_per[-1])),
                rgb_2s=dict(tri=float(e_rgb_tri[-1]), rgb=float(e_rgb_s[-1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/trimodal_simulation.gif")
    ap.add_argument("--mp4", default="")
    ap.add_argument("--gif_every", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(2)
    d = load_split("test")
    S = json.load(open(ROOT / "results/summary.json"))
    mt, _ = load_model("tri")
    ms = [load_model(r)[0] for r in ("rgb", "thermal", "lidar")]
    frames = []
    title = np.full((720, 1280, 3), 22, np.uint8)
    put(title, "Day 79 - One world model, three sensors, 2 seconds ahead", (250, 280), 1.0, TXT, 2)
    put(title, "RGB + thermal + LiDAR depth forecast as ONE video, in fog and snow", (240, 330), 0.8, C["tri"], 2)
    put(title, "SYNTHETIC demo - toy re-implementation of arXiv 2611.22669", (330, 380), 0.7, (160, 160, 160), 1)
    frames += [title] * 15
    info = [render_scene(S["clips"]["fog"], d, mt, ms, S, "scene 1", frames),
            render_scene(S["clips"]["snow"], d, mt, ms, S, "scene 2", frames),
            render_scene(S["clips"]["snow"], d, mt, ms, S, "scene 3", frames, blank=0,
                         note="tri model's RGB input blanked")]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave(out, [cv2.resize(f, (960, 540), interpolation=cv2.INTER_AREA) for f in rgb[::a.gif_every]],
                    duration=0.1 * a.gif_every, loop=0)
    if a.mp4:
        imageio.mimsave(ROOT / a.mp4, rgb, fps=10, codec="libx264", quality=7, macro_block_size=8)
    still = 15 + T_CTX * SUB + T_FUT * SUB + 5
    cv2.imwrite(str(out.with_name("still.png")), frames[still])
    print(f"{len(frames)} frames -> {out}")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
