"""Synthetic 'AI in action' simulation: INPUT vs GROUND TRUTH vs PREDICTION overlay for a FreeOcc-style forecaster.

Top row   : [INPUT: newest noisy occupancy frame] [GROUND TRUTH at +tau] [WARP-ONLY baseline at +tau]
Bottom row: [FREEOCC prediction at +tau: green=TP, red=false alarm, blue=missed, boxes = predicted objects]
            [UNCERTAINTY: blend weight w (bright = ODE branch trusted)] [LIVE TELEMETRY: rolling IoU + IoU per horizon]
The cyan line is the planner's INTENDED ego path (the conditioning signal that FreeOcc uses to remove ego-motion).
Usage: python simulate.py --ckpt checkpoints/freeocc.pt --out assets/demo.mp4 [--show]
"""
import argparse, os, subprocess
import cv2, numpy as np, torch
from models.freeocc import FreeOcc
from src.geometry import G, CELL, rel_pose, warp_bev
from src.synthetic import make_scene, render, ego_pose, corrupt_history, HIST, H, DT

S = 7                       # pixels per cell
P = G * S                   # panel size (336)
W, HH = 1280, 800
BG = (22, 18, 16)
C = dict(txt=(235, 235, 235), dim=(150, 150, 150), tp=(90, 220, 110), fp=(70, 70, 240), fn=(240, 150, 60),
         path=(255, 220, 60), ego=(60, 200, 255), blend=(250, 160, 60), dec=(180, 120, 250), warp=(120, 120, 255))


def put(img, s, org, sc=0.5, col=C["txt"], th=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def to_img(m):             # (G,G) array, index (i fwd, j left) -> screen (forward = up, left = left)
    return np.flip(np.flip(m, 0), 1)


def px(x, y):              # metres in ego frame -> pixel in panel
    return int((G / 2 - y / CELL) * S), int((G / 2 - x / CELL) * S)


def panel(img, x0, y0, title, canvas):
    img[y0:y0 + P, x0:x0 + P] = canvas
    cv2.rectangle(img, (x0, y0), (x0 + P, y0 + P), (90, 90, 90), 1)
    put(img, title, (x0 + 6, y0 - 8), 0.52)


def gray(m, col=(235, 235, 235)):
    c = np.zeros((G, G, 3), np.float32); c[:] = np.array(BG)[None, None] * 1.0
    c += to_img(m)[..., None] * (np.array(col) - np.array(BG))[None, None]
    return cv2.resize(c.clip(0, 255).astype(np.uint8), (P, P), interpolation=cv2.INTER_NEAREST)


def draw_path(canvas, fut_rel):
    pts = [px(0, 0)] + [px(float(p[0]), float(p[1])) for p in fut_rel]
    cv2.polylines(canvas, [np.array(pts, np.int32)], False, C["path"], 2, cv2.LINE_AA)
    cv2.circle(canvas, px(0, 0), 5, C["ego"], -1, cv2.LINE_AA)


def draw_boxes(canvas, pred_bin, min_area=6):
    m = to_img(pred_bin).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        if a >= min_area:
            cv2.rectangle(canvas, (x * S, y * S), ((x + w) * S, (y + h) * S), (255, 255, 255), 1, cv2.LINE_AA)


def overlay(pred_bin, gt):
    c = np.zeros((G, G, 3), np.float32); c[:] = BG
    for mask, col in (((pred_bin > 0) & (gt > 0), C["tp"]), ((pred_bin > 0) & (gt == 0), C["fp"]), ((pred_bin == 0) & (gt > 0), C["fn"])):
        c[to_img(mask.astype(np.float32)) > 0] = col
    return cv2.resize(c.astype(np.uint8), (P, P), interpolation=cv2.INTER_NEAREST)


def heat(m):
    h = cv2.applyColorMap((to_img(m) * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.resize(h, (P, P), interpolation=cv2.INTER_NEAREST)


def iou(a, b):
    u = ((a > 0) | (b > 0)).sum(); return float(((a > 0) & (b > 0)).sum() / u) if u else 1.0


def pick_scene(seed):
    g = torch.Generator().manual_seed(seed); best, bs = None, -1
    for _ in range(40):
        sc = make_scene(1, g); sc["w"] = sc["w"].clamp(-0.45, 0.45)
        t = torch.arange(0, 16) * 0.5; occ = render(sc, t)
        score = occ[:, :, 8:40, 8:40].sum().item()          # lots of structure near the ego path
        if score > bs: best, bs = sc, score
    return best


def draw_telemetry(img, x0, y0, roll, per_h, tick, ticks):
    cv2.rectangle(img, (x0, y0), (x0 + 544, y0 + P), BG, -1); cv2.rectangle(img, (x0, y0), (x0 + 544, y0 + P), (90, 90, 90), 1)
    put(img, "LIVE TELEMETRY  (IoU of occupied cells, scene-level)", (x0 + 8, y0 + 20), 0.5)
    ax = (x0 + 36, y0 + 40, 490, 130)
    cv2.rectangle(img, (ax[0], ax[1]), (ax[0] + ax[2], ax[1] + ax[3]), (60, 60, 60), 1)
    for name, key in (("FreeOcc blend", "blend"), ("decoder only", "dec"), ("warp-only", "warp")):
        v = roll[key]
        if len(v) > 1:
            pts = np.array([[ax[0] + int(i / max(ticks - 1, 1) * ax[2]), ax[1] + ax[3] - int(x * ax[3])] for i, x in enumerate(v)], np.int32)
            cv2.polylines(img, [pts], False, C[key], 2, cv2.LINE_AA)
    for j, (n, k) in enumerate((("FreeOcc blend", "blend"), ("decoder only", "dec"), ("warp-only", "warp"))):
        cv2.line(img, (x0 + 40 + 150 * j, y0 + 190), (x0 + 62 + 150 * j, y0 + 190), C[k], 3); put(img, n, (x0 + 66 + 150 * j, y0 + 194), 0.42)
    put(img, "1.0", (x0 + 6, ax[1] + 10), 0.4, C["dim"]); put(img, "0", (x0 + 14, ax[1] + ax[3]), 0.4, C["dim"]); put(img, "time ->", (x0 + 480, ax[1] + ax[3] + 14), 0.4, C["dim"])
    put(img, "IoU per forecast horizon (current tick)", (x0 + 8, y0 + 222), 0.5)
    for h in range(H):
        bx = x0 + 40 + h * 82
        for q, k in enumerate(("warp", "dec", "blend")):
            hgt = int(per_h[k][h] * 70); cv2.rectangle(img, (bx + q * 22, y0 + 312 - hgt), (bx + q * 22 + 18, y0 + 312), C[k], -1)
        put(img, f"+{(h + 1) * DT:.1f}s", (bx + 6, y0 + 328), 0.42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/freeocc.pt"); ap.add_argument("--out", default="assets/demo.mp4")
    ap.add_argument("--seed", type=int, default=11); ap.add_argument("--ticks", type=int, default=36)
    ap.add_argument("--tau", type=int, default=4, help="headline horizon index (1..6) -> +tau*0.5 s")
    ap.add_argument("--drop", type=float, default=0.3, help="history-frame dropout shown in the sim")
    ap.add_argument("--fps", type=int, default=8); ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    model = FreeOcc(); model.load_state_dict(torch.load(a.ckpt)); model.eval()
    sc = pick_scene(a.seed); g = torch.Generator().manual_seed(a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, HH))
    roll = dict(blend=[], dec=[], warp=[]); k = a.tau - 1
    for tick in range(a.ticks):
        t0 = 1.5 + tick * 0.25                                            # sim clock (s); ticks every 0.25 s
        th = t0 + (torch.arange(HIST).float() - (HIST - 1)) * DT; tf = t0 + torch.arange(1, H + 1).float() * DT
        pose_h, pose_f, pose_c = ego_pose(sc, th), ego_pose(sc, tf), ego_pose(sc, torch.tensor([t0]))
        hist = render(sc, th); fut = render(sc, tf)[0]
        hn = corrupt_history(hist, g, drop=a.drop)
        hist_rel, fut_rel = rel_pose(pose_c, pose_h), rel_pose(pose_c, pose_f)
        with torch.no_grad():
            out = model(hn, hist_rel, fut_rel, sample=True, gen=g)
        warp = torch.stack([warp_bev(hn[:, -1:], fut_rel[:, j])[0, 0] for j in range(H)])
        P_ = {"blend": (out["p"][0] > 0.5).float(), "dec": (out["p_dec"][0] > 0.5).float(), "warp": (warp > 0.5).float()}
        per_h = {n: [iou(P_[n][j].numpy(), fut[j].numpy()) for j in range(H)] for n in P_}
        for n in P_: roll[n].append(float(np.mean(per_h[n])))
        img = np.full((HH, W, 3), 12, np.uint8)
        put(img, "FreeOcc-style 4D occupancy forecasting - SYNTHETIC toy world, NOT nuScenes, NOT the paper's weights", (16, 26), 0.6, (140, 200, 255))
        y1, y2 = 62, 62 + P + 52
        cin = gray(hn[0, -1].numpy()); draw_path(cin, fut_rel[0])
        panel(img, 16, y1, f"INPUT: newest noisy frame ({int(a.drop * 100)}% drop)", cin)
        cgt = gray(fut[k].numpy(), (110, 235, 130)); draw_path(cgt, fut_rel[0, :k + 1])
        panel(img, 368, y1, f"GROUND TRUTH at +{a.tau * DT:.1f} s", cgt)
        cw = overlay(P_["warp"][k].numpy(), fut[k].numpy()); panel(img, 720, y1, f"WARP-ONLY baseline +{a.tau * DT:.1f} s (IoU {per_h['warp'][k]:.2f})", cw)
        cp = overlay(P_["blend"][k].numpy(), fut[k].numpy()); draw_boxes(cp, P_["blend"][k].numpy()); draw_path(cp, fut_rel[0, :k + 1])
        panel(img, 16, y2, f"FREEOCC prediction +{a.tau * DT:.1f} s (IoU {per_h['blend'][k]:.2f})", cp)
        panel(img, 368, y2, f"UNCERTAINTY: log-var (mean blend w={float(out['w'].mean()):.2f})", heat(((out["logvar"][0, k] + 6) / 9).clamp(0, 1).numpy()))
        draw_telemetry(img, 720, y2, roll, per_h, tick, a.ticks)
        put(img, "green = correct", (1072, y1 + 40), 0.5, C["tp"]); put(img, "red = false alarm", (1072, y1 + 64), 0.5, C["fp"])
        put(img, "blue = missed", (1072, y1 + 88), 0.5, C["fn"]); put(img, "cyan = planned ego path", (1072, y1 + 112), 0.5, C["path"])
        put(img, "white box = predicted object", (1072, y1 + 136), 0.45); put(img, f"sim t = {t0:4.2f} s", (1072, y1 + 180), 0.55)
        put(img, f"ego v={float(sc['v'][0]):.1f} m/s  yaw-rate={float(sc['w'][0]):+.2f} rad/s", (1072, y1 + 204), 0.42, C["dim"])
        wr.write(img)
        if show := a.show:
            cv2.imshow("FreeOcc sim", img)
            if cv2.waitKey(int(1000 / a.fps)) == 27: break
    cv2.imwrite(a.out.replace(".mp4", "_last.png"), img); wr.release()
    print("wrote", a.out, "| mean rolling IoU:", {n: round(float(np.mean(v)), 3) for n, v in roll.items()})


if __name__ == "__main__":
    main()
