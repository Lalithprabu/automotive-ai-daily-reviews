"""Real-time synthetic simulation: ego-rider view + BEV overlay of INPUT vs GROUND TRUTH vs PREDICTION.

Left-top  : pseudo first-person rider view (agents as perspective boxes, ego-sway horizon tilt).
Right-top : bird's-eye view in the current ego frame (history grey, GT green, naive CV red, model blue).
Bottom-L  : maneuver probabilities as the clip is progressively observed (anticipation).
Bottom-R  : live telemetry -- running ADE (naive CV vs model) and ground-truth-class confidence.
Usage: python simulate.py --out assets/demo.mp4 [--show] [--scenes 3]
"""
import argparse, math, os
import cv2, numpy as np, torch
from models.motoanticipator import MotoAnticipator
from src.synthetic import make_batch, future_target, to_frame, T_OBS, H, K, DT, MANEUVERS
from train import apparent_cv

W, Hh = 1280, 720
BG, GRID = (24, 20, 18), (60, 52, 46)
COL = dict(hist=(150, 150, 150), gt=(80, 220, 110), cv=(70, 70, 235), pred=(250, 160, 60), ego=(60, 200, 255))
MANCOL = [(180, 180, 180), (250, 160, 60), (80, 220, 110), (70, 70, 235)]


def put(img, s, org, sc=0.55, col=(230, 230, 230), th=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def bev_px(p, cx, cy, scale):  # ego frame: x forward -> up, y left -> left
    return int(cx - p[1] * scale), int(cy - p[0] * scale)


def draw_bev(img, b, i, L, pred, cv_app, tgt, box):
    x0, y0, w, h = box
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), BG, -1)
    cx, cy, sc = x0 + w // 2, y0 + int(h * 0.80), 5.0
    for r in range(10, 70, 10):
        cv2.circle(img, (cx, cy), int(r * sc), GRID, 1)
    cv2.line(img, (cx, y0), (cx, y0 + h), GRID, 1)
    fr = L - 1
    hist_w = b["agents_w"][i, : L]                                   # (L,K,2) world
    for k in range(K):
        hh = to_frame({kk: v[i:i + 1] for kk, v in b.items()}, hist_w[:, k][None], torch.tensor([fr]))[0]
        pts = np.array([bev_px(p, cx, cy, sc) for p in hh.numpy()], np.int32)
        cv2.polylines(img, [pts], False, COL["hist"], 2, cv2.LINE_AA)
        for key, arr in (("gt", tgt), ("cv", cv_app), ("pred", pred)):
            q = np.array([bev_px(p, cx, cy, sc) for p in arr[k].numpy()], np.int32)
            if key == "cv":
                for a, c in zip(q[:-1:2], q[1::2]): cv2.line(img, tuple(a), tuple(c), COL[key], 2, cv2.LINE_AA)
            else:
                cv2.polylines(img, [q], False, COL[key], 3 if key == "pred" else 2, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[-1]), 6, (255, 255, 255), -1, cv2.LINE_AA)
        # bounding box around the agent's current position
        px = pts[-1]; cv2.rectangle(img, (px[0] - 9, px[1] - 9), (px[0] + 9, px[1] + 9), (255, 255, 255), 1)
    ep = b["ego_pos"][i, :L] ; ehist = to_frame({kk: v[i:i + 1] for kk, v in b.items()}, ep[None], torch.tensor([fr]))[0]
    cv2.polylines(img, [np.array([bev_px(p, cx, cy, sc) for p in ehist.numpy()], np.int32)], False, COL["ego"], 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 8, COL["ego"], -1, cv2.LINE_AA)
    put(img, "BEV (ego frame, 10 m rings)", (x0 + 10, y0 + 22), 0.6)
    for j, (n_, c) in enumerate([("history", COL["hist"]), ("ground truth", COL["gt"]), ("naive CV (apparent)", COL["cv"]), ("model", COL["pred"])]):
        cv2.line(img, (x0 + 10, y0 + 44 + 20 * j), (x0 + 38, y0 + 44 + 20 * j), c, 3); put(img, n_, (x0 + 46, y0 + 49 + 20 * j), 0.5)


def draw_view(img, b, i, L, box):
    x0, y0, w, h = box
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (60, 45, 35), -1)
    w_rate = float(b["ego"][i, L - 1, 1]); tilt = int(np.clip(w_rate, -1, 1) * 60)   # horizon shifts with yaw-rate
    hor = y0 + h // 2 - 20
    cv2.rectangle(img, (x0, hor), (x0 + w, y0 + h), (75, 85, 80), -1)
    cv2.line(img, (x0, hor), (x0 + w, hor), (140, 140, 140), 1)
    f = 260.0
    for k in range(K):
        x, y = b["agents_ego"][i, L - 1, k].tolist()
        if x < 2.5: continue
        u = int(x0 + w / 2 - f * y / x) - tilt; bw = int(max(8, 450 / x)); bh = int(max(10, 350 / x))
        v = int(hor + 700 / x)
        cv2.rectangle(img, (u - bw // 2, v - bh), (u + bw // 2, v), (255, 255, 255), 2)
        put(img, f"{x:4.1f} m", (u - bw // 2, v - bh - 4), 0.4)
    put(img, f"Rider view  v={float(b['ego'][i, L - 1, 0]):.1f} m/s  yaw-rate={w_rate:+.2f} rad/s", (x0 + 10, y0 + 22), 0.55)


def draw_probs(img, probs, gt, frac, box):
    x0, y0, w, h = box
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), BG, -1)
    put(img, f"Maneuver anticipation  (observed {int(frac * 100)}% of clip)", (x0 + 10, y0 + 24), 0.6)
    for j, n_ in enumerate(MANEUVERS):
        yy = y0 + 50 + j * 38
        put(img, n_, (x0 + 10, yy + 16), 0.55); cv2.rectangle(img, (x0 + 90, yy), (x0 + 90 + int(probs[j] * (w - 160)), yy + 22), MANCOL[j], -1)
        put(img, f"{probs[j] * 100:4.0f}%", (x0 + w - 62, yy + 17), 0.5)
        if j == gt: cv2.rectangle(img, (x0 + 88, yy - 2), (x0 + w - 70, yy + 24), (255, 255, 255), 1)
    put(img, "white box = ground-truth maneuver", (x0 + 10, y0 + h - 34), 0.45, (180, 180, 180))


def draw_plot(img, series, box, title, ymax):
    x0, y0, w, h = box
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), BG, -1)
    put(img, title, (x0 + 10, y0 + 24), 0.55)
    for name, vals, col in series:
        if len(vals) < 2: continue
        pts = np.array([[x0 + 10 + int(j / (T_OBS) * (w - 20)), y0 + h - 12 - int(min(v, ymax) / ymax * (h - 50))] for j, v in enumerate(vals)], np.int32)
        cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
    for j, (name, _, col) in enumerate(series):
        cv2.line(img, (x0 + w - 190, y0 + 18 + 16 * j), (x0 + w - 170, y0 + 18 + 16 * j), col, 3); put(img, name, (x0 + w - 164, y0 + 22 + 16 * j), 0.42)


def render(model, seed, scenes, writer=None, show=False, last_png=None):
    g = torch.Generator().manual_seed(seed)
    pool = make_batch(400, g)
    stress = pool["ego"][..., 1].abs().mean(1)
    idx = []
    for m in (1, 3, 2, 0)[:scenes]:                                   # a turn, a stop, the other turn, straight
        cand = ((pool["label"] == m)).nonzero().flatten()
        idx.append(int(cand[stress[cand].argmax()]))                  # pick the hardest (most ego-motion) example
    for i in idx:
        gt = int(pool["label"][i]); ade_cv, ade_m, conf = [], [], []
        for L in list(range(6, T_OBS + 1)) + [T_OBS] * 8:
            Lt = torch.tensor([L])
            sub = {k: v[i:i + 1] for k, v in pool.items()}
            with torch.no_grad():
                lg, pred, _ = model(sub["ego"], sub["agents_ego"], Lt)
            tgt = future_target(sub, Lt)[0]; cva = apparent_cv(sub, Lt, model)[0]
            probs = torch.softmax(lg[0, L - 1], -1).numpy()
            if L <= T_OBS and (not ade_m or len(ade_m) < L - 5):
                ade_m.append(float((pred[0] - tgt).norm(dim=-1).mean())); ade_cv.append(float((cva - tgt).norm(dim=-1).mean())); conf.append(float(probs[gt]))
            img = np.full((Hh, W, 3), 12, np.uint8)
            draw_view(img, sub, 0, L, (0, 0, 640, 360))
            draw_bev(img, sub, 0, L, pred[0], cva, tgt, (640, 0, 640, 360))
            draw_probs(img, probs, gt, L / T_OBS, (0, 360, 640, 360))
            draw_plot(img, [("model ADE (m)", ade_m, COL["pred"]), ("naive CV ADE (m)", ade_cv, COL["cv"]), ("P(true maneuver) x10", [c * 10 for c in conf], COL["gt"])],
                      (640, 360, 640, 360), "Live telemetry vs observation progress", 10.0)
            put(img, f"SYNTHETIC DATA - toy model, not the MotoVerse benchmark | scene {idx.index(i) + 1}/{len(idx)} | true maneuver: {MANEUVERS[gt]}", (10, Hh - 8), 0.5, (140, 200, 255))
            if writer is not None: writer.write(img)
            if last_png is not None: cv2.imwrite(last_png, img)
            if show:
                cv2.imshow("MotoAnticipator", img)
                if cv2.waitKey(int(1000 / 6)) == 27: return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/model.pt"); ap.add_argument("--out", default="assets/demo.mp4")
    ap.add_argument("--show", action="store_true"); ap.add_argument("--scenes", type=int, default=3); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fps", type=int, default=6)
    a = ap.parse_args()
    model = MotoAnticipator(); model.load_state_dict(torch.load(a.ckpt)); model.eval()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, Hh))
    render(model, a.seed, a.scenes, wr, a.show, a.out.replace(".mp4", "_last.png"))
    wr.release(); print("wrote", a.out)
