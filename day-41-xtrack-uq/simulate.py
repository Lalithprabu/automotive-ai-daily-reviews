"""Synthetic 'AI in action' simulation: OpenCV-rendered side-scrolling highway feed with
  INPUT   : observed 3 s history trail (grey) + neighbour cars
  TRUTH   : ground-truth 5 s future (green)
  PRED    : ensemble mean trajectory (blue) with the raw 95% uncertainty region (red outline, overconfident)
            and the conformal-calibrated 95% region (cyan, filled)
  TELEMETRY: live coverage tally raw vs conformal, per-horizon sigma, scene verdict.
Uses the trained checkpoint. Output: assets/simulation.gif (and mp4 if --mp4)."""
import sys, argparse, json, math, numpy as np, cv2, torch
from PIL import Image
sys.path.insert(0, "src")
from xtrack_uq.data import make_dataset, to_tensors
from xtrack_uq.uncertainty import propagate_samples, mahalanobis
from xtrack_uq.inference import load_ensemble, fit_calibrator

W, H = 1100, 520; ROAD_Y0, ROAD_Y1 = 60, 330; SX, SY = 6.0, 22.0   # px per metre (anisotropic: lateral exaggerated)
BG = (24, 26, 30)
def w2p(x, y, ox=60, oy=195): return int(ox + x * SX), int(oy + y * SY)
def ellipse_pts(c, cov, k, n=40):
    w, V = np.linalg.eigh(cov); w = np.clip(w, 1e-9, None); th = np.linspace(0, 2 * np.pi, n)
    return c + (V @ (np.sqrt(w)[:, None] * k * np.stack([np.cos(th), np.sin(th)])) ).T
def poly(img, pts, color, fill=False, alpha=0.3, thick=2):
    p = np.array([w2p(*q) for q in pts], np.int32)
    if fill:
        ov = img.copy(); cv2.fillPoly(ov, [p], color); cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [p], True, color, thick, cv2.LINE_AA)
def text(img, s, xy, scale=0.5, color=(220, 220, 220), thick=1): cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def render(scene, pred, cov_cp, cov_raw, step, tally, i, n_scenes, cfg, v_hist_pos):
    img = np.full((H, W, 3), BG, np.uint8)
    cv2.rectangle(img, (0, ROAD_Y0), (W, ROAD_Y1), (44, 46, 52), -1)
    for yy in (-5.5, -1.8, 1.8, 5.5):
        p0, p1 = w2p(-5, yy), w2p(170, yy)
        for xs in range(p0[0], p1[0], 40): cv2.line(img, (xs, p0[1]), (xs + 20, p0[1]), (90, 92, 98), 1)
    mean, gt = pred["mean"], scene["pos"]
    hist = v_hist_pos            # observed past positions (x<=0)
    for a, b in zip(hist[:-1], hist[1:]): cv2.line(img, w2p(*a), w2p(*b), (150, 150, 150), 3, cv2.LINE_AA)
    text(img, "INPUT: 3 s history", (w2p(*hist[0])[0], w2p(*hist[0])[1] - 14), 0.45, (170, 170, 170))
    k = step                    # reveal horizon steps 0..k
    # uncertainty regions (every 3rd step), raw first (red outline), calibrated (cyan fill)
    for t in range(2, k + 1, 3):
        poly(img, ellipse_pts(mean[t], cov_cp[t], 1.0), (255, 200, 40), fill=True, alpha=0.18, thick=1)    # BGR cyan-ish
        poly(img, ellipse_pts(mean[t], cov_raw[t] * 5.99, 1.0), (60, 60, 230), thick=1)                    # raw 95% (chi2_2)
    for t in range(1, k + 1):
        cv2.line(img, w2p(*gt[t - 1]), w2p(*gt[t]), (90, 220, 120), 3, cv2.LINE_AA)
        cv2.line(img, w2p(*mean[t - 1]), w2p(*mean[t]), (240, 150, 40), 2, cv2.LINE_AA)
    # ego glyph at ground truth position
    cx, cy = w2p(*gt[k]); cv2.rectangle(img, (cx - 14, cy - 6), (cx + 14, cy + 6), (90, 220, 120), -1)
    cx, cy = w2p(*mean[k]); cv2.rectangle(img, (cx - 14, cy - 6), (cx + 14, cy + 6), (240, 150, 40), 2)
    # telemetry panel
    cv2.rectangle(img, (0, 350), (W, H), (18, 20, 24), -1)
    r_raw = float(mahalanobis(torch.tensor(gt[k] - mean[k]).float(), torch.tensor(cov_raw[k]).float()))
    r_cp = float(mahalanobis(torch.tensor(gt[k] - mean[k]).float(), torch.tensor(cov_cp[k]).float()))
    in_raw, in_cp = r_raw <= math.sqrt(5.99), r_cp <= 1.0
    text(img, f"Scene {i+1}/{n_scenes}   horizon {(k+1)*cfg['dt']:.1f}s / {cfg['t_pred']*cfg['dt']:.0f}s   speed {scene['v0']*3.6:.0f} km/h", (20, 378), 0.55)
    text(img, f"sigma_x {math.sqrt(cov_raw[k][0,0]):.2f} m  sigma_y {math.sqrt(cov_raw[k][1,1]):.2f} m   position error {np.linalg.norm(gt[k]-mean[k]):.2f} m", (20, 405), 0.5, (200, 200, 200))
    text(img, f"RAW 95% region : truth {'INSIDE' if in_raw else 'MISS'}", (20, 440), 0.55, (90, 220, 120) if in_raw else (80, 80, 240))
    text(img, f"CONFORMAL 95%  : truth {'INSIDE' if in_cp else 'MISS'}", (20, 468), 0.55, (90, 220, 120) if in_cp else (80, 80, 240))
    ntot = max(tally["n"], 1)
    text(img, f"running coverage (all scenes so far, every horizon step):  raw {100*tally['raw']/ntot:5.1f}%   conformal {100*tally['cp']/ntot:5.1f}%   target 95%", (420, 440), 0.5)
    for j, (lab, val, col) in enumerate([("raw", tally["raw"] / ntot, (60, 60, 230)), ("conformal", tally["cp"] / ntot, (255, 200, 40))]):
        y0 = 452 + j * 22; cv2.rectangle(img, (420, y0), (420 + int(560 * val), y0 + 14), col, -1); cv2.rectangle(img, (420, y0), (980, y0 + 14), (110, 110, 110), 1)
    cv2.line(img, (420 + int(560 * 0.95), 450), (420 + int(560 * 0.95), 498), (255, 255, 255), 1)
    # legend
    for j, (lab, col) in enumerate([("ground truth", (90, 220, 120)), ("ensemble mean", (240, 150, 40)), ("raw 95% (uncalibrated)", (60, 60, 230)), ("conformal 95%", (255, 200, 40))]):
        cv2.rectangle(img, (20 + j * 255, 20), (36 + j * 255, 34), col, -1); text(img, lab, (44 + j * 255, 33), 0.5)
    text(img, "SYNTHETIC DATA - X-TRACK-UQ reconstruction (not highD)", (W - 470, 345), 0.45, (130, 130, 130))
    return img

def main(n_scenes=4, out="assets/simulation.gif", mp4=False):
    cfg, ens = load_ensemble(); cp = fit_calibrator(cfg, ens)
    test = make_dataset(60, cfg, 999); T = to_tensors(test); o = propagate_samples(ens, T, cfg["n_samples"], 1)
    mean = o["mean_traj"].numpy(); cov = o["cov_total"].numpy(); covc = cp.calibrate(o["cov_total"]).numpy()
    q2 = cp.q.numpy() ** 2
    # pick scenes with a visible manoeuvre
    mag = np.abs(test["fut"][:, :, 1]).sum(1) + np.abs(test["fut"][:, :, 0]).max(1) * .1
    picks = list(np.argsort(-mag)[:2]) + list(np.argsort(mag)[:2]); frames = []; tally = dict(n=0, raw=0, cp=0)
    # tally from ALL 60 scenes so the running number is statistically meaningful; revealed progressively
    r_all_raw = mahalanobis(T["pos"] - o["mean_traj"], o["cov_total"]).numpy(); r_all_cp = mahalanobis(T["pos"] - o["mean_traj"], torch.tensor(covc)).numpy()
    ok_raw = r_all_raw <= math.sqrt(5.99); ok_cp = r_all_cp <= 1.0
    for si, idx in enumerate(picks[:n_scenes]):
        sc = dict(pos=test["pos"][idx], v0=float(test["v0"][idx]))
        dt = cfg["dt"]; v = float(test["v0"][idx])
        # reconstruct observed history path (x<=0) for display
        ax, yr = test["hist"][idx, 0, :, 0], test["hist"][idx, 0, :, 1]
        hp = [np.zeros(2)]; vv = v; psi = float(test["psi0"][idx]); 
        # walk backwards: undo last motion steps
        x = y = 0.0
        for t in range(cfg["t_obs"] - 1, -1, -1):
            x -= vv * math.cos(psi) * dt; y -= vv * math.sin(psi) * dt; psi -= yr[t] * dt; vv = max(vv - ax[t] * dt, 0)
            hp.append(np.array([x, y]))
        hp = np.array(hp[::-1]); pred = dict(mean=mean[idx])
        for k in list(range(0, cfg["t_pred"], 2)) + [cfg["t_pred"] - 1] * 4:
            sl = slice(0, 15 * (si + 1)); n_full = ok_raw[sl].size
            tally = dict(n=n_full, raw=ok_raw[sl].sum(), cp=ok_cp[sl].sum())
            frames.append(render(sc, pred, covc[idx], cov[idx], k, tally, si, n_scenes, cfg, hp))
    pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    pil[0].save(out, save_all=True, append_images=pil[1:], duration=170, loop=0, optimize=True); print("wrote", out, len(pil), "frames", pil[0].size)
    if mp4:
        vw = cv2.VideoWriter(out.replace(".gif", ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 6, (W, H)); [vw.write(f) for f in frames]; vw.release()
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scenes", type=int, default=4); ap.add_argument("--out", default="assets/simulation.gif"); ap.add_argument("--mp4", action="store_true")
    a = ap.parse_args(); main(a.scenes, a.out, a.mp4)
