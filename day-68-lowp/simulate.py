"""Real-time synthetic "AI in action" simulation for LOWP (OpenCV).

Layout (1280 x 720):
  row 1  : 4 camera feeds + the DEFORMABLE reader's 3D candidates projected into each image
           (dot size / brightness = attention weight; this is where the planner "looks")
  row 2  : the same feeds + the GLOBAL ray reader's cross-attention heat-map
  left   : bird's-eye view: road, lead cars, GT future (green), LOWP (cyan), dense-IPM baseline (orange),
           deformable anchors (yellow)
  right  : telemetry: per-frame L2@3s for LOWP vs dense, speed, command / alpha, CPU latency

Usage:
    python simulate.py --seed 5000 --frames 150 --out assets/lowp_simulation.gif --mp4 assets/lowp_simulation.mp4
SYNTHETIC toy world — not nuScenes / NAVSIM / Bench2Drive, not the authors' code.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from lowp.data import rig_tensors
from lowp.geometry import project_points
from lowp.world import (CAR_L, CAR_W, LANE_W, T_FUT, DT, SIM_DT, make_rig, render, road_in_ego,
                        sample_at, simulate_episode)
from train import build

W_IMG, H_IMG = 320, 160
CMD_NAME = {0: "STRAIGHT", 1: "LEFT", 2: "RIGHT"}
C_GT, C_LOWP, C_DENSE, C_ANCH = (90, 220, 90), (255, 220, 0), (0, 150, 255), (0, 230, 255)   # BGR


def put(img, txt, org, scale=0.45, col=(235, 235, 235), th=1):
    (w, h), b = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = org
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + b), (20, 20, 20), -1)   # dark label background
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def bev_to_px(xy, origin=(320, 370), scale=6.0):
    """ego (x fwd, y left) metres -> panel pixels (up = forward)."""
    xy = np.asarray(xy)
    return np.stack([origin[0] - xy[..., 1] * scale, origin[1] - xy[..., 0] * scale], -1).astype(np.int32)


def draw_box(panel, x, y, yaw, col, fill=True):
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.array([[CAR_L / 2, CAR_W / 2], [CAR_L / 2, -CAR_W / 2], [-CAR_L / 2, -CAR_W / 2], [-CAR_L / 2, CAR_W / 2]])
    pts = corners @ np.array([[c, s], [-s, c]]) + np.array([x, y])
    px = bev_to_px(pts)
    if fill:
        cv2.fillPoly(panel, [px], col)
    else:
        cv2.polylines(panel, [px], True, col, 2)


def line_plot(panel, x0, y0, w, h, series, ymax, title, labels):
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (70, 70, 70), 1)
    put(panel, title, (x0 + 4, y0 + 14), 0.42)
    for (vals, col), lab, k in zip(series, labels, range(len(series))):
        if len(vals) < 2:
            continue
        v = np.clip(np.asarray(vals), 0, ymax)
        xs = x0 + np.linspace(0, w, len(v)) if len(v) > 1 else np.array([x0])
        ys = y0 + h - v / ymax * (h - 20)
        cv2.polylines(panel, [np.stack([xs, ys], -1).astype(np.int32)], False, col, 2, cv2.LINE_AA)
        put(panel, f"{lab}: {vals[-1]:.2f}", (x0 + w - 150, y0 + 16 + 15 * k), 0.4, col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="assets/lowp_simulation.gif")
    ap.add_argument("--mp4", default="")
    ap.add_argument("--live", action="store_true", help="show an OpenCV window while rendering")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))
    hw = tuple(cfg["data"]["img_hw"])
    torch.set_num_threads(1)
    rig = make_rig(*hw)
    K, R, t = (x[None] for x in rig_tensors(rig))
    lowp = build("lowp", cfg).eval(); lowp.load_state_dict(torch.load("checkpoints/lowp_s0.pt"))
    dense = build("dense", cfg).eval(); dense.load_state_dict(torch.load("checkpoints/dense_s0.pt"))

    rng = np.random.default_rng(a.seed)
    sim = simulate_episode(rng, steps=a.start + a.frames + int(T_FUT * DT / SIM_DT) + 2)
    frames, hist = [], dict(l2_lowp=[], l2_dense=[], v=[], ms_lowp=[], ms_dense=[])
    for k in range(a.start, a.start + a.frames):
        s = sample_at(sim, k)
        cams = render(sim, k, rig, *hw, rng=np.random.default_rng(k))           # (4, H, W, 3) RGB
        img = torch.from_numpy(cams).permute(0, 3, 1, 2).float()[None] / 255.0
        st = torch.from_numpy(s["status"])[None]; cmd = torch.tensor([s["cmd"]])
        with torch.no_grad():
            t0 = time.perf_counter(); o = lowp(img, K, R, t, st, cmd); ms_l = 1000 * (time.perf_counter() - t0)
            t0 = time.perf_counter(); od = dense(img, K, R, t, st, cmd); ms_d = 1000 * (time.perf_counter() - t0)
            # deformable candidates (last layer) projected into each camera
            dd = o["def_diag"]
            cands = dd["cands"][-1][0].reshape(-1, 3)                              # (16*32, 3)
            att = dd["attn"][-1][0, :, 1:].reshape(-1)                             # (16*32,)
            grid, valid, _ = project_points(cands[None], K, R, t, hw)
            gatt = o["glob_diag"]["attn"][0].mean(0)                               # (N, Hf, Wf)
        tau_l, tau_d, gt = o["tau"][0].numpy(), od["tau"][0].numpy(), s["fut"]
        hist["l2_lowp"].append(float(np.linalg.norm(tau_l[-1] - gt[-1])))
        hist["l2_dense"].append(float(np.linalg.norm(tau_d[-1] - gt[-1])))
        hist["v"].append(float(s["status"][0] * 10)); hist["ms_lowp"].append(ms_l); hist["ms_dense"].append(ms_d)

        canvas = np.full((720, 1280, 3), 22, np.uint8)
        names = ["FRONT", "FRONT-LEFT", "FRONT-RIGHT", "BACK"]
        amax = float(att.max()) + 1e-6
        for n in range(4):
            base = cv2.resize(cv2.cvtColor(cams[n], cv2.COLOR_RGB2BGR), (W_IMG, H_IMG), interpolation=cv2.INTER_NEAREST)
            top = base.copy()
            g = grid[0, n].numpy(); v = valid[0, n].numpy()
            for (gx, gy), ok, w in zip(g, v, att.numpy()):
                if ok:
                    u = int((gx + 1) / 2 * W_IMG); vv = int((gy + 1) / 2 * H_IMG)
                    r = 1 + int(4 * w / amax)
                    cv2.circle(top, (u, vv), r, (0, int(120 + 135 * w / amax), 255), -1)
            put(top, f"{names[n]}  deformable samples", (5, 14), 0.4)
            canvas[0:H_IMG, n * W_IMG:(n + 1) * W_IMG] = top
            heat = gatt[n].numpy(); heat = heat / (gatt.max().item() + 1e-9)
            heat = cv2.resize((heat * 255).astype(np.uint8), (W_IMG, H_IMG), interpolation=cv2.INTER_LINEAR)
            hm = cv2.applyColorMap(heat, cv2.COLORMAP_MAGMA)
            bot = cv2.addWeighted(base, 0.55, hm, 0.45, 0)
            put(bot, f"{names[n]}  ray-reader attention", (5, 14), 0.4)
            canvas[H_IMG:2 * H_IMG, n * W_IMG:(n + 1) * W_IMG] = bot

        # ---- BEV panel
        bev = np.full((400, 640, 3), (40, 70, 38), np.uint8)
        poly_l = bev_to_px(road_in_ego(sim, k, LANE_W)); poly_r = bev_to_px(road_in_ego(sim, k, -LANE_W))
        cv2.fillPoly(bev, [np.concatenate([poly_l, poly_r[::-1]])], (95, 95, 100))
        cv2.polylines(bev, [poly_l, poly_r], False, (235, 235, 235), 2)
        cv2.polylines(bev, [bev_to_px(road_in_ego(sim, k, 0.0))], False, (40, 200, 230), 1)
        for (x, y, yaw) in s["obs_now"]:
            draw_box(bev, x, y, yaw, (50, 50, 210))
        draw_box(bev, 0, 0, 0, (230, 160, 60))
        for a_xy in dd["anchors"][-1][0].numpy():
            cv2.circle(bev, tuple(bev_to_px(a_xy)), 3, C_ANCH, -1)
        for traj, col in [(gt, C_GT), (tau_d, C_DENSE), (tau_l, C_LOWP)]:
            px = bev_to_px(np.vstack([[0, 0], traj]))
            cv2.polylines(bev, [px], False, col, 2, cv2.LINE_AA)
            for p in px[1:]:
                cv2.circle(bev, tuple(p), 3, col, -1)
        put(bev, "BEV  green=GT  cyan=LOWP  orange=dense-IPM  yellow=anchors", (8, 18), 0.45)
        put(bev, f"t = {k * SIM_DT:5.1f} s   command: {CMD_NAME[s['cmd']]}   alpha = {0.5 if s['cmd'] == 0 else 0.8}",
            (8, 390), 0.45)
        canvas[320:720, 0:640] = bev

        # ---- telemetry
        tel = canvas[320:720, 640:1280]
        line_plot(tel, 10, 8, 620, 120, [(hist["l2_lowp"], C_LOWP), (hist["l2_dense"], C_DENSE)], 8.0,
                  "L2 @ 3 s (m), per frame", ["LOWP", "dense"])
        line_plot(tel, 10, 138, 620, 110, [(hist["v"], (200, 200, 200))], 16.0, "ego speed (m/s)", ["v"])
        line_plot(tel, 10, 258, 620, 100, [(hist["ms_lowp"], C_LOWP), (hist["ms_dense"], C_DENSE)], 60.0,
                  "CPU latency, batch 1, 1 thread (ms)", ["LOWP", "dense"])
        put(tel, "SYNTHETIC toy world - not nuScenes/NAVSIM, not the authors' code", (10, 388), 0.42, (160, 160, 160))
        frames.append(canvas)
        if a.live:
            cv2.imshow("LOWP simulation", canvas)
            if cv2.waitKey(1) == 27:
                break

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave(a.out, [cv2.resize(f, (960, 540)) for f in rgb[::2]], duration=0.2, loop=0)
    if a.mp4:
        imageio.mimsave(a.mp4, rgb, fps=10, codec="libx264", quality=7, macro_block_size=8)
    print(f"saved {len(frames)} frames -> {a.out} {a.mp4}; mean L2@3s LOWP {np.mean(hist['l2_lowp']):.2f} "
          f"dense {np.mean(hist['l2_dense']):.2f}; median ms LOWP {np.median(hist['ms_lowp']):.1f} "
          f"dense {np.median(hist['ms_dense']):.1f}")


if __name__ == "__main__":
    main()
