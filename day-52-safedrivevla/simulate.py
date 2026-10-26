"""Real-time synthetic simulation: INPUT scene vs GROUND-TRUTH strict-follow rollout vs MODEL dream + decision.
Run:  python simulate.py --out assets/safedrivevla_simulation.gif        (headless GIF)
      python simulate.py --live                                        (OpenCV window, ~10 FPS)
Legend: grey = other cars (input) | GREEN ghost = ground-truth consequence of obeying the instruction
        CYAN ghost = model's dreamed ego (red outline when dreamed collision risk > 0.5) | banner = compliance mode.
"""
import argparse, time, numpy as np, torch, cv2, yaml
from src.synthetic import (DT, LANE_W, COLLIDE_GAP, NAV_NAMES, MODE_NAMES, PROFILES, FOLLOW, strict_follow_rollout, label_mode)
from src.safedrivevla import SafeDriveVLAMini

W, Hh, PANEL = 760, 560, 380
XM, YM, EGO_Y = 14.0, 6.0, 400            # px per metre (lateral, longitudinal) ; ego row
LANE_X0 = 190                               # centre of lane 1


def lane_x(l): return int(LANE_X0 + (l - 1) * LANE_W * XM)
def s_y(s): return int(EGO_Y - s * YM)


def car(img, cx, cy, color, thick=-1, label=None):
    w, h = int(1.9 * XM), int(4.5 * YM)
    cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), color, thick)
    if label: cv2.putText(img, label, (cx + w // 2 + 3, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def make_scenario(F=80):
    """Scripted world in the ego frame (ego reference speed). Returns agent lane/s0/v and instruction timeline."""
    lane = np.array([0, 0, 2, 1]); s0 = np.array([-34., 14., 22., 52.]); v = np.array([8., -2.5, -3., -1.0])
    instr = np.zeros(F, int); instr[14:44] = 1; instr[56:] = 2          # KEEP -> LEFT (unsafe) -> KEEP -> RIGHT (clear once the lane-2 car has passed)
    return lane, s0, v, instr


def scene_dict(ego_lane, nav, lane, s, v):
    return dict(ego_lane=np.array([ego_lane]), nav=np.array([nav]), lane=lane[None], s=s[None], v=v[None],
                valid=np.ones((1, len(lane)), np.float32))


def tensors(sc):
    ag = np.concatenate([np.eye(3, dtype=np.float32)[sc["lane"]], (sc["s"] / 30)[..., None], (sc["v"] / 6)[..., None],
                         sc["valid"][..., None]], -1).astype(np.float32)
    return torch.as_tensor(ag), torch.as_tensor(np.eye(3, dtype=np.float32)[sc["ego_lane"]]), torch.as_tensor(sc["nav"])


def draw_frame(model, f, ego_lane_f, ego_lane_i, nav, lane, s_now, v, hist):
    img = np.full((Hh, W, 3), 22, np.uint8)
    for l in range(4):                                                    # road
        x = int(LANE_X0 + (l - 1.5) * LANE_W * XM)
        cv2.line(img, (x, 0), (x, Hh), (90, 90, 90), 1, cv2.LINE_AA)
    sc = scene_dict(ego_lane_i, nav, lane, s_now, v)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(*tensors(sc))
    ms = (time.perf_counter() - t0) * 1e3
    coll, gap, off = strict_follow_rollout(sc)
    gt_mode = int(label_mode(coll, off)[0]); mode = int(out["mode"][0]); risk = out["risk"][0].sigmoid().numpy()
    target = ego_lane_i + [0, -1, 1][nav]
    # 1) INPUT: other cars (grey) + their constant-velocity ghosts at t = 2 s, 4 s
    for k in range(len(lane)):
        car(img, lane_x(lane[k]), s_y(s_now[k]), (150, 150, 150), label=f"{v[k]:+.1f}m/s")
        for tt in (4, 8):
            cv2.line(img, (lane_x(lane[k]), s_y(s_now[k])), (lane_x(lane[k]), s_y(s_now[k] + v[k] * DT * tt)), (70, 70, 70), 1)
    # 2) GROUND TRUTH: strict-follow ghosts (green; red X where the true rollout collides)
    for t in range(0, 8, 2):
        occ = target if (t >= 2 and 0 <= target <= 2) else ego_lane_i
        cv2.rectangle(img, (lane_x(occ) - 13, EGO_Y - 14), (lane_x(occ) + 13, EGO_Y + 14), (60, 200, 90), 1)
        if coll[0, t]:
            cv2.putText(img, "X", (lane_x(max(0, min(2, occ))) - 6, EGO_Y - 18 - 4 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 2)
    # 3) MODEL DREAM: cyan ghost path of the strict-follow ego (red when dreamed risk high)
    pts = []
    for t in range(8):
        x = lane_x(ego_lane_i) + (lane_x(target) - lane_x(ego_lane_i)) * (PROFILES[FOLLOW][t] if 0 <= target <= 2 else 0)
        pts.append((int(x), int(EGO_Y - 16 * t)))
        col = (60, 60, 255) if risk[t] > 0.5 else (255, 220, 40)
        cv2.circle(img, pts[-1], 5, col, -1 if risk[t] > 0.5 else 1, cv2.LINE_AA)
    for a, b in zip(pts[:-1], pts[1:]): cv2.line(img, a, b, (255, 220, 40), 1, cv2.LINE_AA)
    # ego car at its (possibly changing) lateral position
    ex = int(LANE_X0 + (ego_lane_f - 1) * LANE_W * XM)
    car(img, ex, EGO_Y, (255, 160, 40), label="EGO")
    # banner
    ok = (mode == gt_mode)
    bcol = {0: (60, 180, 80), 1: (30, 170, 230), 2: (60, 60, 230)}[mode]
    cv2.rectangle(img, (0, 0), (W, 46), bcol, -1)
    cv2.putText(img, f"INSTRUCTION: {NAV_NAMES[nav]}   ->   MODE: {MODE_NAMES[mode]}   (GT: {MODE_NAMES[gt_mode]}{'' if ok else ' MISMATCH'})",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    # telemetry panel: dreamed risk bars + probabilities
    x0 = W - PANEL + 10
    cv2.putText(img, "DREAMED COLLISION RISK (strict follow)", (x0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    for t in range(8):
        h = int(risk[t] * 90); xx = x0 + t * 42
        cv2.rectangle(img, (xx, 190 - h), (xx + 30, 190), (60, 60, 255) if risk[t] > 0.5 else (255, 220, 40), -1)
        cv2.rectangle(img, (xx, 190 - int(coll[0, t] * 90)), (xx + 30, 190), (60, 200, 90), 1)       # green outline = ground truth
        cv2.putText(img, f"{(t+1)*DT:.1f}s", (xx, 208), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    pr = torch.softmax(out["mode_logits"], -1)[0].numpy()
    for i, nm in enumerate(MODE_NAMES):
        y = 250 + i * 34
        cv2.putText(img, nm, (x0, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.rectangle(img, (x0 + 90, y), (x0 + 90 + int(pr[i] * 240), y + 18), {0: (60, 180, 80), 1: (30, 170, 230), 2: (60, 60, 230)}[i], -1)
    hist.append(float(risk.max()))
    for i in range(1, len(hist)):                                              # running max-risk sparkline
        cv2.line(img, (x0 + (i - 1) * 4, 450 - int(hist[i - 1] * 60)), (x0 + i * 4, 450 - int(hist[i] * 60)), (255, 220, 40), 1)
    cv2.putText(img, f"t={f:03d}  min-gap(GT)={gap[0].min()*20:4.1f} m  infer={ms:4.1f} ms", (x0, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, "green = GT consequence | cyan/red = model dream", (x0, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(img, "Synthetic demo - reconstruction of arXiv:2610.20609, not the authors' model/numbers", (10, Hh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="assets/safedrivevla_mini.pt")
    ap.add_argument("--out", default="assets/safedrivevla_simulation.gif"); ap.add_argument("--live", action="store_true")
    ap.add_argument("--frames", type=int, default=80); a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))
    model = SafeDriveVLAMini(cfg["dim"], cfg["heads"], cfg["enc_layers"], cfg["horizon"]); model.load_state_dict(torch.load(a.ckpt)); model.eval()
    lane, s0, v, instr = make_scenario(a.frames)
    ego_lane_i, ego_lane_f, move_start, move_to, frames, hist, prev_nav = 1, 1.0, None, None, [], [], 0
    for f in range(a.frames):
        s_now = s0 + v * f * 0.1 * 2.0                                         # world advances 0.2 s per frame
        nav = int(instr[f])
        if nav != prev_nav and nav != 0 and move_start is None:                # instruction arrives: ask the model
            sc = scene_dict(ego_lane_i, nav, lane, s_now, v)
            with torch.no_grad(): m = int(model(*tensors(sc))["mode"][0])
            tgt = ego_lane_i + [0, -1, 1][nav]
            if m == FOLLOW and 0 <= tgt <= 2: move_start, move_to = f, tgt
        prev_nav = nav
        if move_start is not None:                                             # execute the FOLLOW lane change over 8 frames
            p = min(1.0, (f - move_start) / 8.0); ego_lane_f = ego_lane_i + (move_to - ego_lane_i) * p
            if p >= 1.0: ego_lane_i, ego_lane_f, move_start = move_to, float(move_to), None
        img = draw_frame(model, f, ego_lane_f, ego_lane_i, nav, lane, s_now, v, hist)
        if a.live:
            cv2.imshow("SafeDriveVLA-Mini", img)
            if cv2.waitKey(100) == 27: break
        else:
            frames.append(img[..., ::-1])
    if not a.live:
        import imageio
        imageio.mimsave(a.out, frames, duration=0.1, loop=0); print("wrote", a.out, len(frames), "frames")


if __name__ == "__main__":
    main()
