"""Render the "AI in action" simulation: synthetic dashcam feed + GT vs dense-BEV (old) vs LighTROcc (new) occupancy forecasts
+ live telemetry. Outputs GIF (+ optional MP4). Demo scenes are chosen by a NON-outcome rule (>=3 vehicles, >=2 moving),
never by model performance.
Usage: python simulate.py --config config.yaml --ckpt checkpoints --out assets/lightrocc_simulation.gif [--mp4] [--live]
"""
import argparse, io, os, sys, time
import numpy as np, torch, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from lightrocc import LighTROcc, DenseBEVOccForecaster
from lightrocc.data import make_scene, sample_scene
from lightrocc.geometry import voxel_centers
from lightrocc.metrics import voxel_iou, dense_instances

BG, FG, GRID = "#0d1117", "#e6edf3", "#30363d"
GT_C, OLD_C, NEW_C = "#3fb950", "#f0883e", "#58a6ff"


def box_corners(x, y, yaw, L, W):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2], [-L / 2, W / 2]])
    return np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], 1)   # (x,y)


def pick_scenes(cfg, n=3, seed0=950000):
    out, vox, s = [], voxel_centers(cfg), seed0
    while len(out) < n:
        sc = make_scene(s, cfg, vox)
        nv = int(sc["valid"].sum())
        moving = int(((sc["cen"][:, 4] - sc["cen"][:, 0]).norm(dim=-1)[sc["valid"]] > 5.0).sum())
        if nv >= 3 and moving >= 2:
            out.append(sc)
        s += 1
    return out


def draw_camera(ax, sc, cfg, out_l, t):
    c = cfg["camera"]; f = c["focal"]
    ax.set_xlim(0, c["w"]); ax.set_ylim(c["h"], 0); ax.set_facecolor("#10151c"); ax.set_xticks([]); ax.set_yticks([])
    hor = c["h"] / 2
    ax.add_patch(Rectangle((0, 0), c["w"], hor, color="#162033", lw=0))
    ax.add_patch(Polygon([[c["w"] / 2 - 1, hor], [c["w"] / 2 + 1, hor], [c["w"] + 4, c["h"]], [-4, c["h"]]], color="#1c232d", lw=0))
    for i in range(sc["valid"].shape[0]):
        if not sc["valid"][i]:
            continue
        x, y = sc["cen"][i, 0].tolist(); yaw = sc["yaw"][i, 0].item(); L, W, H = sc["size"][i].tolist()
        u = c["w"] / 2 - f * y / x; v = hor + f * (c["cam_height"] - 0.8) / x
        w_px = f * (abs(L * np.sin(yaw)) + abs(W * np.cos(yaw))) / x; h_px = f * H / x
        ax.add_patch(Rectangle((u - w_px / 2, v - h_px / 2), w_px, h_px, fc="#2d3b4f", ec=GT_C, lw=1.4, ls="--"))
    ex = out_l["exist"][0].numpy()
    for q in range(ex.shape[0]):
        if ex[q] < 0.5:
            continue
        uq, vq = out_l["uv"][0, q].numpy(); sz = out_l["size"][0, q].numpy(); dq = out_l["depth"][0, q].item()
        w_px, h_px = f * sz[1] / dq * 1.4, f * sz[2] / dq
        ax.add_patch(Rectangle((uq - w_px / 2, vq - h_px / 2), w_px, h_px, fill=False, ec=NEW_C, lw=1.8))
        ax.text(uq, vq - h_px / 2 - 0.4, f"#{q} d={dq:.0f}m p={ex[q]:.2f}", color=NEW_C, fontsize=7, ha="center")
    ax.set_title("Camera feed (synthetic) — green: truth · blue: query + lifted depth", color=FG, fontsize=9)


def draw_bev(ax, sc, cfg, probs, t, mode, out_l=None, thr=0.5):
    w = cfg["world"]
    ax.set_facecolor(BG); ax.set_xlim(16, -16); ax.set_ylim(w["x_min"], w["x_min"] + w["grid_x"])
    ax.set_aspect("equal"); ax.tick_params(colors="#8b949e", labelsize=7)
    for sp in ax.spines.values(): sp.set_color(GRID)
    bev = probs[t].reshape(w["grid_x"], w["grid_y"], w["grid_z"]).max(-1).values.numpy()
    ext = [16, -16, w["x_min"], w["x_min"] + w["grid_x"]]
    col = OLD_C if mode == "old" else NEW_C
    rgba = np.zeros(bev.shape + (4,)); rgba[..., :3] = matplotlib.colors.to_rgb(col); rgba[..., 3] = np.clip(bev, 0, 1) * 0.85
    rgba = rgba[:, ::-1]                                     # column j is y=-16+j; axis is inverted (left = +y)
    ax.imshow(rgba, extent=ext, origin="lower", aspect="equal", interpolation="nearest", zorder=2)
    for i in range(sc["valid"].shape[0]):
        if not sc["valid"][i]: continue
        L, W, _ = sc["size"][i].tolist()
        for k in range(t):                                                   # GT trail
            ax.plot(sc["cen"][i, k:k + 2, 1], sc["cen"][i, k:k + 2, 0], color=GT_C, lw=0.8, alpha=0.4, zorder=3)
        bc = box_corners(sc["cen"][i, t, 0].item(), sc["cen"][i, t, 1].item(), sc["yaw"][i, t].item(), L, W)
        ax.add_patch(Polygon(bc[:, ::-1], fill=False, ec=GT_C, lw=1.5, zorder=4))
    if mode == "new":
        ex = out_l["exist"][0].numpy()
        for q in range(ex.shape[0]):
            if ex[q] < thr: continue
            cen = out_l["centers"][0, :, q, :2].numpy()
            ax.plot(cen[:t + 1, 1], cen[:t + 1, 0], color="white", lw=1.2, zorder=5)
            ax.scatter(cen[t, 1], cen[t, 0], s=26, c=NEW_C, edgecolors="white", zorder=6)
            ax.text(cen[t, 1], cen[t, 0] + 1.8, f"#{q}", color="white", fontsize=8, ha="center", zorder=7)
    else:
        for cxy in dense_instances(probs[t], cfg):
            ax.scatter(cxy[1], cxy[0], marker="x", s=30, c="white", zorder=6)
    ax.scatter([0], [w["x_min"]], marker="^", c="white", s=40, zorder=6, clip_on=False)


def render_frame(sc, cfg, models, t, tele, fps):
    old, new = models["old"], models["new"]
    fig = plt.figure(figsize=(20, 5.6), dpi=72, facecolor=BG)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1.0], wspace=0.14, left=0.02, right=0.99, top=0.86, bottom=0.08)
    ax0, ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(4)]
    draw_camera(ax0, sc, cfg, new["inst"], t)
    draw_bev(ax1, sc, cfg, old["probs"], t, "old")
    ax1.set_title("OLD WAY · dense BEV voxel regression\n(blobs, no identity)", color=OLD_C, fontsize=9)
    draw_bev(ax2, sc, cfg, new["probs"], t, "new", new["inst"])
    ax2.set_title("NEW · LighTROcc-style instance Gaussians\n(persistent IDs, one pass)", color=NEW_C, fontsize=9)
    ax3.set_facecolor(BG); ax3.tick_params(colors="#8b949e", labelsize=7)
    for sp in ax3.spines.values(): sp.set_color(GRID)
    hz = np.arange(0, t + 1) * 0.5
    ax3.plot(hz, tele["old_iou"][: t + 1], color=OLD_C, marker="o", label="dense BEV  voxel-IoU")
    ax3.plot(hz, tele["new_iou"][: t + 1], color=NEW_C, marker="o", label="instance Gauss. voxel-IoU")
    ax3.set_xlim(-0.1, 2.1); ax3.set_ylim(0, 1); ax3.set_xlabel("forecast horizon (s)", color=FG, fontsize=8)
    ax3.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=7, loc="upper right")
    ax3.set_title("Telemetry (this scene)", color=FG, fontsize=9)
    txt = (f"t+{t*0.5:.1f}s   instances: truth {int(sc['valid'].sum())} · query {int((new['inst']['exist'][0]>0.5).sum())}"
           f" · dense blobs {len(dense_instances(old['probs'][t], cfg))}\n"
           f"params dense {fps['p_old']/1e3:.0f}K / inst {fps['p_new']/1e3:.0f}K · CPU b=1 {fps['ms_old']:.0f} ms vs {fps['ms_new']:.0f} ms")
    ax3.text(0.02, 0.04, txt, transform=ax3.transAxes, color=FG, fontsize=7.5, va="bottom")
    fig.suptitle(f"AI in action — occupancy forecasting from camera · t+{t*0.5:.1f}s  (synthetic toy scenes, NOT nuScenes)", color=FG, fontsize=12, y=0.97)
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=BG); plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml"); ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--out", default="assets/lightrocc_simulation.gif"); ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--dense-ckpt", default="dense_long.pt", help="old-way baseline checkpoint (default: the 3x-longer, stronger one)")
    ap.add_argument("--mp4", action="store_true"); ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    lt, dn = LighTROcc(cfg).eval(), DenseBEVOccForecaster(cfg).eval()
    lt.load_state_dict(torch.load(f"{a.ckpt}/lightrocc.pt")); dn.load_state_dict(torch.load(f"{a.ckpt}/{a.dense_ckpt}"))
    scenes = pick_scenes(cfg, a.scenes)
    x = scenes[0]["img"][None]
    def ms(m, **kw):
        with torch.no_grad():
            for _ in range(3): m(x, **kw)
            t0 = time.time()
            for _ in range(20): m(x, **kw)
        return (time.time() - t0) / 20 * 1000
    fps = dict(ms_new=ms(lt), ms_old=ms(dn), p_new=sum(p.numel() for p in lt.parameters()), p_old=sum(p.numel() for p in dn.parameters()))
    frames = []
    for sc in scenes:
        with torch.no_grad():
            o_new = lt(sc["img"][None], return_instances=True); o_old = dn(sc["img"][None])
        tele = dict(new_iou=voxel_iou(o_new["probs"], sc["occ"][None]).numpy(), old_iou=voxel_iou(o_old["probs"], sc["occ"][None]).numpy())
        models = dict(old=dict(probs=o_old["probs"][0]), new=dict(probs=o_new["probs"][0], inst=o_new))
        for t in range(cfg["world"]["t_future"] + 1):
            frames.append(render_frame(sc, cfg, models, t, tele, fps))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=900, loop=0)
    print(f"wrote {a.out}: {len(frames)} frames {frames[0].size}")
    if a.mp4:
        import cv2
        vw = cv2.VideoWriter(a.out.replace(".gif", ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 1.5, frames[0].size)
        for fr in frames: vw.write(cv2.cvtColor(np.array(fr), cv2.COLOR_RGB2BGR))
        vw.release(); print("wrote mp4")
    if a.live:                                                        # real-time viewer (needs a display)
        import cv2
        while True:
            for fr in frames:
                cv2.imshow("LighTROcc simulation", cv2.cvtColor(np.array(fr), cv2.COLOR_RGB2BGR))
                if cv2.waitKey(700) == 27: return


if __name__ == "__main__":
    main()
