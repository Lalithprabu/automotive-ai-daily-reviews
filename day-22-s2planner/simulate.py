"""
Render a live-feeling simulation of the trained S2Planner reconstruction:
for a sequence of held-out synthetic driving scenes, animate the
coarse-to-fine trajectory refinement (ego-conditioned init -> 3
camera-cross-attention decoder stages), overlaid on:
  - a bird's-eye-view (BEV) panel with lane geometry, obstacle bounding
    boxes, ground truth vs. predicted trajectory;
  - the center camera's synthetic feature view, with projected trajectory
    points and projected obstacle bounding-box wireframes (real pinhole
    projection, via src/geometry.py -- the same geometry the model's
    cross-attention uses to sample features);
  - a live telemetry strip: per-stage displacement error and the driving
    command for the current scene.

Usage: python3 simulate.py --config config.yaml --out outputs/s2planner_simulation.gif
"""
import argparse
import yaml
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from PIL import Image

from src.dataset import SyntheticDrivingSceneDataset, collate_scenes, COMMAND_NAMES
from src.geometry import make_camera_rig, project_points
from src.model import S2Planner

OBSTACLE_CLASS = 3
CENTER_CAM = 1


def obstacle_bev_boxes(landmarks_xyz, landmark_class, landmark_extent):
    boxes = []
    for i in range(landmarks_xyz.shape[0]):
        if landmark_class[i].item() != OBSTACLE_CLASS:
            continue
        cx, cy, cz = landmarks_xyz[i].tolist()
        hw, hl = landmark_extent[i].tolist()
        boxes.append((cx, cy, cz, hw, hl))
    return boxes


def box_corners_3d(cx, cy, cz, hw, hl, height=1.6):
    # axis-aligned (ego-frame) box: x forward = length, y lateral = width
    base_z = 0.0
    top_z = height
    corners = []
    for dx in (-hl, hl):
        for dy in (-hw, hw):
            corners.append((cx + dx, cy + dy, base_z))
            corners.append((cx + dx, cy + dy, top_z))
    return torch.tensor(corners, dtype=torch.float32)  # (8, 3)


def render_cam_view(cam_feat):
    # cam_feat: (C=4, S, S) -> false-color RGB
    C, S, _ = cam_feat.shape
    ground = cam_feat[0].numpy()
    left_lane = cam_feat[1].numpy()
    right_lane = cam_feat[2].numpy()
    obstacle = cam_feat[3].numpy()
    img = np.zeros((S, S, 3), dtype=np.float32)
    img[..., 0] = 0.05 + 0.9 * obstacle + 0.9 * right_lane * 0.3
    img[..., 1] = 0.08 + 0.9 * obstacle * 0.85 + 0.9 * left_lane * 0.2
    img[..., 2] = 0.12 + 0.9 * left_lane + 0.9 * right_lane
    img = np.clip(img + ground[..., None] * 0.05, 0, 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--out", type=str, default="outputs/s2planner_simulation.gif")
    ap.add_argument("--n_scenes", type=int, default=6)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(cfg["output"]["checkpoint"], map_location="cpu", weights_only=False)
    model = S2Planner(d_feat=ckpt["config"]["model"]["d_feat"], n_scales=ckpt["config"]["model"]["n_scales"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rig = make_camera_rig(image_size=cfg["data"]["image_size"])
    ds = SyntheticDrivingSceneDataset(n_samples=args.n_scenes, seed=99, image_size=cfg["data"]["image_size"])

    scenes = []
    with torch.no_grad():
        for i in range(args.n_scenes):
            item = ds[i]
            batch = collate_scenes([item])
            out = model(batch["cam_features"], batch["ego_history"], batch["command"])
            stage_ades = []
            gt = batch["future_gt"]
            for traj in out["stage_trajectories"]:
                stage_ades.append(torch.norm(traj - gt, dim=-1).mean().item())
            scenes.append({"item": item, "out": out, "stage_ades": stage_ades, "command": int(item["command"].argmax().item())})

    frames_per_scene_hold = 3
    n_stages = len(scenes[0]["out"]["stage_trajectories"])
    frame_index = []  # (scene_idx, stage_idx)
    for si in range(args.n_scenes):
        for stage in range(n_stages):
            frame_index.append((si, stage))
        for _ in range(frames_per_scene_hold):
            frame_index.append((si, n_stages - 1))

    fig = plt.figure(figsize=(13, 5.2))
    ax_bev = fig.add_subplot(1, 3, 1)
    ax_cam = fig.add_subplot(1, 3, 2)
    ax_tel = fig.add_subplot(1, 3, 3)
    fig.suptitle("S2Planner reconstruction -- coarse-to-fine trajectory refinement (synthetic scenes)", fontsize=10)

    running_ades = []

    def draw_frame(frame_i):
        si, stage = frame_index[frame_i]
        scene = scenes[si]
        item = scene["item"]
        out = scene["out"]
        gt = item["future_gt"].numpy()
        pred_stage = out["stage_trajectories"][stage][0].numpy()
        init = out["stage_trajectories"][0][0].numpy()
        cmd_name = COMMAND_NAMES[scene["command"]]

        # ---------------- BEV panel ----------------
        ax_bev.clear()
        lm_xyz = item["landmarks_xyz"]
        lm_cls = item["landmark_class"]
        lm_ext = item["landmark_extent"]
        left_lane_pts = lm_xyz[lm_cls == 1]
        right_lane_pts = lm_xyz[lm_cls == 2]
        ax_bev.plot(left_lane_pts[:, 0], left_lane_pts[:, 1], "--", color="#9aa0a6", lw=1)
        ax_bev.plot(right_lane_pts[:, 0], right_lane_pts[:, 1], "--", color="#9aa0a6", lw=1)

        for cx, cy, cz, hw, hl in obstacle_bev_boxes(lm_xyz, lm_cls, lm_ext):
            rect = patches.Rectangle((cx - hl, cy - hw), 2 * hl, 2 * hw, linewidth=1.5, edgecolor="#f4b400", facecolor="none")
            ax_bev.add_patch(rect)
            ax_bev.text(cx, cy + hw + 0.6, "VEHICLE", fontsize=6, color="#f4b400", ha="center")

        ax_bev.plot(0, 0, marker=(3, 0, -90), markersize=14, color="#1a73e8")
        ax_bev.plot(gt[:, 0], gt[:, 1], "-o", color="#0f9d58", markersize=3, label="ground truth", lw=2)
        ax_bev.plot(init[:, 0], init[:, 1], ":", color="#9aa0a6", lw=1.5, label="ego-conditioned init")
        stage_color = "#4285f4" if stage < n_stages - 1 else "#db4437"
        ax_bev.plot(pred_stage[:, 0], pred_stage[:, 1], "-o", color=stage_color, markersize=3, lw=2, label=f"S2Planner (stage {stage})")
        ax_bev.set_xlim(-10, 42)
        ax_bev.set_ylim(-14, 14)
        ax_bev.set_aspect("equal")
        ax_bev.set_title(f"BEV | scene {si+1}/{args.n_scenes} | cmd={cmd_name}", fontsize=9)
        ax_bev.legend(loc="upper left", fontsize=6, framealpha=0.85)
        ax_bev.set_xlabel("x forward (m)", fontsize=7)
        ax_bev.set_ylabel("y left (m)", fontsize=7)

        # ---------------- Camera panel ----------------
        ax_cam.clear()
        cam_img = render_cam_view(item["cam_features"][CENTER_CAM])
        S = cam_img.shape[0]
        ax_cam.imshow(cam_img, origin="upper")

        # project predicted trajectory points into the center camera
        xyz_pred = torch.cat([torch.tensor(pred_stage, dtype=torch.float32), torch.zeros(pred_stage.shape[0], 1)], dim=-1)
        uv, valid = project_points(rig, xyz_pred)
        uv_c, valid_c = uv[CENTER_CAM], valid[CENTER_CAM]
        pts = uv_c[valid_c].numpy()
        if len(pts):
            ax_cam.plot(pts[:, 0], pts[:, 1], "o-", color=stage_color, markersize=4, lw=1.5)

        for cx, cy, cz, hw, hl in obstacle_bev_boxes(lm_xyz, lm_cls, lm_ext):
            corners = box_corners_3d(cx, cy, cz, hw, hl)
            uvb, validb = project_points(rig, corners)
            uvb_c, validb_c = uvb[CENTER_CAM].numpy(), validb[CENTER_CAM].numpy()
            edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (2, 6), (6, 4), (4, 0), (1, 3), (3, 7), (7, 5), (5, 1)]
            for a, b in edges:
                if validb_c[a] and validb_c[b]:
                    ax_cam.plot([uvb_c[a, 0], uvb_c[b, 0]], [uvb_c[a, 1], uvb_c[b, 1]], color="#f4b400", lw=1)
        ax_cam.set_xlim(0, S)
        ax_cam.set_ylim(S, 0)
        ax_cam.set_title("center camera (synthetic feature view)\n+ camera-projected trajectory & bbox wireframes", fontsize=8)
        ax_cam.set_xticks([])
        ax_cam.set_yticks([])

        # ---------------- Telemetry panel ----------------
        ax_tel.clear()
        stage_ades = scene["stage_ades"]
        xs = list(range(len(stage_ades)))
        colors = ["#9aa0a6"] + ["#4285f4"] * (len(stage_ades) - 2) + ["#db4437"]
        bar_colors = [colors[i] if i <= stage else "#e0e0e0" for i in xs]
        ax_tel.bar(xs, [a if i <= stage else 0 for i, a in enumerate(stage_ades)], color=bar_colors)
        ax_tel.set_xticks(xs)
        ax_tel.set_xticklabels(["init"] + [f"L{i}" for i in range(1, len(stage_ades) - 1)] + ["final"], fontsize=7)
        ax_tel.set_ylabel("ADE vs. ground truth (m)", fontsize=7)
        ax_tel.set_ylim(0, max(stage_ades) * 1.2 + 0.1)
        ax_tel.set_title("live coarse-to-fine refinement telemetry", fontsize=8)

        cur_ade = stage_ades[stage]
        running = np.mean(running_ades + [cur_ade]) if stage == n_stages - 1 else (np.mean(running_ades) if running_ades else cur_ade)
        ax_tel.text(
            0.02, 0.98,
            f"scene ADE (this stage): {cur_ade:.2f} m\nrunning avg final ADE: {running:.2f} m\nscenes processed: {si + (1 if stage == n_stages-1 else 0)}/{args.n_scenes}",
            transform=ax_tel.transAxes, fontsize=7, va="top", family="monospace",
        )
        if stage == n_stages - 1 and frame_index[min(frame_i + 1, len(frame_index) - 1)][0] != si:
            running_ades.append(cur_ade)

        fig.tight_layout(rect=[0, 0, 1, 0.95])

    pil_frames = []
    for i in range(len(frame_index)):
        draw_frame(i)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        pil_frames.append(Image.open(buf).convert("RGB").copy())
        buf.close()

    pil_frames[0].save(
        args.out, save_all=True, append_images=pil_frames[1:], duration=450, loop=0, optimize=False
    )
    print(f"Saved simulation GIF to {args.out} ({len(pil_frames)} frames)")

    final_ades = [s["stage_ades"][-1] for s in scenes]
    init_ades = [s["stage_ades"][0] for s in scenes]
    print(f"Across {args.n_scenes} held-out scenes: mean final ADE {np.mean(final_ades):.3f}m vs. mean coarse-init ADE {np.mean(init_ades):.3f}m")


if __name__ == "__main__":
    main()
