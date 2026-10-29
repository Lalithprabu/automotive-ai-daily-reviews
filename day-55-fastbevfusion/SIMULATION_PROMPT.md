# Simulation Script / Prompt (for an image/video AI or rendering engine)
**Format:** 16:9, 1240x620, ~6 s at 6 fps, light technical-dashboard look.
**Top-left strip (camera, INPUT):** a thin 140-degree azimuth strip of coloured vertical bumps (no depth cue). Green triangles above it mark true vehicles; several unmarked bumps are poster-like decoys.
**Left (BEV scene):** top-down 48 m x 48 m grid, ego at bottom centre. Grey squares = low LiDAR returns, black squares = tall returns (vehicle-like, but poles also look like this). GROUND TRUTH = thick green rectangles. LiDAR-only detections = orange x. FastBEVFusion detections = blue diamonds with white edge. Soft cyan/blue shading = the cross-attention row of the farthest vehicle's LiDAR token (it should light up along one camera ray).
**Right:** running-F1 curves (orange LiDAR-only, purple add-fusion, blue FastBEVFusion) with a red band on frames 20-31 labelled "camera extrinsics error 3 deg"; the purple curve should sag there.
**Header:** "AI in action: LiDAR + camera fusion detects vehicles in BEV (no depth estimation)" + live frame/yaw/F1 readout. Footer: "Synthetic demo, not the paper's data".
