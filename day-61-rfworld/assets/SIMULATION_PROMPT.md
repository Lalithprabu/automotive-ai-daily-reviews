# Simulation script / prompt (image-video AI or rendering engine)
16:9 (1240x640), ~6 s at 3 fps, light technical dashboard, title "AI in action: a rollout-free world model scores 256 candidate trajectories in ONE forward pass".
**Left (INPUT -> PREDICTION -> GROUND TRUTH), 64x64 ego-centric bird's-eye "camera + LiDAR" feed (forward = up):** green drivable road band with a cyan centre line, dark off-road; moving vehicles as orange blobs; a blue ego triangle at the bottom centre.
 - INPUT = the feed itself.
 - PREDICTION = a fan of 256 thin arcs from the ego, coloured by the model's predicted driving score (viridis: purple = bad, yellow = good); the model's pick is a thick BLUE DASHED path with dots.
 - GROUND TRUTH = a thick SOLID GREEN arc (best trajectory under the privileged scorer) and RED RINGS marking where each vehicle really is at +1.5 s (dotted) and +3 s (solid).
 Arcs that cross a ring or leave the road must be dark purple; the blue pick should bend away from vehicles that drift into the lane.
**Right top:** scatter of predicted vs true score for all 256 anchors with a dashed y=x line; blue dot = model's pick, green dot = true best.
**Right middle:** ego speed (m/s) and distance to nearest vehicle vs frame; black vertical line on collision.
**Right bottom:** mean |predicted - true| score error and selection regret vs frame.
Footer: "Synthetic demo - toy re-implementation, NOT NAVSIM, NOT the paper's results".
